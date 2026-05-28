"""Lazy time-series loader for EMSES grid HDF5 files.

:class:`GridDataSeries` memory-maps a sequence of ``{name}00_0000.h5``
files and produces :class:`~emout.core.data.data.Data` slices on demand.
"""

import warnings
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from emout.local_data_policy import (
    require_local_data_access,
    is_local_data_access_disabled,
    normalize_local_data_policy,
)
from emout.plot.animation_plot import ANIMATER_PLOT_MODE, FrameUpdater
from emout.utils import DataFileInfo, UnitTranslator

from ._base import _REMOTE_PLOT_HANDLED
from .data import Data3d, Data4d
from .factory import data_from_array, data_from_payload
from .selectors import (
    compose_selector as _compose_selector,
    normalize_selector as _normalize_selector,
    selector_length as _selector_length,
    selector_positions as _selector_positions,
    selector_to_metadata_slice as _selector_to_metadata_slice,
)
from .surface_roi import is_spatial_3d_selection, plot_surfaces_roi_selectors, surface_roi_selectors

_MATERIALIZED_UNSET = object()


def _materialize_nested(value):
    if isinstance(value, (GridDataSelection, GridDataMean)):
        return value.materialize()
    if isinstance(value, dict):
        return {key: _materialize_nested(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_materialize_nested(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_materialize_nested(val) for val in value)
    return value


class GridDataSeries:
    """Manage 3-D time-series grid data.

    Attributes
    ----------
    datafile : DataFileInfo
        Data file metadata.
    h5 : h5py.File
        HDF5 file handle.
    group : h5py.Datasets
        HDF5 dataset group.
    name : str
        Dataset name.
    """

    def __init__(
        self,
        filename: PathLike,
        name: str,
        tunit: UnitTranslator = None,
        axisunit: UnitTranslator = None,
        valunit: UnitTranslator = None,
        local_data_policy: Union[str, None] = None,
    ):
        """Create a 3-D time-series data object.

        Parameters
        ----------
        filename : str or Path
            Path to the HDF5 file.
        name : str
            Dataset name.
        """
        self.datafile = DataFileInfo(filename)
        self.h5 = h5py.File(str(filename), "r")
        self.group = self.h5[list(self.h5.keys())[0]]
        self._index2key = {int(key): key for key in self.group.keys()}
        first_key = self._index2key[min(self._index2key.keys())]
        self._grid_shape = tuple(self.group[first_key].shape)
        self.tunit = tunit
        self.axisunit = axisunit
        self.valunit = valunit
        self._local_data_policy = normalize_local_data_policy(local_data_policy)

        self.name = name
        self._emout_dir = None  # set by GridLoader to enable remote rendering
        self._emout_open_kwargs = None
        self._article_recorder = None
        self._article_source_shape = self.shape

    def __repr__(self) -> str:
        return f"<GridDataSeries: name={self.name!r}, timesteps={len(self)}, file={self.filename.name}>"

    @property
    def trange(self) -> List[int]:
        """Return the sorted list of available time-step indices.

        Returns
        -------
        list of int
            Available time-step indices in ascending order.
        """
        return sorted(self._index2key.keys())

    def close(self) -> None:
        """Close the HDF5 file."""
        self.h5.close()

    def __enter__(self) -> "GridDataSeries":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the context manager and close the HDF5 file."""
        self.close()

    def time_series(self, x, y, z) -> np.ndarray:
        """Return the time series for the specified spatial range.

        Parameters
        ----------
        x : int or slice
            X coordinate or range.
        y : int or slice
            Y coordinate or range.
        z : int or slice
            Z coordinate or range.

        Returns
        -------
        numpy.ndarray
            Time-series data for the specified range.
        """
        self._require_local_data_access("read time series", f"{self.name}.time_series")
        series = []
        indexes = sorted(self._index2key.keys())
        for index in indexes:
            series.append(self._read_selection(index, (z, y, x)))
        return np.array(series)

    @property
    def filename(self) -> Path:
        """Return the file name.

        Returns
        -------
        Path
            File name.
        """
        return self.datafile.filename

    @property
    def directory(self) -> Path:
        """Return the directory name.

        Returns
        -------
        Path
            Directory path.
        """
        return self.datafile.directory

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Return the spatial grid shape as ``(z, y, x)``."""
        return self._grid_shape

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Return the logical 4-D shape as ``(t, z, y, x)`` without reading data."""
        return (len(self), *self.grid_shape)

    @property
    def ndim(self) -> int:
        """Return the number of logical dimensions."""
        return 4

    @property
    def lazy(self) -> "GridDataSelection":
        """Return a lazy 4-D selector for staged indexing.

        ``series[:]`` also returns this selector. Use ``series.lazy`` when
        you want to make the lazy intent explicit before additional spatial
        slicing, e.g.
        ``data.phisp.lazy[:][:, 1, 1, 1]`` or
        ``data.phisp.lazy[:].select_space(1, 1, 1)``.
        """
        return GridDataSelection(self)

    def _read_selection(self, index: int, spatial_item) -> np.ndarray:
        """Read only the requested spatial subset for one timestep."""
        self._require_local_data_access("read field data", f"{self.name}[{index}]")
        if index not in self._index2key:
            raise IndexError(f"Time index {index} does not exist. Available indices: {sorted(self._index2key.keys())}")

        key = self._index2key[index]
        return np.array(self.group[key][spatial_item])

    def _require_local_data_access(self, operation: str, target: Union[str, None] = None) -> None:
        require_local_data_access(self._local_data_policy, operation, target)

    def _create_data_with_index(self, index: int) -> Data3d:
        """Create a Data3d snapshot for the given time index.

        Parameters
        ----------
        index : int
            Time index.

        Returns
        -------
        Data3d
            3-D data at the specified timestep.

        Raises
        ------
        IndexError
            If the specified time index does not exist.
        """
        if index < 0:
            index = len(self) + index
        if index not in self._index2key:
            raise IndexError(f"Time index {index} does not exist. Available indices: {sorted(self._index2key.keys())}")

        axisunits = [self.tunit] + [self.axisunit] * 3

        data = Data3d(
            self._read_selection(index, (slice(None), slice(None), slice(None))),
            filename=self.filename,
            name=self.name,
            axisunits=axisunits,
            valunit=self.valunit,
        )
        data._emout_dir = self._emout_dir
        data._emout_open_kwargs = self._emout_open_kwargs
        data._emout_inp = getattr(self, "_emout_inp", None)
        data._emout_unit = getattr(self, "_emout_unit", None)
        data._article_recorder = getattr(self, "_article_recorder", None)
        data._article_source_shape = getattr(self, "_article_source_shape", self.shape)
        from emout.article import record_data_access

        record_data_access(data, kind="materialize")
        return data

    def __create_data_with_indexes(self, indexes: List[int], tslice: slice = None) -> Data4d:
        """Create a Data4d from a range of time indices.

        Parameters
        ----------
        indexes : list
            List of time indices.
        tslice : slice, optional
            Time index range, by default ``None``.

        Returns
        -------
        Data4d
            4-D data spanning the specified timesteps.
        """
        if tslice is not None:
            start = tslice.start or 0
            stop = tslice.stop or len(self)
            step = tslice.step or 1
            tslice = slice(start, stop, step)

        array = [self[i] for i in indexes]

        axisunits = [self.tunit] + [self.axisunit] * 3

        return Data4d(
            np.array(array),
            filename=self.filename,
            name=self.name,
            tslice=tslice,
            axisunits=axisunits,
            valunit=self.valunit,
        )

    def __getitem__(self, item: Union[int, slice, List[int], Tuple[Union[int, slice, List[int]]]]):
        """Return a slice of the time-series data.

        Parameters
        ----------
        item : int or slice or list or tuple(int or slice or list)
            Time / spatial index or range (t, z, y, x).

        Returns
        -------
        object
            A ``Data*`` instance when the result is 3-D or smaller, otherwise
            a lazy :class:`GridDataSelection`.

        Raises
        ------
        TypeError
            If *item* has an unsupported type.
        """
        if isinstance(item, (tuple, int, slice, list)):
            return self.lazy[item]

        raise TypeError(f"Unsupported index type {type(item).__name__}; expected int, slice, or list")

    def chain(self, other_series: "GridDataSeries") -> "MultiGridDataSeries":
        """Chain this series with another.

        Parameters
        ----------
        other_series : GridDataSeries
            Series to append.

        Returns
        -------
        MultiGridDataSeries
            Concatenated series.
        """
        return MultiGridDataSeries(self, other_series)

    def __add__(self, other_series: "GridDataSeries") -> "MultiGridDataSeries":
        """Chain this series with another via the ``+`` operator.

        Parameters
        ----------
        other_series : GridDataSeries
            Series to append.

        Returns
        -------
        MultiGridDataSeries
            Concatenated series.
        """
        if not isinstance(other_series, GridDataSeries):
            raise TypeError(f"Cannot chain GridDataSeries with {type(other_series).__name__}")

        return self.chain(other_series)

    def __iter__(self):
        """Return an iterator over timesteps.

        Returns
        -------
        Iterator[Data3d]
            Iterator yielding `Data3d` instances in time order.
        """
        indexes = sorted(self._index2key.keys())
        for index in indexes:
            yield self[index]

    def __len__(self):
        """Return the number of timesteps.

        Returns
        -------
        int
            Number of timesteps.
        """
        return len(self._index2key)


class MultiGridDataSeries(GridDataSeries):
    """Manage multiple concatenated 3-D time-series datasets.

    Attributes
    ----------
    datafile : DataFileInfo
        Data file metadata.
    name : str
        Dataset name.
    tunit : UnitTranslator
        Time unit translator.
    axisunit : UnitTranslator
        Spatial axis unit translator.
    valunit : UnitTranslator
        Value unit translator.
    """

    def __init__(self, *series):
        """Initialize from one or more GridDataSeries.

        Parameters
        ----------
        *series : GridDataSeries or MultiGridDataSeries
            Series to concatenate.
        """
        self.series = []
        for data in series:
            self.series += self.__expand(data)

        self.datafile = self.series[0].datafile
        self.tunit = self.series[0].tunit
        self.axisunit = self.series[0].axisunit
        self.valunit = self.series[0].valunit
        self.name = self.series[0].name
        self._grid_shape = self.series[0].grid_shape
        self._emout_dir = getattr(self.series[0], "_emout_dir", None)
        self._emout_open_kwargs = getattr(self.series[0], "_emout_open_kwargs", None)
        self._emout_inp = getattr(self.series[0], "_emout_inp", None)
        self._emout_unit = getattr(self.series[0], "_emout_unit", None)
        self._local_data_policy = getattr(self.series[0], "_local_data_policy", None)
        self._article_recorder = getattr(self.series[0], "_article_recorder", None)
        self._article_source_shape = self.shape

    def __repr__(self) -> str:
        return f"<MultiGridDataSeries: name={self.name!r}, timesteps={len(self)}, segments={len(self.series)}>"

    def __expand(self, data_series: Union["GridDataSeries", "MultiGridDataSeries"]) -> List[GridDataSeries]:
        """Flatten a (Multi)GridDataSeries into a list of GridDataSeries.

        Parameters
        ----------
        data_series : GridDataSeries or MultiGridDataSeries
            Object to expand.

        Returns
        -------
        list of GridDataSeries
            Flattened list.

        Raises
        ------
        TypeError
            If the object is not a GridDataSeries.
        """
        if not isinstance(data_series, GridDataSeries):
            raise TypeError(f"Expected GridDataSeries, got {type(data_series).__name__}")
        if not isinstance(data_series, MultiGridDataSeries):
            return [data_series]

        expanded = []
        for data in data_series.series:
            expanded += self.__expand(data)

        return expanded

    def close(self) -> None:
        """Close the HDF5 file."""
        for data in self.series:
            data.h5.close()

    def time_series(self, x: Union[int, slice], y: Union[int, slice], z: Union[int, slice]):
        """Return the time series for the specified spatial range.

        Parameters
        ----------
        x : int or slice
            X coordinate or range.
        y : int or slice
            Y coordinate or range.
        z : int or slice
            Z coordinate or range.

        Returns
        -------
        numpy.ndarray
            Time-series data for the specified range.
        """
        self._require_local_data_access("read time series", f"{self.name}.time_series")
        series = np.concatenate([data.time_series(x, y, z) for data in self.series])
        return series

    @property
    def filename(self) -> Path:
        """Return the file name of the first series.

        Returns
        -------
        Path
            File name.
        """
        return self.series[0].datafile.filename

    @property
    def filenames(self) -> List[Path]:
        """Return the list of file names.

        Returns
        -------
        list of Path
            File names.
        """
        return [data.filename for data in self.series]

    @property
    def directory(self) -> Path:
        """Return the directory of the first series.

        Returns
        -------
        Path
            Directory path.
        """
        return self.series[0].datafile.directory

    @property
    def directories(self) -> List[Path]:
        """Return the list of directories.

        Returns
        -------
        list of Path
            Directory paths.
        """
        return [data.directory for data in self.series]

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Return the spatial grid shape as ``(z, y, x)``."""
        return self._grid_shape

    def _create_data_with_index(self, index: int) -> Data3d:
        """Create a Data3d snapshot for the given time index.

        Parameters
        ----------
        index : int
            Time index.

        Returns
        -------
        Data3d
            3-D data at the specified timestep.

        Raises
        ------
        IndexError
            If the specified time index does not exist.
        """
        if index < 0:
            index += len(self)
        if index < len(self.series[0]):
            return self.series[0][index]

        length = len(self.series[0])
        for series in self.series[1:]:
            series_len = len(series) - 1  # first entry overlaps

            if index < series_len + length:
                local_index = index - length + 1
                return series[local_index]

            length += series_len

        raise IndexError(f"Time index {index} is out of range for MultiGridDataSeries with {len(self)} timesteps")

    def _read_selection(self, index: int, spatial_item) -> np.ndarray:
        """Read only the requested spatial subset for one timestep."""
        self._require_local_data_access("read field data", f"{self.name}[{index}]")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Time index {index} is out of range for MultiGridDataSeries with {len(self)} timesteps")

        if index < len(self.series[0]):
            return self.series[0]._read_selection(index, spatial_item)

        length = len(self.series[0])
        for series in self.series[1:]:
            series_len = len(series) - 1
            if index < series_len + length:
                local_index = index - length + 1
                return series._read_selection(local_index, spatial_item)
            length += series_len

        raise IndexError(f"Time index {index} is out of range for MultiGridDataSeries with {len(self)} timesteps")

    def __iter__(self):
        """Return an iterator over all timesteps.

        Returns
        -------
        Iterator
            Chained iterator across all sub-series.
        """
        iters = [iter(self.series[0])]
        for data in self.series[1:]:
            it = iter(data)
            next(it)  # skip the first entry (overlaps with previous series)
            iters.append(it)
        return chain.from_iterable(iters)

    def __len__(self) -> int:
        """Return the total number of timesteps.

        Returns
        -------
        int
            Total number of timesteps.
        """
        return int(np.sum([len(data) for data in self.series]) - (len(self.series) - 1))


class GridDataSelection(NDArrayOperatorsMixin):
    """Lazy staged selector for a :class:`GridDataSeries`.

    Instances keep the selected ``(t, z, y, x)`` ranges as metadata and only
    materialise arrays once the result has at most three dimensions or the
    caller asks for functionality that requires a real :class:`Data4d`.
    """

    __array_priority__ = 1000

    def __init__(self, series: GridDataSeries, selectors=None):
        self.series = series
        self.datafile = series.datafile
        self.name = series.name
        self.axisunits = [series.tunit] + [series.axisunit] * 3
        self.valunit = series.valunit
        self._emout_dir = getattr(series, "_emout_dir", None)
        self._emout_open_kwargs = getattr(series, "_emout_open_kwargs", None)
        self._emout_inp = getattr(series, "_emout_inp", None)
        self._emout_unit = getattr(series, "_emout_unit", None)
        self._local_data_policy = getattr(series, "_local_data_policy", None)
        self._article_recorder = getattr(series, "_article_recorder", None)
        self._article_source_shape = getattr(series, "_article_source_shape", None)
        self._axis_lengths = (len(series), *series.grid_shape)
        self._materialized = _MATERIALIZED_UNSET

        if selectors is None:
            selectors = (slice(None),) * 4
        if len(selectors) != 4:
            raise ValueError("GridDataSelection expects 4 selectors in (t, z, y, x) order")

        self._selectors = tuple(
            _normalize_selector(selector, size) for selector, size in zip(selectors, self._axis_lengths)
        )

    def __repr__(self) -> str:
        axes = "".join(self.use_axes)
        return f"<GridDataSelection: name={self.name!r}, shape={self.shape}, axes={axes}>"

    @property
    def filename(self) -> Path:
        return self.series.filename

    @property
    def directory(self) -> Path:
        return self.series.directory

    @property
    def inp(self):
        inp = getattr(self, "_emout_inp", None)
        if inp is None:
            raise AttributeError("This selection is not associated with an Emout input")
        return inp

    @property
    def unit(self):
        unit = getattr(self, "_emout_unit", None)
        if unit is None:
            raise AttributeError("This selection is not associated with Emout units")
        return unit

    @property
    def boundaries(self):
        from emout.core.boundaries import BoundaryCollection

        return BoundaryCollection(
            getattr(self, "_emout_inp", None),
            getattr(self, "_emout_unit", None),
            remote_open_kwargs=getattr(self, "_emout_open_kwargs", None),
        )

    @property
    def boundary(self):
        return self.boundaries

    def _axis_values(self, axis: int) -> np.ndarray:
        selector = self._selectors[axis]
        size = self._axis_lengths[axis]
        if isinstance(selector, int):
            return np.array([selector])
        return np.array(tuple(_selector_positions(selector, size)))

    @property
    def slices(self):
        return [
            _selector_to_metadata_slice(selector, size) for selector, size in zip(self._selectors, self._axis_lengths)
        ]

    @property
    def slice_axes(self):
        return [axis for axis, selector in enumerate(self._selectors) if not isinstance(selector, int)]

    @property
    def use_axes(self):
        to_axis = {3: "x", 2: "y", 1: "z", 0: "t"}
        return [to_axis[axis] for axis in self.slice_axes]

    @property
    def shape(self):
        return tuple(
            _selector_length(selector, size)
            for selector, size in zip(self._selectors, self._axis_lengths)
            if not isinstance(selector, int)
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def x(self) -> np.ndarray:
        return self._axis_values(3)

    @property
    def y(self) -> np.ndarray:
        return self._axis_values(2)

    @property
    def z(self) -> np.ndarray:
        return self._axis_values(1)

    @property
    def t(self) -> np.ndarray:
        return self._axis_values(0)

    @property
    def x_si(self) -> np.ndarray:
        return self.axisunits[3].reverse(self.x)

    @property
    def y_si(self) -> np.ndarray:
        return self.axisunits[2].reverse(self.y)

    @property
    def z_si(self) -> np.ndarray:
        return self.axisunits[1].reverse(self.z)

    @property
    def t_si(self) -> np.ndarray:
        return self.axisunits[0].reverse(self.t)

    def axis(self, ax: int) -> np.ndarray:
        index = self.slice_axes[ax]
        return self._axis_values(index)

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of unsized scalar selection")
        return self.shape[0]

    def __iter__(self):
        self._require_local_data_access("iterate materialized field data", self._target_name())
        return iter(self.materialize())

    def __getattr__(self, name):
        if name in {"__array_interface__", "__array_struct__"}:
            raise AttributeError(name)
        self._require_local_data_access("access materialized field attribute", f"{self._target_name()}.{name}")
        return getattr(self.materialize(), name)

    def __array__(self, dtype=None):
        self._require_local_data_access("convert field data to a NumPy array", self._target_name())
        array = self.materialize()
        if hasattr(array, "to_numpy"):
            array = array.to_numpy()
        return np.asarray(array, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self._require_local_data_access("apply a NumPy ufunc to field data", self._target_name())
        resolved_inputs = tuple(_materialize_nested(value) for value in inputs)
        resolved_kwargs = {key: _materialize_nested(value) for key, value in kwargs.items()}
        return getattr(ufunc, method)(*resolved_inputs, **resolved_kwargs)

    def __array_function__(self, func, types, args, kwargs):
        self._require_local_data_access("apply a NumPy function to field data", self._target_name())
        resolved_args = tuple(_materialize_nested(value) for value in args)
        resolved_kwargs = {key: _materialize_nested(value) for key, value in kwargs.items()}
        return func(*resolved_args, **resolved_kwargs)

    def __reduce__(self):
        self._require_local_data_access("pickle materialized field data", self._target_name())
        return self.materialize().__reduce__()

    def __reduce_ex__(self, protocol):
        self._require_local_data_access("pickle materialized field data", self._target_name())
        return self.materialize().__reduce_ex__(protocol)

    def _expand_item(self, item):
        if not isinstance(item, tuple):
            item = (item,)

        if item.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        if Ellipsis in item:
            idx = item.index(Ellipsis)
            fill = (slice(None),) * (self.ndim - (len(item) - 1))
            item = item[:idx] + fill + item[idx + 1 :]

        if len(item) > self.ndim:
            raise IndexError(f"too many indices for {self.ndim}-dimensional selection")
        return item

    def __getitem__(self, item):
        if self._materialized is not _MATERIALIZED_UNSET:
            self._require_local_data_access("slice materialized field data", self._target_name())
            return self.materialize()[item]

        item = self._expand_item(item)
        selectors = list(self._selectors)
        axes = self.slice_axes

        for dim, sub_selector in enumerate(item):
            if sub_selector is None:
                raise IndexError("newaxis is not supported on GridDataSelection")
            axis = axes[dim]
            selectors[axis] = _compose_selector(selectors[axis], sub_selector, self._axis_lengths[axis])

        result = type(self)(self.series, tuple(selectors))
        if result.ndim <= 3 and not result._local_data_access_disabled():
            if result.ndim == 3 and result._article_recorder is not None and is_spatial_3d_selection(result._selectors):
                return result
            return result.materialize()
        return result

    def select_space(self, *item):
        """Select only spatial axes while preserving the current time range.

        Examples
        --------
        ``data.phisp.lazy[:].select_space(1, 1, 1)`` is equivalent to
        ``data.phisp.lazy[:][:, 1, 1, 1]``.
        """
        if len(item) == 1 and isinstance(item[0], tuple):
            item = item[0]
        if 0 not in self.slice_axes:
            return self[item]
        return self[(slice(None), *item)]

    def to_numpy(self) -> np.ndarray:
        """Materialise the current selection as a plain NumPy array."""
        self._require_local_data_access("convert field data to a NumPy array", self._target_name())
        array = self.materialize()
        if hasattr(array, "to_numpy"):
            return array.to_numpy()
        from emout.article import record_data_access

        record_data_access(array, kind="to_numpy")
        return np.asarray(array)

    def materialize(self):
        """Materialise the current selection as a Data object or scalar."""
        self._require_local_data_access("materialize field data locally", self._target_name())
        if self._materialized is not _MATERIALIZED_UNSET:
            return self._materialized

        tsel, zsel, ysel, xsel = self._selectors
        spatial_item = (zsel, ysel, xsel)

        if isinstance(tsel, int):
            array = self.series._read_selection(tsel, spatial_item)
        else:
            t_indices = tuple(_selector_positions(tsel, len(self.series)))
            array = np.array([self.series._read_selection(index, spatial_item) for index in t_indices])

        if np.isscalar(array) or np.ndim(array) == 0:
            return np.asarray(array).item()

        data = data_from_array(
            array,
            filename=self.filename,
            name=self.name,
            tslice=self.slices[0],
            zslice=self.slices[1],
            yslice=self.slices[2],
            xslice=self.slices[3],
            slice_axes=self.slice_axes,
            axisunits=self.axisunits,
            valunit=self.valunit,
            local_data_policy=self._local_data_policy,
            emout_dir=self._emout_dir,
            emout_open_kwargs=self._emout_open_kwargs,
            emout_inp=self._emout_inp,
            emout_unit=self._emout_unit,
            article_recorder=self._article_recorder,
            article_source_shape=self._article_source_shape or self._axis_lengths,
        )
        if not hasattr(data, "slices"):
            self._materialized = data
            return data
        from emout.article import record_data_access

        record_data_access(data, kind="materialize")
        self._materialized = data
        return data

    def min(self):
        """Return the minimum value without materialising the full 4-D array."""
        self._require_local_data_access("compute a local field minimum", self._target_name())
        if self._materialized is not _MATERIALIZED_UNSET:
            return self.materialize().min()

        tsel, zsel, ysel, xsel = self._selectors
        spatial_item = (zsel, ysel, xsel)

        if isinstance(tsel, int):
            return np.min(self.series._read_selection(tsel, spatial_item))

        minimum = None
        for index in _selector_positions(tsel, len(self.series)):
            value = np.min(self.series._read_selection(index, spatial_item))
            minimum = value if minimum is None else min(minimum, value)
        return minimum

    def max(self):
        """Return the maximum value without materialising the full 4-D array."""
        self._require_local_data_access("compute a local field maximum", self._target_name())
        if self._materialized is not _MATERIALIZED_UNSET:
            return self.materialize().max()

        tsel, zsel, ysel, xsel = self._selectors
        spatial_item = (zsel, ysel, xsel)

        if isinstance(tsel, int):
            return np.max(self.series._read_selection(tsel, spatial_item))

        maximum = None
        for index in _selector_positions(tsel, len(self.series)):
            value = np.max(self.series._read_selection(index, spatial_item))
            maximum = value if maximum is None else max(maximum, value)
        return maximum

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        """Return a lazy time average when reducing the time axis."""
        if out is not None:
            raise NotImplementedError("GridDataSelection.mean does not support out")
        if keepdims:
            raise NotImplementedError("GridDataSelection.mean does not support keepdims=True")

        reduction_axes = self._normalize_reduction_axes(axis)
        if reduction_axes == (0,) and 0 in self.slice_axes:
            return GridDataMean(self, dtype=dtype)

        self._require_local_data_access("compute a local field mean", self._target_name())
        materialized_axis = self._reduction_axes_to_materialized_axis(reduction_axes)
        return self.materialize().mean(axis=materialized_axis, dtype=dtype, out=out, keepdims=keepdims)

    def _normalize_reduction_axes(self, axis) -> Tuple[int, ...]:
        if axis is None:
            if 0 in self.slice_axes:
                return (0,)
            return tuple(self.slice_axes)

        if isinstance(axis, str):
            axis = (axis,)
        elif isinstance(axis, tuple):
            axis = axis
        else:
            axis = (axis,)

        axes = []
        name_to_axis = {"t": 0, "z": 1, "y": 2, "x": 3}
        for item in axis:
            if isinstance(item, str):
                if item not in name_to_axis:
                    raise ValueError(f"Unknown axis name: {item!r}")
                source_axis = name_to_axis[item]
            else:
                dim = int(item)
                if dim < 0:
                    dim += self.ndim
                if not 0 <= dim < self.ndim:
                    raise ValueError(f"axis {item!r} is out of bounds for mean field with ndim {self.ndim}")
                source_axis = self.slice_axes[dim]
            if source_axis not in axes:
                axes.append(source_axis)
        return tuple(sorted(axes))

    def _reduction_axes_to_materialized_axis(self, reduction_axes: Tuple[int, ...]):
        dims = tuple(self.slice_axes.index(axis) for axis in reduction_axes)
        if len(dims) == 1:
            return dims[0]
        return dims

    def build_frame_updater(
        self,
        axis: int = 0,
        title: Union[str, None] = None,
        notitle: bool = False,
        offsets: Union[Tuple[Union[float, str], Union[float, str], Union[float, str]], None] = None,
        use_si: bool = True,
        vmin: float = None,
        vmax: float = None,
        **kwargs,
    ) -> FrameUpdater:
        """Build a frame updater for lazy 4-D selections."""
        self._require_local_data_access("build a local animation frame updater", self._target_name())
        if use_si and self.valunit is not None:
            vmin = vmin if vmin is not None else self.valunit.reverse(self.min())
            vmax = vmax if vmax is not None else self.valunit.reverse(self.max())
        else:
            vmin = vmin if vmin is not None else self.min()
            vmax = vmax if vmax is not None else self.max()

        return FrameUpdater(self, axis, title, notitle, offsets, use_si, vmin=vmin, vmax=vmax, **kwargs)

    def gifplot(
        self,
        fig: Union[plt.Figure, None] = None,
        axis: int = 0,
        mode: str = None,
        action: ANIMATER_PLOT_MODE = "to_html",
        filename: PathLike = None,
        show: bool = False,
        savefilename: PathLike = None,
        interval: int = 200,
        repeat: bool = True,
        title: Union[str, None] = None,
        notitle: bool = False,
        offsets: Union[Tuple[Union[float, str], Union[float, str], Union[float, str]], None] = None,
        use_si: bool = True,
        vmin: float = None,
        vmax: float = None,
        to_html: bool = False,
        return_updater: bool = False,
        **kwargs,
    ):
        """Create an animation without materialising the full 4-D array."""
        if to_html:
            warnings.warn(
                "The 'to_html' flag is deprecated. Please use gifplot(action='to_html') instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if return_updater:
            warnings.warn(
                "The 'return_updater' flag is deprecated. Please use gifplot(action='frames') instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            action = "frames"

        if mode is None:
            updater = self.build_frame_updater(axis, title, notitle, offsets, use_si, vmin, vmax, **kwargs)
        else:
            updater = self.build_frame_updater(
                axis,
                title,
                notitle,
                offsets,
                use_si,
                vmin,
                vmax,
                mode=mode,
                **kwargs,
            )

        if action == "frames":
            return updater

        animator = updater.to_animator()
        return animator.plot(
            fig=fig,
            action=action,
            filename=filename,
            show=show,
            savefilename=savefilename,
            interval=interval,
            repeat=repeat,
            to_html=to_html,
        )

    def plot(self, **kwargs):
        """Plot the selection, using remote rendering when local reads are disabled."""
        remote = self._try_remote_plot(**kwargs)
        if remote is _REMOTE_PLOT_HANDLED:
            return None
        return self.materialize().plot(**kwargs)

    def plot_surfaces(
        self,
        surfaces,
        *,
        ax=None,
        use_si: bool = True,
        vmin=None,
        vmax=None,
        **kwargs,
    ):
        """Plot a bounded 3-D surface field without recording a full volume."""
        if self.ndim != 3 or not is_spatial_3d_selection(self._selectors):
            raise ValueError("plot_surfaces requires one time index and all three spatial axes")

        bounds = kwargs.get("bounds")
        roi_selectors = plot_surfaces_roi_selectors(
            self._selectors,
            self._axis_lengths,
            self.axisunits,
            self.valunit,
            use_si,
            bounds,
        )
        data = type(self)(self.series, roi_selectors).materialize()
        return data._plot_surfaces_local(
            surfaces,
            ax=ax,
            use_si=use_si,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

    def _local_data_access_disabled(self) -> bool:
        return is_local_data_access_disabled(self._local_data_policy)

    def _require_local_data_access(self, operation: str, target: Union[str, None] = None) -> None:
        require_local_data_access(self._local_data_policy, operation, target)

    def _target_name(self) -> str:
        return f"{self.name}{self._to_recipe_index()}"

    def _to_recipe_index(self):
        """Reconstruct a GridDataSeries[index]-style tuple from selectors."""
        result = []
        for selector in self._selectors:
            result.append(selector)
        return tuple(result)

    def _get_remote_open_kwargs(self):
        if self._emout_open_kwargs is not None:
            return dict(self._emout_open_kwargs)
        if self._emout_dir is None:
            return None
        return {"directory": str(self._emout_dir)}

    def _try_remote_plot(self, **plot_kwargs):
        """Render this selection remotely when local field reads are disabled."""
        remote_kwargs = self._get_remote_open_kwargs()

        from emout.distributed.remote_figure import (
            is_recording,
            record_field_plot,
            request_session,
        )

        if is_recording():
            session = request_session(remote_kwargs)
            if session is None and self._local_data_access_disabled():
                self._require_local_data_access(
                    "record a remote field plot without a remote session",
                    self._target_name(),
                )
            record_field_plot(self.name, self._to_recipe_index(), plot_kwargs, emout_kwargs=remote_kwargs)
            return _REMOTE_PLOT_HANDLED

        if not self._local_data_access_disabled():
            if remote_kwargs is None:
                return None
            from emout.distributed.remote_render import _await_remote, get_or_create_session

            session = get_or_create_session(emout_kwargs=remote_kwargs)
            if session is None:
                return None
            payload = _await_remote(session.fetch_field(self.name, self._to_recipe_index(), emout_kwargs=remote_kwargs))
            local_data = data_from_payload(payload)
            return local_data.plot(**plot_kwargs)

        if remote_kwargs is None:
            self._require_local_data_access("plot field data locally", self._target_name())

        from emout.distributed.remote_render import _await_remote, display_image, get_or_create_session

        session = get_or_create_session(emout_kwargs=remote_kwargs)
        if session is None:
            self._require_local_data_access("plot field data without a remote session", self._target_name())

        ax = plot_kwargs.pop("ax", None)
        img = _await_remote(
            session.render_field(
                self.name,
                self._to_recipe_index(),
                emout_kwargs=remote_kwargs,
                **plot_kwargs,
            )
        )
        display_image(img, ax=ax)
        return _REMOTE_PLOT_HANDLED


class GridDataMean(NDArrayOperatorsMixin):
    """Lazy time-mean reduction for a :class:`GridDataSelection`."""

    __array_priority__ = 1000

    def __init__(self, selection: GridDataSelection, *, selectors=None, dtype=None):
        self.selection = selection
        self.series = selection.series
        self.datafile = selection.datafile
        self.name = selection.name
        self.axisunits = selection.axisunits
        self.valunit = selection.valunit
        self._emout_dir = selection._emout_dir
        self._emout_open_kwargs = selection._emout_open_kwargs
        self._emout_inp = getattr(selection, "_emout_inp", None)
        self._emout_unit = getattr(selection, "_emout_unit", None)
        self._local_data_policy = selection._local_data_policy
        self._article_recorder = selection._article_recorder
        self._article_source_shape = selection._article_source_shape
        self._axis_lengths = selection._axis_lengths
        self._selectors = tuple(selection._selectors if selectors is None else selectors)
        self._dtype = dtype
        self._materialized = _MATERIALIZED_UNSET

    def __repr__(self) -> str:
        axes = "".join(self.use_axes)
        return f"<GridDataMean: name={self.name!r}, shape={self.shape}, axes={axes}>"

    @property
    def filename(self) -> Path:
        return self.selection.filename

    @property
    def directory(self) -> Path:
        return self.selection.directory

    @property
    def inp(self):
        inp = getattr(self, "_emout_inp", None)
        if inp is None:
            raise AttributeError("This mean field is not associated with an Emout input")
        return inp

    @property
    def unit(self):
        unit = getattr(self, "_emout_unit", None)
        if unit is None:
            raise AttributeError("This mean field is not associated with Emout units")
        return unit

    @property
    def boundaries(self):
        from emout.core.boundaries import BoundaryCollection

        return BoundaryCollection(
            getattr(self, "_emout_inp", None),
            getattr(self, "_emout_unit", None),
            remote_open_kwargs=getattr(self, "_emout_open_kwargs", None),
        )

    @property
    def boundary(self):
        return self.boundaries

    @property
    def slices(self):
        return [
            _selector_to_metadata_slice(selector, size) for selector, size in zip(self._selectors, self._axis_lengths)
        ]

    @property
    def slice_axes(self):
        return [axis for axis, selector in enumerate(self._selectors) if axis != 0 and not isinstance(selector, int)]

    @property
    def use_axes(self):
        to_axis = {3: "x", 2: "y", 1: "z", 0: "t"}
        return [to_axis[axis] for axis in self.slice_axes]

    @property
    def shape(self):
        return tuple(
            _selector_length(selector, size)
            for axis, (selector, size) in enumerate(zip(self._selectors, self._axis_lengths))
            if axis != 0 and not isinstance(selector, int)
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __array__(self, dtype=None):
        array = self.to_numpy()
        return np.asarray(array, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        resolved_inputs = tuple(_materialize_nested(value) for value in inputs)
        resolved_kwargs = {key: _materialize_nested(value) for key, value in kwargs.items()}
        return getattr(ufunc, method)(*resolved_inputs, **resolved_kwargs)

    def __array_function__(self, func, types, args, kwargs):
        resolved_args = tuple(_materialize_nested(value) for value in args)
        resolved_kwargs = {key: _materialize_nested(value) for key, value in kwargs.items()}
        return func(*resolved_args, **resolved_kwargs)

    def _expand_item(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        if item.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if Ellipsis in item:
            idx = item.index(Ellipsis)
            fill = (slice(None),) * (self.ndim - (len(item) - 1))
            item = item[:idx] + fill + item[idx + 1 :]
        if len(item) > self.ndim:
            raise IndexError(f"too many indices for {self.ndim}-dimensional mean field")
        return item

    def __getitem__(self, item):
        item = self._expand_item(item)
        selectors = list(self._selectors)
        for dim, sub_selector in enumerate(item):
            if sub_selector is None:
                raise IndexError("newaxis is not supported on GridDataMean")
            axis = self.slice_axes[dim]
            selectors[axis] = _compose_selector(selectors[axis], sub_selector, self._axis_lengths[axis])
        return type(self)(self.selection, selectors=tuple(selectors), dtype=self._dtype)

    def materialize(self, selectors=None):
        """Materialise the time mean as a Data object or scalar."""
        self.selection._require_local_data_access("compute a local time mean", self._target_name())
        if selectors is None and self._materialized is not _MATERIALIZED_UNSET:
            return self._materialized

        selectors = self._selectors if selectors is None else tuple(selectors)
        tsel, zsel, ysel, xsel = selectors
        t_indices = _selector_positions(tsel, self._axis_lengths[0])
        if not t_indices:
            raise ValueError("Cannot compute mean over an empty time selection")

        spatial_item = (zsel, ysel, xsel)
        dtype = self._dtype or np.float64
        total = None
        for index in t_indices:
            array = np.asarray(self.series._read_selection(index, spatial_item), dtype=dtype)
            total = array if total is None else total + array
        mean = total / len(t_indices)

        if np.isscalar(mean) or np.ndim(mean) == 0:
            return np.asarray(mean).item()

        data = data_from_array(
            mean,
            filename=self.filename,
            name=self.name,
            tslice=_selector_to_metadata_slice(tsel, self._axis_lengths[0]),
            zslice=_selector_to_metadata_slice(zsel, self._axis_lengths[1]),
            yslice=_selector_to_metadata_slice(ysel, self._axis_lengths[2]),
            xslice=_selector_to_metadata_slice(xsel, self._axis_lengths[3]),
            slice_axes=[axis for axis, selector in enumerate(selectors) if axis != 0 and not isinstance(selector, int)],
            axisunits=self.axisunits,
            valunit=self.valunit,
            local_data_policy=self._local_data_policy,
            emout_dir=self._emout_dir,
            emout_open_kwargs=self._emout_open_kwargs,
            emout_inp=self._emout_inp,
            emout_unit=self._emout_unit,
            article_recorder=self._article_recorder,
            article_source_shape=self._article_source_shape or self._axis_lengths,
        )
        from emout.article import record_data_access

        record_data_access(
            data,
            kind="mean",
            kwargs={
                "operation": "mean",
                "reduction_axes": ["t"],
                "source_selector": self._to_recipe_index(selectors),
            },
        )
        if selectors == self._selectors:
            self._materialized = data
        return data

    def to_numpy(self) -> np.ndarray:
        data = self.materialize()
        return np.asarray(data)

    def plot(self, **kwargs):
        data = self.materialize()
        recorder = getattr(data, "_article_recorder", None)
        data._article_recorder = None
        try:
            return data.plot(**kwargs)
        finally:
            data._article_recorder = recorder

    def plot_surfaces(
        self,
        surfaces,
        *,
        ax=None,
        use_si: bool = True,
        vmin=None,
        vmax=None,
        **kwargs,
    ):
        """Plot a bounded time-mean 3-D surface field."""
        if self.ndim != 3 or self.slice_axes != [1, 2, 3]:
            raise ValueError("plot_surfaces requires a time mean over all three spatial axes")

        roi_selectors = surface_roi_selectors(
            self._selectors,
            self._axis_lengths,
            self.axisunits,
            self.valunit,
            use_si,
            kwargs.get("bounds"),
        )
        data = self.materialize(roi_selectors)
        return data._plot_surfaces_local(
            surfaces,
            ax=ax,
            use_si=use_si,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

    def _target_name(self) -> str:
        return f"{self.name}.mean({self._to_recipe_index()})"

    def _to_recipe_index(self, selectors=None):
        return tuple(self._selectors if selectors is None else selectors)
