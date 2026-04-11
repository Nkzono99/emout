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

from emout.plot.animation_plot import ANIMATER_PLOT_MODE, FrameUpdater
from emout.utils import DataFileInfo, UnitTranslator

from .data import Data1d, Data2d, Data3d, Data4d

_MATERIALIZED_UNSET = object()


def _normalize_index(index: int, size: int) -> int:
    index = int(index)
    if index < 0:
        index += size
    if not 0 <= index < size:
        raise IndexError(f"index {index} is out of bounds for axis with size {size}")
    return index


def _normalize_selector(selector, size: int):
    if isinstance(selector, slice):
        rng = range(*selector.indices(size))
        return slice(rng.start, rng.stop, rng.step)
    if isinstance(selector, list):
        return tuple(_normalize_index(index, size) for index in selector)
    if isinstance(selector, tuple):
        return tuple(_normalize_index(index, size) for index in selector)
    if isinstance(selector, (int, np.integer)):
        return _normalize_index(selector, size)
    raise TypeError(f"Unsupported selector type {type(selector).__name__}; expected int, slice, list, or tuple")


def _selector_positions(selector, size: int):
    if isinstance(selector, slice):
        return range(*selector.indices(size))
    if isinstance(selector, tuple):
        return selector
    if isinstance(selector, list):
        return tuple(selector)
    raise TypeError(f"Cannot enumerate positions for selector type {type(selector).__name__}")


def _selector_length(selector, size: int) -> int:
    return len(_selector_positions(selector, size))


def _selector_to_metadata_slice(selector, size: int) -> slice:
    if isinstance(selector, int):
        return slice(selector, selector + 1, 1)
    if isinstance(selector, slice):
        rng = range(*selector.indices(size))
        return slice(rng.start, rng.stop, rng.step)

    positions = tuple(_selector_positions(selector, size))
    if len(positions) == 0:
        return slice(0, 0, 1)
    if len(positions) == 1:
        return slice(positions[0], positions[0] + 1, 1)

    step = positions[1] - positions[0]
    if all((right - left) == step for left, right in zip(positions, positions[1:])):
        return slice(positions[0], positions[-1] + step, step)

    # Data metadata is slice-based; fall back to relative coordinates for
    # irregular explicit selections.
    return slice(0, len(positions), 1)


def _compose_selector(base_selector, sub_selector, size: int):
    if isinstance(base_selector, int):
        raise TypeError("Cannot sub-index an axis that has already been reduced to a scalar")

    positions = _selector_positions(base_selector, size)

    if isinstance(sub_selector, list):
        plen = len(positions)
        return tuple(positions[_normalize_index(index, plen)] for index in sub_selector)

    result = positions[sub_selector]
    if isinstance(result, range):
        return slice(result.start, result.stop, result.step)
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    if isinstance(result, np.integer):
        return int(result)
    return result


def _materialize_nested(value):
    if isinstance(value, GridDataSelection):
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

        self.name = name
        self._emout_dir = None  # set by GridLoader to enable remote rendering
        self._emout_open_kwargs = None

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
        if index not in self._index2key:
            raise IndexError(f"Time index {index} does not exist. Available indices: {sorted(self._index2key.keys())}")

        key = self._index2key[index]
        return np.array(self.group[key][spatial_item])

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
        return iter(self.materialize())

    def __getattr__(self, name):
        return getattr(self.materialize(), name)

    def __array__(self, dtype=None):
        array = self.materialize()
        if hasattr(array, "to_numpy"):
            array = array.to_numpy()
        return np.asarray(array, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        resolved_inputs = tuple(_materialize_nested(value) for value in inputs)
        resolved_kwargs = {key: _materialize_nested(value) for key, value in kwargs.items()}
        return getattr(ufunc, method)(*resolved_inputs, **resolved_kwargs)

    def __array_function__(self, func, types, args, kwargs):
        resolved_args = tuple(_materialize_nested(value) for value in args)
        resolved_kwargs = {key: _materialize_nested(value) for key, value in kwargs.items()}
        return func(*resolved_args, **resolved_kwargs)

    def __reduce__(self):
        return self.materialize().__reduce__()

    def __reduce_ex__(self, protocol):
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
        if result.ndim <= 3:
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
        array = self.materialize()
        if hasattr(array, "to_numpy"):
            return array.to_numpy()
        return np.asarray(array)

    def materialize(self):
        """Materialise the current selection as a Data object or scalar."""
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

        params = {
            "filename": self.filename,
            "name": self.name,
            "tslice": self.slices[0],
            "zslice": self.slices[1],
            "yslice": self.slices[2],
            "xslice": self.slices[3],
            "slice_axes": self.slice_axes,
            "axisunits": self.axisunits,
            "valunit": self.valunit,
        }

        if array.ndim == 1:
            data = Data1d(array, **params)
        elif array.ndim == 2:
            data = Data2d(array, **params)
        elif array.ndim == 3:
            data = Data3d(array, **params)
        elif array.ndim == 4:
            data = Data4d(array, **params)
        else:
            self._materialized = array
            return array

        data._emout_dir = self._emout_dir
        data._emout_open_kwargs = self._emout_open_kwargs
        self._materialized = data
        return data

    def min(self):
        """Return the minimum value without materialising the full 4-D array."""
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
        """Delegate to the materialised :class:`Data4d`."""
        return self.materialize().plot(**kwargs)
