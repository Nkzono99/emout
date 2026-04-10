"""Lazy time-series loader for EMSES grid HDF5 files.

:class:`GridDataSeries` memory-maps a sequence of ``{name}00_0000.h5``
files and produces :class:`~emout.core.data.data.Data` slices on demand.
"""

from itertools import chain
from os import PathLike
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np

import emout.utils as utils
from emout.utils import DataFileInfo, UnitTranslator

from .data import Data3d, Data4d


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
            key = self._index2key[index]
            series.append(self.group[key][z, y, x])
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
        if index not in self._index2key:
            raise IndexError(f"Time index {index} does not exist. Available indices: {sorted(self._index2key.keys())}")

        key = self._index2key[index]

        axisunits = [self.tunit] + [self.axisunit] * 3

        data = Data3d(
            np.array(self.group[key]),
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

    def __getitem__(
        self, item: Union[int, slice, List[int], Tuple[Union[int, slice, List[int]]]]
    ) -> Union["Data3d", "Data4d"]:
        """Return a slice of the time-series data.

        Parameters
        ----------
        item : int or slice or list or tuple(int or slice or list)
            Time / spatial index or range (t, z, y, x).

        Returns
        -------
        Data3d or Data4d
            Sliced data.

        Raises
        ------
        TypeError
            If *item* has an unsupported type.
        """
        # When xyz ranges are also specified
        if isinstance(item, tuple):
            if isinstance(item[0], int):
                return self[item[0]][item[1:]]
            else:
                slices = (slice(None), *item[1:])
                return self[item[0]][slices]

        # Below: only the t range is specified
        if isinstance(item, int):  # single t index
            index = item
            if index < 0:
                index = len(self) + index
            return self._create_data_with_index(index)

        elif isinstance(item, slice):  # t given as a slice
            indexes = list(utils.range_with_slice(item, maxlen=len(self)))
            return self.__create_data_with_indexes(indexes, tslice=item)

        elif isinstance(item, list):  # t given as a list
            return self.__create_data_with_indexes(item)

        else:
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

        # If data_series is MultiGridDataSeries, expand and merge.
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
        if index < len(self.series[0]):
            return self.series[0][index]

        length = len(self.series[0])
        for series in self.series[1:]:
            # The first entry overlaps with the last of the previous series
            series_len = len(series) - 1

            if index < series_len + length:
                local_index = index - length + 1
                return series[local_index]

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
        # First entry of each appended series overlaps with the last of the previous
        """Return the total number of timesteps.

        Returns
        -------
        int
            Total number of timesteps.
        """
        return np.sum([len(data) for data in self.series]) - (len(self.series) - 1)
