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
    """3次元時系列データを管理する.

    Attributes
    ----------
    datafile : DataFileInfo
        データファイル情報
    h5 : h5py.File
        hdf5ファイルオブジェクト
    group : h5py.Datasets
        データセット
    name : str
        データセット名
    """

    def __init__(
        self,
        filename: PathLike,
        name: str,
        tunit: UnitTranslator = None,
        axisunit: UnitTranslator = None,
        valunit: UnitTranslator = None,
    ):
        """3次元時系列データを生成する.

        Parameters
        ----------
        filename : str or Path
            ファイル名
        name : str
            データの名前
        """
        self.datafile = DataFileInfo(filename)
        self.h5 = h5py.File(str(filename), "r")
        self.group = self.h5[list(self.h5.keys())[0]]
        self._index2key = {int(key): key for key in self.group.keys()}
        self.tunit = tunit
        self.axisunit = axisunit
        self.valunit = valunit

        self.name = name

    def close(self) -> None:
        """hdf5ファイルを閉じる."""
        self.h5.close()

    def time_series(self, x, y, z) -> np.ndarray:
        """指定した範囲の時系列データを取得する.

        Parameters
        ----------
        x : int or slice
            x座標
        y : int or slice
            y座標
        z : int or slice
            z座標

        Returns
        -------
        numpy.ndarray
            指定した範囲の時系列データ
        """
        series = []
        indexes = sorted(self._index2key.keys())
        for index in indexes:
            key = self._index2key[index]
            series.append(self.group[key][z, y, x])
        return np.array(series)

    @property
    def filename(self) -> Path:
        """ファイル名を返す.

        Returns
        -------
        Path
            ファイル名
        """
        return self.datafile.filename

    @property
    def directory(self) -> Path:
        """ディレクトリ名を返す.

        Returns
        -------
        Path
            ディレクトリ名
        """
        return self.datafile.directory

    def _create_data_with_index(self, index: int) -> Data3d:
        """時間が指定された場合に、その時間におけるData3dを生成する.

        Parameters
        ----------
        index : int
            時間インデックス

        Returns
        -------
        Data3d
            生成したData3d

        Raises
        ------
        IndexError
            指定した時間が存在しない場合の例外
        """
        if index not in self._index2key:
            raise IndexError()

        key = self._index2key[index]

        axisunits = [self.tunit] + [self.axisunit] * 3

        return Data3d(
            np.array(self.group[key]),
            filename=self.filename,
            name=self.name,
            axisunits=axisunits,
            valunit=self.valunit,
        )

    def __create_data_with_indexes(
        self, indexes: List[int], tslice: slice = None
    ) -> Data4d:
        """時間が範囲で指定された場合に、Data4dを生成する.

        Parameters
        ----------
        indexes : list
            時間インデックスのリスト
        tslice : slice, optional
            時間インデックスの範囲, by default None

        Returns
        -------
        Data4d
            生成したData4d
        """
        if tslice is not None:
            start = tslice.start or 0
            stop = tslice.stop or len(self)
            step = tslice.step or 1
            tslice = slice(start, stop, step)

        array = []
        for i in indexes:
            array.append(self[i])

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
        """時系列データをスライスしたものを返す.

        Parameters
        ----------
        item : int or slice or list or tuple(int or slice or list)
            tzxyインデックスの範囲

        Returns
        -------
        Data3d or Data4d
            スライスされたデータ

        Raises
        ------
        TypeError
            itemのタイプが正しくない場合の例外
        """
        # xyzの範囲も指定された場合
        if isinstance(item, tuple):
            if isinstance(item[0], int):
                return self[item[0]][item[1:]]
            else:
                slices = (slice(None), *item[1:])
                return self[item[0]][slices]

        # 以下、tの範囲のみ指定された場合
        if isinstance(item, int):  # tが一つだけ指定された場合
            index = item
            if index < 0:
                index = len(self) + index
            return self._create_data_with_index(index)

        elif isinstance(item, slice):  # tがスライスで指定された場合
            indexes = list(utils.range_with_slice(item, maxlen=len(self)))
            return self.__create_data_with_indexes(indexes, tslice=item)

        elif isinstance(item, list):  # tがリストで指定された場合
            return self.__create_data_with_indexes(item)

        else:
            raise TypeError()

    def chain(self, other_series: "GridDataSeries") -> "MultiGridDataSeries":
        """GridDataSeriesを結合する.

        Parameters
        ----------
        other_series : GridDataSeries
            結合するGridDataSeries

        Returns
        -------
        MultiGridDataSeries
            結合したGridDataSeries
        """
        return MultiGridDataSeries(self, other_series)

    def __add__(self, other_series: "GridDataSeries") -> "MultiGridDataSeries":
        """GridDataSeriesを結合する.

        Parameters
        ----------
        other_series : GridDataSeries
            結合するGridDataSeries

        Returns
        -------
        MultiGridDataSeries
            結合したGridDataSeries
        """
        if not isinstance(other_series, GridDataSeries):
            raise TypeError()

        return self.chain(other_series)

    def __iter__(self):
        indexes = sorted(self._index2key.keys())
        for index in indexes:
            yield self[index]

    def __len__(self):
        return len(self._index2key)


class MultiGridDataSeries(GridDataSeries):
    """連続する複数の3次元時系列データを管理する.

    Attributes
    ----------
    datafile : DataFileInfo
        データファイル情報
    name : str
        データセット名
    tunit : UnitTranslator
        時間の単位変換器
    axisunit : UnitTranslator
        空間軸の単位変換器
    valunit : UnitTranslator
        値の単位変換器
    """

    def __init__(self, *series):
        self.series = []
        for data in series:
            self.series += self.__expand(data)

        self.datafile = self.series[0].datafile
        self.tunit = self.series[0].tunit
        self.axisunit = self.series[0].axisunit
        self.valunit = self.series[0].valunit

        self.name = self.series[0].name

    def __expand(
        self, data_series: Union["GridDataSeries", "MultiGridDataSeries"]
    ) -> List[GridDataSeries]:
        """与えられたオブジェクトがMultiGridDataSeriesなら展開してGridDataSeriesのリストとして返す.

        Parameters
        ----------
        data_series : GridDataSeries or MultiGridDataSeries
            オブジェクト

        Returns
        -------
        list(GridDataSeries)
            GridDataSeriesのリスト

        Raises
        ------
        TypeError
            オブジェクトがGridDataSeriesでない場合の例外
        """
        if not isinstance(data_series, GridDataSeries):
            raise TypeError()
        if not isinstance(data_series, MultiGridDataSeries):
            return [data_series]

        # data_seriesがMultiGridDataSeriesならデータを展開して結合する.
        expanded = []
        for data in data_series.series:
            expanded += self.__expand(data)

        return expanded

    def close(self) -> None:
        """hdf5ファイルを閉じる."""
        for data in self.series:
            self.series.h5.close()

    def time_series(
        self, x: Union[int, slice], y: Union[int, slice], z: Union[int, slice]
    ):
        """指定した範囲の時系列データを取得する.

        Parameters
        ----------
        x : int or slice
            x座標
        y : int or slice
            y座標
        z : int or slice
            z座標

        Returns
        -------
        numpy.ndarray
            指定した範囲の時系列データ
        """
        series = np.concatenate([data.time_series(x, y, z) for data in self.series])
        return series

    @property
    def filename(self) -> Path:
        """先頭データのファイル名を返す.

        Returns
        -------
        Path
            ファイル名
        """
        return self.series[0].datafile.filename

    @property
    def filenames(self) -> List[Path]:
        """ファイル名のリストを返す.

        Returns
        -------
        list(Path)
            ファイル名のリスト
        """
        return [data.filename for data in self.series]

    @property
    def directory(self) -> Path:
        """先頭データのディレクトリ名を返す.

        Returns
        -------
        Path
            ディレクトリ名
        """
        return self.series[0].datafile.directory

    @property
    def directories(self) -> List[Path]:
        """ディレクトリ名のリストを返す.

        Returns
        -------
        list(Path)
            ディレクトリ名のリスト
        """
        return [data.directory for data in self.series]

    def _create_data_with_index(self, index: int) -> Data3d:
        """時間が指定された場合に、その時間におけるData3dを生成する.

        Parameters
        ----------
        index : int
            時間インデックス

        Returns
        -------
        Data3d
            生成したData3d

        Raises
        ------
        IndexError
            指定した時間が存在しない場合の例外
        """
        if index < len(self.series[0]):
            return self.series[0][index]

        length = len(self.series[0])
        for series in self.series[1:]:
            # 先頭データは前のデータの最後尾と重複しているためカウントしない
            series_len = len(series) - 1

            if index < series_len + length:
                local_index = index - length + 1
                return series[local_index]

            length += series_len

        raise IndexError()

    def __iter__(self):
        iters = [iter(self.series[0])]
        for data in self.series[1:]:
            it = iter(data)
            next(it)  # 先頭データを捨てる
            iters.append(it)
        return chain(iters)

    def __len__(self) -> int:
        # 先頭データは前のデータの最後尾と重複しているためカウントしない
        return np.sum([len(data) for data in self.series]) - (len(self.series) - 1)
