import re
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import pandas as pd

import emout.utils as utils
from emout.utils import (
    DataFileInfo,
    InpFile,
    RegexDict,
    UnitConversionKey,
    Units,
    UnitTranslator,
)

from .data import Data3d, Data4d
from .util import ndp_unit, t_unit
from .vector_data import VectorData2d


class Emout:
    """EMSES出力・inpファイルを管理する.

    Attributes
    ----------
    directory : Path
        管理するディレクトリ
    dataname : GridData
        3次元データ(datanameは"phisp"などのhdf5ファイルの先頭の名前)
    """

    name2unit = RegexDict(
        {
            r"phisp": lambda self: self.unit.phi,
            # r'nd[12]p': ndp_unit,
            r"nd1p": ndp_unit(0),
            r"nd2p": ndp_unit(1),
            r"nd3p": ndp_unit(2),
            r"nd4p": ndp_unit(3),
            r"nd5p": ndp_unit(4),
            r"nd6p": ndp_unit(5),
            r"nd7p": ndp_unit(6),
            r"nd8p": ndp_unit(7),
            r"nd9p": ndp_unit(8),
            r"rho": lambda self: self.unit.rho,
            r"rhobk": lambda self: self.unit.rho,
            r"j.*": lambda self: self.unit.J,
            r"b[xyz]": lambda self: self.unit.H,
            r"e[xyz]": lambda self: self.unit.E,
            r"t": t_unit,
            r"axis": lambda self: self.unit.length,
            r"rhobksp[1-9]": lambda self: self.unit.rho,
        }
    )

    def __init__(self, directory="./", append_directories=[], inpfilename="plasma.inp"):
        """EMSES出力・inpファイルを管理するオブジェクトを生成する.

        Parameters
        ----------
        directory : str or Path
            管理するディレクトリ, by default './'
        append_directories : list(str) or list(Path)
            管理する継続ディレクトリのリスト, by default []
        inpfilename : str, optional
            パラメータファイルの名前, by default 'plasma.inp'
        """
        if not isinstance(directory, Path):
            directory = Path(directory)
        self.directory = directory

        self.append_directories = []
        for append_directory in append_directories:
            if not isinstance(append_directory, Path):
                append_directory = Path(append_directory)
            self.append_directories.append(append_directory)

        # パラメータファイルの読み取りと単位変換器の初期化
        self._inp = None
        self._unit = None
        if inpfilename is not None and (directory / inpfilename).exists():
            self._inp = InpFile(directory / inpfilename)
            convkey = UnitConversionKey.load(directory / inpfilename)
            if convkey is not None:
                self._unit = Units(dx=convkey.dx, to_c=convkey.to_c)

    def __fetch_filepath(self, directory: Path, pattern: str) -> Path:
        filepathes = list(directory.glob(pattern))
        if len(filepathes) == 0:
            raise Exception(f"{pattern} is not found.")
        if len(filepathes) >= 2:
            raise Exception(
                f"There are multiple files that satisfy {pattern}.  Please specify so that just one is specified."
            )

        filepath = filepathes[0]

        return filepath

    def __load_griddata(self, h5file_path: Path) -> "GridDataSeries":
        if self.unit is None:
            tunit = None
            axisunit = None
        else:
            tunit = Emout.name2unit.get("t", lambda self: None)(self)
            axisunit = Emout.name2unit.get("axis", lambda self: None)(self)

        name = str(h5file_path.name).replace("00_0000.h5", "")

        if self.unit is None:
            valunit = None
        else:
            valunit = Emout.name2unit.get(name, lambda self: None)(self)

        data = GridDataSeries(
            h5file_path, name, tunit=tunit, axisunit=axisunit, valunit=valunit
        )

        return data

    def __getattr__(self, __name: str) -> "GridDataSeries":
        m = re.match("(.+)([xyz])([xyz])$", __name)
        if m:
            dname = m.group(1)
            axis1 = m.group(2)
            axis2 = m.group(3)
            vector_data = VectorData2d(
                [getattr(self, f"{dname}{axis1}"), getattr(self, f"{dname}{axis2}")],
                name=__name,
            )

            setattr(self, __name, vector_data)

            return vector_data

        filepath = self.__fetch_filepath(self.directory, f"{__name}00_0000.h5")
        griddata = self.__load_griddata(filepath)

        for append_directory in self.append_directories:
            filepath = self.__fetch_filepath(append_directory, f"{__name}00_0000.h5")
            griddata_append = self.__load_griddata(filepath)

            griddata = griddata.chain(griddata_append)

        setattr(self, __name, griddata)

        return griddata

    @property
    def inp(self) -> Union[InpFile, None]:
        """パラメータの辞書(Namelist)を返す.

        Returns
        -------
        InpFile or None
            パラメータの辞書(Namelist)
        """
        return self._inp

    @property
    def unit(self) -> Union[Units, None]:
        """単位変換オブジェクトを返す.

        Returns
        -------
        Units or None
            単位変換オブジェクト
        """
        return self._unit

    @property
    def icur(self) -> pd.DataFrame:

        names = []
        for ispec in range(self.inp.nspec):
            names.append(f"{ispec+1}_step")
            for ipc in range(self.inp.npc):
                names.append(f"{ispec+1}_body{ipc+1}")
                names.append(f"{ispec+1}_body{ipc+1}_ema")

        df = pd.read_csv(self.directory / "icur", sep="\s+", header=None, names=names)

        return df

    @property
    def pbody(self) -> pd.DataFrame:
        names = ["step"] + [f"body{i+1}" for i in range(self.inp.npc + 1)]

        df = pd.read_csv(self.directory / "pbody", sep="\s+", names=names)

        return df


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
            xslice = item[1]
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
