import re
from itertools import chain
from pathlib import Path

import emout.plot as emplt
import emout.utils as utils
import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from emout.utils import (DataFileInfo, InpFile, RegexDict, UnitConversionKey,
                         Units)

from .util import t_unit, ndp_unit, nd3p_unit


class Emout:
    """EMSES出力・inpファイルを管理する.

    Attributes
    ----------
    directory : Path
        管理するディレクトリ
    dataname : GridData
        3次元データ(datanameは"phisp"などのhdf5ファイルの先頭の名前)
    """

    name2unit = RegexDict({
        r'phisp': lambda self: self.unit.phi,
        r'nd[12]p': ndp_unit,
        r'nd3p': nd3p_unit,
        r'rho': lambda self: self.unit.rho,
        r'rhobk': lambda self: self.unit.rho,
        r'j.*': lambda self: self.unit.J,
        r'b[xyz]': lambda self: self.unit.H,
        r'e[xyz]': lambda self: self.unit.E,
        r't': t_unit,
        r'axis': lambda self: self.unit.length,
    })

    def __init__(self, directory='./', append_directories=[], inpfilename='plasma.inp'):
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

        for series in self.__fetch_series(self.directory):
            setattr(self, series.name, series)

        for append_directory in self.append_directories:
            for series in self.__fetch_series(append_directory):
                new_series = getattr(self, series.name).chain(series)
                setattr(self, series.name, new_series)

    def __fetch_series(self, directory):
        """指定したディレクトリ内のh5ファイルを探査し、GridDataSeriesのリストとして返す.

        Parameters
        ----------
        directory : Path
            ディレクトリ

        Returns
        -------
        list(GridDataSeries)
            GridDataSeriesのリスト
        """
        series = []

        if self.unit is None:
            tunit = None
            axisunit = None
        else:
            tunit = Emout.name2unit.get('t', lambda self: None)(self)
            axisunit = Emout.name2unit.get('axis', lambda self: None)(self)

        for h5file_path in directory.glob('*.h5'):
            name = str(h5file_path.name).replace('00_0000.h5', '')

            if self.unit is None:
                valunit = None
            else:
                valunit = Emout.name2unit.get(name, lambda self: None)(self)

            data = GridDataSeries(h5file_path,
                                  name,
                                  tunit=tunit,
                                  axisunit=axisunit,
                                  valunit=valunit)
            series.append(data)

        return series

    @property
    def inp(self):
        """パラメータの辞書(Namelist)を返す.

        Returns
        -------
        InpFile or None
            パラメータの辞書(Namelist)
        """
        return self._inp

    @property
    def unit(self):
        """単位変換オブジェクトを返す.

        Returns
        -------
        Units or None
            単位変換オブジェクト
        """
        return self._unit


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

    def __init__(self, filename, name, tunit=None, axisunit=None, valunit=None):
        """3次元時系列データを生成する.

        Parameters
        ----------
        filename : str or Path
            ファイル名
        name : str
            データの名前
        """
        self.datafile = DataFileInfo(filename)
        self.h5 = h5py.File(str(filename), 'r')
        self.group = self.h5[list(self.h5.keys())[0]]
        self._index2key = {int(key): key for key in self.group.keys()}
        self.tunit = tunit
        self.axisunit = axisunit
        self.valunit = valunit

        self.name = name

    def close(self):
        """hdf5ファイルを閉じる.
        """
        self.h5.close()

    def time_series(self, x, y, z):
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
    def filename(self):
        """ファイル名を返す.

        Returns
        -------
        Path
            ファイル名
        """
        return self.datafile.filename

    @property
    def directory(self):
        """ディレクトリ名を返す.

        Returns
        -------
        Path
            ディレクトリ名
        """
        return self.datafile.directory

    def _create_data_with_index(self, index):
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

        return Data3d(np.array(self.group[key]),
                      filename=self.filename,
                      name=self.name,
                      axisunits=axisunits,
                      valunit=self.valunit)

    def __create_data_with_indexes(self, indexes, tslice=None):
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

        return Data4d(np.array(array),
                      filename=self.filename,
                      name=self.name,
                      tslice=tslice,
                      axisunits=axisunits,
                      valunit=self.valunit)

    def __getitem__(self, item):
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

    def chain(self, other_series):
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

    def __add__(self, other_series):
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

    def __expand(self, data_series):
        """与えられたオブジェクトがMultiGridDataSeriesなら展開してGridDataSeriesのリストとして返す.

        Parameters
        ----------
        data_series : GridDataSeries or MultGridDataSeries
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

    def close(self):
        """hdf5ファイルを閉じる.
        """
        for data in self.series:
            series.h5.close()

    def time_series(self, x, y, z):
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
        series = np.concatenate([data.time_series(x, y, z)
                                 for data in self.series])
        return series

    @property
    def filename(self):
        """先頭データのファイル名を返す.

        Returns
        -------
        Path
            ファイル名
        """
        return self.series[0].datafile.filename

    @property
    def filenames(self):
        """ファイル名のリストを返す.

        Returns
        -------
        list(Path)
            ファイル名のリスト
        """
        return [data.filename for data in self.series]

    @property
    def directory(self):
        """先頭データのディレクトリ名を返す.

        Returns
        -------
        Path
            ディレクトリ名
        """
        return self.series[0].datafile.directory

    @property
    def directories(self):
        """ディレクトリ名のリストを返す.

        Returns
        -------
        list(Path)
            ディレクトリ名のリスト
        """
        return [data.directory for data in self.series]

    def _create_data_with_index(self, index):
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

    def __len__(self):
        # 先頭データは前のデータの最後尾と重複しているためカウントしない
        return np.sum([len(data) for data in self.series]) - (len(self.series) - 1)


class Data(np.ndarray):
    """3次元データを管理する.

    Attributes
    ----------
    datafile : DataFileInfo
        データファイル情報
    name : str
        データ名
    slices : list(slice)
        管理するデータのxyz方向それぞれの範囲
    slice_axes : list(int)
        データ軸がxyzのどの方向に対応しているか表すリスト(0: t, 1: z, 2: y, 3: x)
    axisunits : list(UnitTranslator) or None
        軸の単位変換器
    valunit : UnitTranslator or None
        値の単位変換器
    """
    def __new__(cls,
                input_array,
                filename=None,
                name=None,
                xslice=None,
                yslice=None,
                zslice=None,
                tslice=None,
                slice_axes=None,
                axisunits=None,
                valunit=None):
        obj = np.asarray(input_array).view(cls)
        obj.datafile = DataFileInfo(filename)
        obj.name = name

        obj.axisunits = axisunits
        obj.valunit = valunit

        if xslice is None:
            xslice = slice(0, obj.shape[3], 1)
        if yslice is None:
            yslice = slice(0, obj.shape[2], 1)
        if zslice is None:
            zslice = slice(0, obj.shape[1], 1)
        if tslice is None:
            tslice = slice(0, obj.shape[0], 1)
        if slice_axes is None:
            slice_axes = [0, 1, 2, 3]

        obj.slices = [tslice, zslice, yslice, xslice]
        obj.slice_axes = slice_axes

        return obj

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item, )

        new_obj = super().__getitem__(item)

        if not isinstance(new_obj, Data):
            return new_obj

        self.__add_slices(new_obj, item)

        params = {
            'filename': new_obj.filename,
            'name': new_obj.name,
            'xslice': new_obj.xslice,
            'yslice': new_obj.yslice,
            'zslice': new_obj.zslice,
            'tslice': new_obj.tslice,
            'slice_axes': new_obj.slice_axes,
            'axisunits': new_obj.axisunits,
            'valunit': new_obj.valunit
        }

        if len(new_obj.shape) == 1:
            if isinstance(new_obj, Data1d):
                return new_obj
            return Data1d(new_obj, **params)
        elif len(new_obj.shape) == 2:
            if isinstance(new_obj, Data2d):
                return new_obj
            return Data2d(new_obj, **params)
        elif len(new_obj.shape) == 3:
            if isinstance(new_obj, Data3d):
                return new_obj
            return Data3d(new_obj, **params)
        elif len(new_obj.shape) == 4:
            if isinstance(new_obj, Data4d):
                return new_obj
            return Data4d(new_obj, **params)
        else:
            return new_obj

    def __add_slices(self, new_obj, item):
        """管理するデータの範囲を新しいオブジェクトに追加する.

        Parameters
        ----------
        new_obj : Data
            新しく生成されたデータオブジェクト
        item : int or slice or tuple(int or slice)
            スライス
        """
        slices = [*self.slices]
        axes = [*self.slice_axes]
        for i, axis in enumerate(axes):
            if i < len(item):
                slice_obj = item[i]
            else:
                continue

            if not isinstance(slice_obj, slice):
                slice_obj = slice(slice_obj, slice_obj+1, 1)
                axes[i] = -1

            obj_start = slice_obj.start
            obj_stop = slice_obj.stop
            obj_step = slice_obj.step

            new_start = self.slices[axis].start
            new_stop = self.slices[axis].stop
            new_step = self.slices[axis].step

            if obj_start is not None:
                if obj_start < 0:
                    obj_start = self.shape[i] + obj_start
                new_start += self.slices[axis].step * obj_start

            if slice_obj.stop is not None:
                if obj_stop < 0:
                    obj_stop = self.shape[i] + obj_stop
                new_stop = self.slices[axis].start + \
                    self.slices[axis].step * obj_stop

            if obj_step is not None:
                new_step *= obj_step

            slices[axis] = slice(new_start, new_stop, new_step)

        axes = [axis for axis in axes if axis != -1]
        setattr(new_obj, 'slices', slices)
        setattr(new_obj, 'slice_axes', axes)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.datafile = getattr(obj, 'datafile', None)
        self.name = getattr(obj, 'name', None)
        self.slices = getattr(obj, 'slices', None)
        self.slice_axes = getattr(obj, 'slice_axes', None)
        self.axisunits = getattr(obj, 'axisunits', None)
        self.valunit = getattr(obj, 'valunit', None)

    @property
    def filename(self):
        """ファイル名を返す.

        Returns
        -------
        Path
            ファイル名.
        """
        return self.datafile.filename

    @property
    def directory(self):
        """ディレクトリ名を返す

        Returns
        -------
        Path
            ディレクトリ名
        """
        return self.datafile.directory

    @ property
    def xslice(self):
        """管理するx方向の範囲を返す.

        Returns
        -------
        slice
            管理するx方向の範囲
        """
        return self.slices[3]

    @ property
    def yslice(self):
        """管理するy方向の範囲を返す.

        Returns
        -------
        slice
            管理するy方向の範囲
        """
        return self.slices[2]

    @ property
    def zslice(self):
        """管理するz方向の範囲を返す.

        Returns
        -------
        slice
            管理するz方向の範囲
        """
        return self.slices[1]

    @property
    def tslice(self):
        """管理するt方向の範囲を返す.

        Returns
        -------
        slice
            管理するt方向の範囲
        """
        return self.slices[0]

    def axis(self, ax):
        index = self.slice_axes[ax]
        axis_slice = self.slices[index]
        return np.array(*utils.slice2tuple(axis_slice))

    @property
    def x(self):
        """x軸.

        Returns
        -------
        np.ndarray
            x軸
        """
        return np.arange(*utils.slice2tuple(self.xslice))

    @property
    def y(self):
        """y軸.

        Returns
        -------
        np.ndarray
            y軸
        """
        return np.arange(*utils.slice2tuple(self.yslice))

    @property
    def z(self):
        """z軸.

        Returns
        -------
        np.ndarray
            z軸
        """
        return np.arange(*utils.slice2tuple(self.zslice))

    @property
    def t(self):
        """t軸.

        Returns
        -------
        np.ndarray
            t軸
        """
        slc = self.tslice
        maxlen = (slc.stop - slc.start) // slc.step
        return np.array(utils.range_with_slice(self.tslice, maxlen=maxlen))

    @property
    def x_si(self):
        """SI単位系でのx軸.

        Returns
        -------
        np.ndarray
            SI単位系でのx軸
        """
        return self.axisunits[3].reverse(self.x)

    @property
    def y_si(self):
        """SI単位系でのy軸.

        Returns
        -------
        np.ndarray
            SI単位系でのy軸
        """
        return self.axisunits[2].reverse(self.y)

    @property
    def z_si(self):
        """SI単位系でのz軸.

        Returns
        -------
        np.ndarray
            SI単位系でのz軸
        """
        return self.axisunits[1].reverse(self.z)

    @property
    def t_si(self):
        """SI単位系でのt軸.

        Returns
        -------
        np.ndarray
            SI単位系でのt軸
        """
        return self.axisunits[0].reverse(self.t)

    @property
    def val_si(self):
        """SI単位系での値.

        Returns
        -------
        np.ndarray
            SI単位系での値
        """
        return self.valunit.reverse(self)

    @ property
    def use_axes(self):
        """データ軸がxyzのどの方向に対応しているか表すリストを返す.

        Returns
        -------
        list(str)
            データ軸がxyzのどの方向に対応しているか表すリスト(['x'], ['x', 'z'], etc)
        """
        to_axis = {3: 'x', 2: 'y', 1: 'z', 0: 't'}
        return list(map(lambda a: to_axis[a], self.slice_axes))

    def masked(self, mask):
        """マスクされたデータを返す.

        Parameters
        ----------
        mask : numpy.ndarray or predicate
            マスク行列またはマスクを返す関数

        Returns
        -------
        SlicedData
            マスクされたデータ
        """
        masked = self.copy()
        if isinstance(mask, np.ndarray):
            masked[mask] = np.nan
        else:
            masked[mask(masked)] = np.nan
        return masked

    def plot(self, **kwargs):
        """データをプロットする.
        """
        raise NotImplementedError()

    def gifplot(self,
                fig=None,
                axis=0,
                show=False,
                savefilename=None,
                interval=200,
                repeat=True,
                title=None,
                notitle=False,
                offsets=None,
                use_si=False,
                vmin=None,
                vmax=None,
                **kwargs):
        """gifアニメーションを作成する

        Parameters
        ----------
        fig : Figure
            アニメーションを描画するFigure(Noneの場合新しく作成する), by default None
        axis : int, optional
            アニメーションする軸, by default 0
        show : bool, optional
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default False
        savefilename : str, optional
            保存するファイル名(Noneの場合保存しない), by default None
        interval : int, optional
            フレーム間のインターバル(ミリ秒), by default 400
        repeat : bool
            アニメーションをループするならTrue, by default True
        title : str, optional
            タイトル(Noneの場合データ名(phisp等)), by default None
        notitle : bool, optional
            タイトルを付けない場合True, by default False
        offsets : (float or str, float or str, float or str)
            プロットのx,y,z軸のオフセット('left': 最初を0, 'center': 中心を0, 'right': 最後尾を0, float: 値だけずらす), by default None
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default False
        """
        def _offseted(line, offset):
            if offset == 'left':
                line -= line[0]
            elif offset == 'center':
                line -= line[len(line) // 2]
            elif offset == 'right':
                line -= line[-1]
            else:
                line += offset
            return line

        def _update(i, vmin, vmax):
            plt.clf()

            # 指定した軸でスライス
            slices = [slice(None)] * len(self.shape)
            slices[axis] = i
            val = self[tuple(slices)]

            # タイトルの設定
            if notitle:
                _title = title if len(title) > 0 else None
            else:
                ax = self.slice_axes[axis]
                slc = self.slices[ax]
                maxlen = self.shape[axis]

                line = np.array(utils.range_with_slice(slc, maxlen=maxlen))

                if offsets is not None:
                    line = _offseted(line, offsets[0])

                index = line[i]

                if use_si:  # SI単位系を用いる場合
                    title_format = title + '({} {})'
                    axisunit = self.axisunits[ax]
                    _title = title_format.format(
                        axisunit.reverse(index), axisunit.unit)

                else:  # EMSES単位系を用いる場合
                    title_format = title + '({})'
                    _title = title_format.format(index)

            if offsets is not None:
                offsets2d = offsets[1:]
            else:
                offsets2d = None

            val.plot(vmin=vmin, vmax=vmax, title=_title,
                     use_si=use_si, offsets=offsets2d, **kwargs)

        if title is None:
            title = self.name

        if use_si:
            vmin = vmin or self.valunit.reverse(self.min())
            vmax = vmax or self.valunit.reverse(self.max())
        else:
            vmin = vmin or self.min()
            vmax = vmax or self.max()

        if fig is None:
            fig = plt.figure()

        ani = animation.FuncAnimation(
            fig,
            _update,
            fargs=(vmin, vmax),
            interval=interval,
            frames=self.shape[axis],
            repeat=repeat)

        if savefilename is not None:
            ani.save(savefilename, writer='quantized-pillow')
        elif show:
            plt.show()
        else:
            return fig, ani


class Data4d(Data):
    """4次元データを管理する.
    """
    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)

        if 'xslice' not in kwargs:
            kwargs['xslice'] = slice(0, obj.shape[3], 1)
        if 'yslice' not in kwargs:
            kwargs['yslice'] = slice(0, obj.shape[2], 1)
        if 'zslice' not in kwargs:
            kwargs['zslice'] = slice(0, obj.shape[1], 1)
        if 'tslice' not in kwargs:
            kwargs['tslice'] = slice(0, obj.shape[0], 1)
        if 'slice_axes' not in kwargs:
            kwargs['slice_axes'] = [0, 1, 2, 3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(mode='auto', **kwargs):
        """3次元データをプロットする.(未実装)

        Parameters
        ----------
        mode : str, optional
            [description], by default 'auto'
        """
        if mode == 'auto':
            mode = ''.join(sorted(self.use_axes))
        pass


class Data3d(Data):
    """3次元データを管理する.
    """
    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)

        if 'xslice' not in kwargs:
            kwargs['xslice'] = slice(0, obj.shape[2], 1)
        if 'yslice' not in kwargs:
            kwargs['yslice'] = slice(0, obj.shape[1], 1)
        if 'zslice' not in kwargs:
            kwargs['zslice'] = slice(0, obj.shape[0], 1)
        if 'tslice' not in kwargs:
            kwargs['tslice'] = slice(0, 1, 1)
        if 'slice_axes' not in kwargs:
            kwargs['slice_axes'] = [1, 2, 3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(mode='auto', **kwargs):
        """3次元データをプロットする.(未実装)

        Parameters
        ----------
        mode : str, optional
            [description], by default 'auto'
        """
        if mode == 'auto':
            mode = ''.join(sorted(self.use_axes))
        pass


class Data2d(Data):
    """3次元データの2次元面を管理する.
    """
    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)

        if 'xslice' not in kwargs:
            kwargs['xslice'] = slice(0, obj.shape[1], 1)
        if 'yslice' not in kwargs:
            kwargs['yslice'] = slice(0, obj.shape[0], 1)
        if 'zslice' not in kwargs:
            kwargs['zslice'] = slice(0, 1, 1)
        if 'tslice' not in kwargs:
            kwargs['tslice'] = slice(0, 1, 1)
        if 'slice_axes' not in kwargs:
            kwargs['slice_axes'] = [2, 3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(self, axes='auto', show=False, use_si=False, offsets=None, mode='cm', **kwargs):
        """2次元データをプロットする.

        Parameters
        ----------
        axes : str, optional
            プロットする軸('xy', 'zx', etc), by default 'auto'
        show : bool
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default False
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default False
        offsets : (float or str, float or str, float or str)
            プロットのx,y,z軸のオフセット('left': 最初を0, 'center': 中心を0, 'right': 最後尾を0, float: 値だけずらす), by default None
        mode : str
            プロットの種類('cm': カラーマップ, 'cont': 等高線プロット, 'surf': サーフェースプロット)
        mesh : (numpy.ndarray, numpy.ndarray), optional
            メッシュ, by default None
        savefilename : str, optional
            保存するファイル名(Noneの場合保存しない), by default None
        cmap : matplotlib.Colormap or str or None, optional
            カラーマップ, by default cm.coolwarm
        vmin : float, optional
            最小値, by default None
        vmax : float, optional
            最大値, by default None
        figsize : (float, float), optional
            図のサイズ, by default None
        xlabel : str, optional
            x軸のラベル, by default None
        ylabel : str, optional
            y軸のラベル, by default None
        title : str, optional
            タイトル, by default None
        interpolation : str, optional
            用いる補間方法, by default 'bilinear'
        dpi : int, optional
            解像度(figsizeが指定された場合は無視される), by default 10

        Returns
        -------
        AxesImage or None
            プロットしたimageデータ(保存またはshowした場合None)

        Raises
        ------
        Exception
            プロットする軸のパラメータが間違っている場合の例外
        Exception
            プロットする軸がデータにない場合の例外
        Exception
            データの次元が2でない場合の例外
        """
        if axes == 'auto':
            axes = ''.join(sorted(self.use_axes))

        if not re.match(r'x[yzt]|y[xzt]|z[xyt]|t[xyz]', axes):
            raise Exception(
                'Error: axes "{axes}" cannot be used with Data2d'.format(axes=axes))
        if axes[0] not in self.use_axes or axes[1] not in self.use_axes:
            raise Exception(
                'Error: axes "{axes}" cannot be used because {axes}-axis does not exist in this data.'.format(axes=axes))
        if len(self.shape) != 2:
            raise Exception(
                'Error: axes "{axes}" cannot be used because data is not 2dim shape.'.format(axes=axes))

        # x: 3, y: 2, z:1 t:0
        axis1 = self.slice_axes[self.use_axes.index(axes[0])]
        axis2 = self.slice_axes[self.use_axes.index(axes[1])]

        x = np.arange(*utils.slice2tuple(self.slices[axis1]))
        y = np.arange(*utils.slice2tuple(self.slices[axis2]))
        z = self if axis1 > axis2 else self.T  # 'xz'等の場合は転置

        if use_si:
            xunit = self.axisunits[axis1]
            yunit = self.axisunits[axis2]

            x = xunit.reverse(x)
            y = yunit.reverse(y)
            z = self.valunit.reverse(z)

            _xlabel = '{} [{}]'.format(axes[0], xunit.unit)
            _ylabel = '{} [{}]'.format(axes[1], yunit.unit)
            _title = '{} [{}]'.format(self.name, self.valunit.unit)
        else:
            _xlabel = axes[0]
            _ylabel = axes[1]
            _title = self.name

        def _offseted(line, offset):
            if offset == 'left':
                line -= line.ravel()[0]
            elif offset == 'center':
                line -= line.ravel()[line.size // 2]
            elif offset == 'right':
                line -= line.ravel()[-1]
            else:
                line += offset
            return line

        kwargs['xlabel'] = kwargs.get('xlabel', None) or _xlabel
        kwargs['ylabel'] = kwargs.get('ylabel', None) or _ylabel
        kwargs['title'] = kwargs.get('title', None) or _title

        if mode == 'surf':
            mesh = np.meshgrid(x, y)

            kwargs['zlabel'] = kwargs.get('zlabel', None) or _title
            val = z
            if 'x' not in self.use_axes:
                y, z = mesh
                x = self.x_si[0] if use_si else self.x[0]
                x = np.zeros_like(mesh[0]) + x
            elif 'y' not in self.use_axes:
                x, z = mesh
                y = self.y_si[0] if use_si else self.y[0]
                y = np.zeros_like(mesh[0]) + y
            elif 'z' not in self.use_axes:
                x, y = mesh
                z = self.z_si[0] if use_si else self.z[0]
                z = np.zeros_like(mesh[0]) + z

            if offsets is not None:
                x = _offseted(x, offsets[0])
                y = _offseted(y, offsets[1])
                z = _offseted(z, offsets[2])
                val = _offseted(val, offsets[3])

            imgs = emplt.plot_surface(x, y, z, val, **kwargs)
        else:
            if offsets is not None:
                x = _offseted(x, offsets[0])
                y = _offseted(y, offsets[1])
                z = _offseted(z, offsets[2])
            mesh = np.meshgrid(x, y)
            
            imgs = []
            if 'cm' in mode:
                img = emplt.plot_2dmap(z, mesh=mesh, **kwargs)
                imgs.append(img)

            if 'cont' in mode:
                img = emplt.plot_2d_contour(z, mesh=mesh, **kwargs)
                imgs.append(img)

        if show:
            plt.show()
            return None
        else:
            return imgs[0] if len(imgs) == 1 else imgs


class Data1d(Data):
    """3次元データの1次元直線を管理する.
    """
    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)

        if 'xslice' not in kwargs:
            kwargs['xslice'] = slice(0, obj.shape[1], 1)
        if 'yslice' not in kwargs:
            kwargs['yslice'] = slice(0, 1, 1)
        if 'zslice' not in kwargs:
            kwargs['zslice'] = slice(0, 1, 1)
        if 'tslice' not in kwargs:
            kwargs['tslice'] = slice(0, 1, 1)
        if 'slice_axes' not in kwargs:
            kwargs['slice_axes'] = [3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(self, show=False, use_si=False, offsets=None, **kwargs):
        """1次元データをプロットする.

        Parameters
        ----------
        show : bool
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default False
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default False
        offsets : (float or str, float or str)
            プロットのx,y軸のオフセット('left': 最初を0, 'center': 中心を0, 'right': 最後尾を0, float: 値だけずらす), by default None
        savefilename : str, optional
            保存するファイル名, by default None
        vmin : float, optional
            最小値, by default None
        vmax : float, optional
            最大値, by default None
        figsize : (float, float), optional
            図のサイズ, by default None
        xlabel : str, optional
            横軸のラベル, by default None
        ylabel : str, optional
            縦軸のラベル, by default None
        label : str, optional
            ラベル, by default None
        title : str, optional
            タイトル, by default None

        Returns
        -------
        Line2D or None
            プロットデータを表す線オブジェクト(保存または show した場合None)

        Raises
        ------
        Exception
            データの次元が1でない場合の例外
        """
        if len(self.shape) != 1:
            raise Exception(
                'Error: cannot plot because data is not 1dim shape.')

        axis = self.slice_axes[0]
        x = np.arange(*utils.slice2tuple(self.slices[axis]))
        y = self

        # "EMSES Unit" to "Physical Unit"
        if use_si:
            xunit = self.axisunits[axis]

            x = xunit.reverse(x)
            y = self.valunit.reverse(y)

            _xlabel = '{} [{}]'.format(self.use_axes[0], xunit.unit)
            _ylabel = '{} [{}]'.format(self.name, self.valunit.unit)
        else:
            _xlabel = self.use_axes[0]
            _ylabel = self.name

        def _offseted(line, offset):
            if offset == 'left':
                line -= line[0]
            elif offset == 'center':
                line -= line[len(line) // 2]
            elif offset == 'right':
                line -= line[-1]
            else:
                line += offset
            return line

        if offsets is not None:
            x = _offseted(x, offsets[0])
            y = _offseted(y, offsets[1])

        kwargs['xlabel'] = kwargs.get('xlabel', None) or _xlabel
        kwargs['ylabel'] = kwargs.get('ylabel', None) or _ylabel

        line = emplt.plot_line(y, x=x, **kwargs)

        if show:
            plt.show()
            return None
        else:
            return line


class VectorData2d(utils.Group):
    def __init__(self, objs):
        x_data, y_data = objs
        super().__init__([x_data, y_data])
        self.x_data = x_data
        self.y_data = y_data

    def __setattr__(self, key, value):
        if key in ('x_data', 'y_data'):
            super().__dict__[key] = value
            return
        super().__setattr__(key, value)

    def plot(self, axes='auto', show=False, use_si=False, offsets=None, **kwargs):
        """2次元データをプロットする.

        Parameters
        ----------
        axes : str, optional
            プロットする軸('xy', 'zx', etc), by default 'auto'
        show : bool
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default False
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default False
        offsets : (float or str, float or str, float or str)
            プロットのx,y,z軸のオフセット('left': 最初を0, 'center': 中心を0, 'right': 最後尾を0, float: 値だけずらす), by default None
        mesh : (numpy.ndarray, numpy.ndarray), optional
            メッシュ, by default None
        savefilename : str, optional
            保存するファイル名(Noneの場合保存しない), by default None
        cmap : matplotlib.Colormap or str or None, optional
            カラーマップ, by default cm.coolwarm
        vmin : float, optional
            最小値, by default None
        vmax : float, optional
            最大値, by default None
        figsize : (float, float), optional
            図のサイズ, by default None
        xlabel : str, optional
            x軸のラベル, by default None
        ylabel : str, optional
            y軸のラベル, by default None
        title : str, optional
            タイトル, by default None
        interpolation : str, optional
            用いる補間方法, by default 'bilinear'
        dpi : int, optional
            解像度(figsizeが指定された場合は無視される), by default 10

        Returns
        -------
        AxesImage or None
            プロットしたimageデータ(保存またはshowした場合None)

        Raises
        ------
        Exception
            プロットする軸のパラメータが間違っている場合の例外
        Exception
            プロットする軸がデータにない場合の例外
        Exception
            データの次元が2でない場合の例外
        """
        if axes == 'auto':
            axes = ''.join(sorted(self.objs[0].use_axes))

        if not re.match(r'x[yzt]|y[xzt]|z[xyt]|t[xyz]', axes):
            raise Exception(
                'Error: axes "{axes}" cannot be used with Data2d'.format(axes=axes))
        if axes[0] not in self.objs[0].use_axes or axes[1] not in self.objs[0].use_axes:
            raise Exception(
                'Error: axes "{axes}" cannot be used because {axes}-axis does not exist in this data.'.format(axes=axes))
        if len(self.objs[0].shape) != 2:
            raise Exception(
                'Error: axes "{axes}" cannot be used because data is not 2dim shape.'.format(axes=axes))

        # x: 3, y: 2, z:1 t:0
        axis1 = self.objs[0].slice_axes[self.objs[0].use_axes.index(axes[0])]
        axis2 = self.objs[0].slice_axes[self.objs[0].use_axes.index(axes[1])]

        x = np.arange(*utils.slice2tuple(self.objs[0].slices[axis1]))
        y = np.arange(*utils.slice2tuple(self.objs[0].slices[axis2]))

        if use_si:
            xunit = self.objs[0].axisunits[axis1]
            yunit = self.objs[0].axisunits[axis2]
            valunit = self.objs[0].valunit

            x = xunit.reverse(x)
            y = yunit.reverse(y)

            _xlabel = '{} [{}]'.format(axes[0], xunit.unit)
            _ylabel = '{} [{}]'.format(axes[1], yunit.unit)
            _title = '{} [{}]'.format(
                self.objs[0].name, valunit.unit)

            x_data = self.x_data.val_si
            y_data = self.y_data.val_si
        else:
            _xlabel = axes[0]
            _ylabel = axes[1]
            _title = self.objs[0].name

            x_data = self.x_data
            y_data = self.y_data

        def _offseted(line, offset):
            if offset == 'left':
                line -= line[0]
            elif offset == 'center':
                line -= line[len(line) // 2]
            elif offset == 'right':
                line -= line[-1]
            else:
                line += offset
            return line

        if offsets is not None:
            x = _offseted(x, offsets[0])
            y = _offseted(y, offsets[1])

        kwargs['xlabel'] = kwargs.get('xlabel', None) or _xlabel
        kwargs['ylabel'] = kwargs.get('ylabel', None) or _ylabel
        kwargs['title'] = kwargs.get('title', None) or _title

        mesh = np.meshgrid(x, y)
        img = emplt.plot_2d_vector(
            x_data, y_data, mesh=mesh, **kwargs)

        if show:
            plt.show()
            return None
        else:
            return img
