import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import emout.plot as emplt
import emout.utils as utils
from emout.utils import InpFile, UnitConversionKey, Units, DataFileInfo, RegexDict


def none_unit(out):
    return utils.UnitTranslator(
        1,
        1,
        name='',
        unit=''
    )


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
        r'nd[0-9]+p': none_unit,
        r'rho': lambda self: self.unit.rho,
        r'rhobk': lambda self: self.unit.rho,
        r'j.*': lambda self: self.unit.J,
        r'b[xyz]': lambda self: self.unit.H,
        r'e[xyz]': lambda self: self.unit.E,
        r't': lambda self: self.unit.t,
        r'axis': lambda self: self.unit.length,
    })

    def __init__(self, directory='./', inpfilename='plasma.inp'):
        """EMSES出力・inpファイルを管理するオブジェクトを生成する.

        Parameters
        ----------
        directory : str or Path
            管理するディレクトリ, by default './'
        inpfilename : str, optional
            パラメータファイルの名前, by default 'plasma.inp'
        """
        if not isinstance(directory, Path):
            directory = Path(directory)
        self.directory = directory

        # パラメータファイルの読み取りと単位変換器の初期化
        self._inp = None
        self._unit = None
        if inpfilename is not None and (directory / inpfilename).exists():
            self._inp = InpFile(directory / inpfilename)
            convkey = UnitConversionKey.load(directory / inpfilename)
            if convkey is not None:
                self._unit = Units(dx=convkey.dx, to_c=convkey.to_c)

        for h5file_path in self.directory.glob('*.h5'):
            name = str(h5file_path.name).replace('00_0000.h5', '')

            if self.unit is None:
                tunit = None
                axisunit = None
                valunit = None
            else:
                tunit = Emout.name2unit.get('t', lambda self: None)(self)
                axisunit = Emout.name2unit.get('axis', lambda self: None)(self)
                valunit = Emout.name2unit.get(name, lambda self: None)(self)

            series = GridDataSeries(h5file_path,
                                    name,
                                    tunit=tunit,
                                    axisunit=axisunit,
                                    valunit=valunit)
            setattr(self, name, series)

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

    def __create_data_with_index(self, index):
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
            if item < 0:
                item = len(self) + item
            return self.__create_data_with_index(item)

        elif isinstance(item, slice):  # tがスライスで指定された場合
            indexes = list(utils.range_with_slice(item, maxlen=len(self)))
            return self.__create_data_with_indexes(indexes, tslice=item)

        elif isinstance(item, list):  # tがリストで指定された場合
            return self.__create_data_with_indexes(item)

        else:
            raise TypeError()

    def __iter__(self):
        indexes = sorted(self._index2key.keys())
        for index in indexes:
            yield self[index]

    def __len__(self):
        return len(self._index2key)


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
    axisunit : UnitTranslator or None
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

    def plot(self, axes='auto', show=False, use_si=False, **kwargs):
        """2次元データをプロットする.

        Parameters
        ----------
        axes : str, optional
            プロットする軸('xy', 'zx', etc), by default 'auto'
        show : bool
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default False
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default False
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

        kwargs['xlabel'] = kwargs.get('xlabel', None) or _xlabel
        kwargs['ylabel'] = kwargs.get('ylabel', None) or _ylabel
        kwargs['title'] = kwargs.get('title', None) or _title

        mesh = np.meshgrid(x, y)
        emplt.plot_2dmap(z, mesh=mesh, **kwargs)

        if show:
            plt.show()


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

    def plot(self, show=False, use_si=False, **kwargs):
        """1次元データをプロットする.

        Parameters
        ----------
        show : bool
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default False
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default False
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

        kwargs['xlabel'] = kwargs.get('xlabel', None) or _xlabel
        kwargs['ylabel'] = kwargs.get('ylabel', None) or _ylabel

        emplt.plot_line(y, x=x, **kwargs)

        if show:
            plt.show()
