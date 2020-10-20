import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import emout.plot as emplt
import emout.utils as utils
from emout.utils import InpFile, UnitConversionKey, Units


class Emout:
    """EMSES出力・inpファイルを管理する.

    Attributes
    ----------
    directory : Path
        管理するディレクトリ
    dataname : GridData
        3次元データ(datanameは"phisp"などのhdf5ファイルの先頭の名前)
    """

    def __init__(self, directory='./', inpfilename='plasma.inp'):
        """[summary]

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

        for h5file_path in self.directory.glob('*.h5'):
            name = str(h5file_path.name).replace('00_0000.h5', '')
            setattr(self, name, GridDataSeries(h5file_path, name))

        if inpfilename is not None:
            self._inp = InpFile(directory / inpfilename)
            convkey = UnitConversionKey.load(directory / inpfilename)
            if convkey is not None:
                self._unit = Units(dx=convkey.dx, to_c=convkey.to_c)

    @property
    def inp(self):
        """パラメータの辞書(Namelist)を返す.

        Returns
        -------
        InpFile or None
            パラメータの辞書(Namelist)
        """
        try:
            return self._inp
        except AttributeError:
            return None

    @property
    def unit(self):
        """単位変換オブジェクトを返す.

        Returns
        -------
        Units or None
            単位変換オブジェクト
        """
        try:
            return self._unit
        except:
            return None


class GridDataSeries:
    """3次元時系列データを管理する.

    Attributes
    ----------
    h5 : h5py.File
        hdf5ファイルオブジェクト
    group : h5py.Datasets
        データセット
    name : str
        データセット名
    """

    def __init__(self, filename, name):
        self.h5 = h5py.File(str(filename), 'r')
        self.group = self.h5[list(self.h5.keys())[0]]
        self._index2key = {int(key): key for key in self.group.keys()}

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

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError()
        if index not in self._index2key:
            raise IndexError()
        key = self._index2key[index]
        name = "{} {}".format(self.name, index)
        return GridData(np.array(self.group[key]), name=name)

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
    name : str
        データ名
    xslice : slice
        管理するデータのx方向の範囲
    yslice : slice
        管理するデータのy方向の範囲
    zslice : slice
        管理するデータのz方向の範囲
    slice_axes : list(int)
        データ軸がxyzのどの方向に対応しているか表すリスト(0: z, 1: y, 2: x)
    """
    def __new__(cls, input_array, name=None, xslice=None, yslice=None, zslice=None, slice_axes=None):
        obj = np.asarray(input_array).view(cls)
        obj.name = name

        if xslice is None:
            xslice = slice(0, obj.shape[2], 1)
        if yslice is None:
            yslice = slice(0, obj.shape[1], 1)
        if zslice is None:
            zslice = slice(0, obj.shape[0], 1)
        if slice_axes is None:
            slice_axes = [0, 1, 2]

        obj.slices = [zslice, yslice, xslice]
        obj.slice_axes = slice_axes

        return obj

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item, )

        new_obj = super().__getitem__(item)

        if not isinstance(new_obj, Data):
            return new_obj

        self.__add_slices(new_obj, item)

        if len(new_obj.shape) == 1:
            if isinstance(new_obj, LineData):
                return new_obj
            return LineData(new_obj,
                            name=new_obj.name,
                            xslice=new_obj.xslice,
                            yslice=new_obj.yslice,
                            zslice=new_obj.zslice,
                            slice_axes=new_obj.slice_axes)
        elif len(new_obj.shape) == 2:
            if isinstance(new_obj, SlicedData):
                return new_obj
            return SlicedData(new_obj,
                              name=new_obj.name,
                              xslice=new_obj.xslice,
                              yslice=new_obj.yslice,
                              zslice=new_obj.zslice,
                              slice_axes=new_obj.slice_axes)
        elif len(new_obj.shape) == 3:
            if isinstance(new_obj, GridData):
                return new_obj
            return GridData(new_obj,
                            name=new_obj.name,
                            xslice=new_obj.xslice,
                            yslice=new_obj.yslice,
                            zslice=new_obj.zslice,
                            slice_axes=new_obj.slice_axes)
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
        self.name = getattr(obj, 'name', None)
        self.slices = getattr(obj, 'slices', None)
        self.slice_axes = getattr(obj, 'slice_axes', None)

    @ property
    def xslice(self):
        """管理するx方向の範囲を返す.

        Returns
        -------
        slice
            管理するx方向の範囲
        """
        return self.slices[2]

    @ property
    def yslice(self):
        """管理するy方向の範囲を返す.

        Returns
        -------
        slice
            管理するy方向の範囲
        """
        return self.slices[1]

    @ property
    def zslice(self):
        """管理するz方向の範囲を返す.

        Returns
        -------
        slice
            管理するz方向の範囲
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
        to_axis = {2: 'x', 1: 'y', 0: 'z'}
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


class GridData(Data):
    """3次元データを管理する.
    """
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


class SlicedData(Data):
    """3次元データの2次元面を管理する.
    """

    def plot(self, axes='auto', show=False, **kwargs):
        """2次元データをプロットする.

        Parameters
        ----------
        axes : str, optional
            プロットする軸('xy', 'zx', etc), by default 'auto'
        show : bool
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default False
        data2d : numpy.ndarray
            2次元データ
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

        if not re.match(r'x[yz]|y[xz]|z[xy]', axes):
            raise Exception(
                'Error: axes "{axes}" cannot be used with SlicedData'.format(axes=axes))
        if axes[0] not in self.use_axes or axes[1] not in self.use_axes:
            raise Exception(
                'Error: axes "{axes}" cannot be used because {axes}-axis does not exist in this data.'.format(axes=axes))
        if len(self.shape) != 2:
            raise Exception(
                'Error: axes "{axes}" cannot be used because data is not 2dim shape.'.format(axes=axes))

        # x: 2, y: 1, z:0
        axis1 = self.slice_axes[self.use_axes.index(axes[0])]
        axis2 = self.slice_axes[self.use_axes.index(axes[1])]
        x = np.arange(*utils.slice2tuple(self.slices[axis1]))
        y = np.arange(*utils.slice2tuple(self.slices[axis2]))

        kwargs['xlabel'] = kwargs.get('xlabel', None) or axes[0]
        kwargs['ylabel'] = kwargs.get('ylabel', None) or axes[1]
        kwargs['title'] = kwargs.get('title', None) or self.name

        mesh = np.meshgrid(x, y)
        if axis1 > axis2:
            emplt.plot_2dmap(self, mesh=mesh, **kwargs)
        else:
            emplt.plot_2dmap(self.T, mesh=mesh, **kwargs)

        if show:
            plt.show()


class LineData(Data):
    """3次元データの1次元直線を管理する.
    """

    def plot(self, show=False, **kwargs):
        """1次元データをプロットする.

        Parameters
        ----------
        show : bool
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default False
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
        horizon_data = np.arange(*utils.slice2tuple(self.slices[axis]))

        kwargs['xlabel'] = kwargs.get('xlabel', None) or self.use_axes[0]
        kwargs['ylabel'] = kwargs.get('ylabel', None) or self.name

        emplt.plot_line(self, x=horizon_data, **kwargs)
        if show:
            plt.show()
