import re
import warnings
from os import PathLike
from pathlib import Path
from typing import Callable, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import emout.utils as utils
from emout.plot.animation_plot import ANIMATER_PLOT_MODE, FrameUpdater
from emout.utils import DataFileInfo


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

    def __new__(
        cls,
        input_array,
        filename=None,
        name=None,
        xslice=None,
        yslice=None,
        zslice=None,
        tslice=None,
        slice_axes=None,
        axisunits=None,
        valunit=None,
    ):
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
            item = (item,)

        new_obj = super().__getitem__(item)

        if not isinstance(new_obj, Data):
            return new_obj

        self.__add_slices(new_obj, item)

        params = {
            "filename": new_obj.filename,
            "name": new_obj.name,
            "xslice": new_obj.xslice,
            "yslice": new_obj.yslice,
            "zslice": new_obj.zslice,
            "tslice": new_obj.tslice,
            "slice_axes": new_obj.slice_axes,
            "axisunits": new_obj.axisunits,
            "valunit": new_obj.valunit,
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
                slice_obj = slice(slice_obj, slice_obj + 1, 1)
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
                new_stop = self.slices[axis].start + self.slices[axis].step * obj_stop

            if obj_step is not None:
                new_step *= obj_step

            slices[axis] = slice(new_start, new_stop, new_step)

        axes = [axis for axis in axes if axis != -1]
        setattr(new_obj, "slices", slices)
        setattr(new_obj, "slice_axes", axes)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.datafile = getattr(obj, "datafile", None)
        self.name = getattr(obj, "name", None)
        self.slices = getattr(obj, "slices", None)
        self.slice_axes = getattr(obj, "slice_axes", None)
        self.axisunits = getattr(obj, "axisunits", None)
        self.valunit = getattr(obj, "valunit", None)

    @property
    def filename(self) -> Path:
        """ファイル名を返す.

        Returns
        -------
        Path
            ファイル名.
        """
        return self.datafile.filename

    @property
    def directory(self) -> Path:
        """ディレクトリ名を返す

        Returns
        -------
        Path
            ディレクトリ名
        """
        return self.datafile.directory

    @property
    def xslice(self) -> slice:
        """管理するx方向の範囲を返す.

        Returns
        -------
        slice
            管理するx方向の範囲
        """
        return self.slices[3]

    @property
    def yslice(self) -> slice:
        """管理するy方向の範囲を返す.

        Returns
        -------
        slice
            管理するy方向の範囲
        """
        return self.slices[2]

    @property
    def zslice(self) -> slice:
        """管理するz方向の範囲を返す.

        Returns
        -------
        slice
            管理するz方向の範囲
        """
        return self.slices[1]

    @property
    def tslice(self) -> slice:
        """管理するt方向の範囲を返す.

        Returns
        -------
        slice
            管理するt方向の範囲
        """
        return self.slices[0]

    def axis(self, ax: int) -> np.ndarray:
        index = self.slice_axes[ax]
        axis_slice = self.slices[index]
        return np.array(*utils.slice2tuple(axis_slice))

    @property
    def x(self) -> np.ndarray:
        """x軸.

        Returns
        -------
        np.ndarray
            x軸
        """
        return np.arange(*utils.slice2tuple(self.xslice))

    @property
    def y(self) -> np.ndarray:
        """y軸.

        Returns
        -------
        np.ndarray
            y軸
        """
        return np.arange(*utils.slice2tuple(self.yslice))

    @property
    def z(self) -> np.ndarray:
        """z軸.

        Returns
        -------
        np.ndarray
            z軸
        """
        return np.arange(*utils.slice2tuple(self.zslice))

    @property
    def t(self) -> np.ndarray:
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
    def x_si(self) -> np.ndarray:
        """SI単位系でのx軸.

        Returns
        -------
        np.ndarray
            SI単位系でのx軸
        """
        return self.axisunits[3].reverse(self.x)

    @property
    def y_si(self) -> np.ndarray:
        """SI単位系でのy軸.

        Returns
        -------
        np.ndarray
            SI単位系でのy軸
        """
        return self.axisunits[2].reverse(self.y)

    @property
    def z_si(self) -> np.ndarray:
        """SI単位系でのz軸.

        Returns
        -------
        np.ndarray
            SI単位系でのz軸
        """
        return self.axisunits[1].reverse(self.z)

    @property
    def t_si(self) -> np.ndarray:
        """SI単位系でのt軸.

        Returns
        -------
        np.ndarray
            SI単位系でのt軸
        """
        return self.axisunits[0].reverse(self.t)

    @property
    def val_si(self) -> "Data":
        """SI単位系での値.

        Returns
        -------
        Data
            SI単位系での値
        """
        return self.valunit.reverse(self)

    @property
    def use_axes(self) -> List[str]:
        """データ軸がxyzのどの方向に対応しているか表すリストを返す.

        Returns
        -------
        list(str)
            データ軸がxyzのどの方向に対応しているか表すリスト(['x'], ['x', 'z'], etc)
        """
        to_axis = {3: "x", 2: "y", 1: "z", 0: "t"}
        return list(map(lambda a: to_axis[a], self.slice_axes))

    def masked(
        self, mask: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]
    ) -> "Data":
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

    def to_numpy(self) -> np.ndarray:
        """numpyのndarrayに変換する."""
        return np.array(self)

    def plot(self, **kwargs):
        """データをプロットする."""
        raise NotImplementedError()

    def build_frame_updater(
        self,
        axis: int = 0,
        title: Union[str, None] = None,
        notitle: bool = False,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        use_si: bool = True,
        vmin: float = None,
        vmax: float = None,
        **kwargs,
    ) -> FrameUpdater:
        """アニメーション描画処理を構築する.

        Parameters
        ----------
        axis : int, optional
            アニメーションする軸, by default 0
        title : str, optional
            タイトル(Noneの場合データ名(phisp等)), by default None
        notitle : bool, optional
            タイトルを付けない場合True, by default False
        offsets : (float or str, float or str, float or str)
            プロットのx,y,z軸のオフセット('left': 最初を0, 'center': 中心を0, 'right': 最後尾を0, float: 値だけずらす), by default None
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default False
        vmin : float, optional
            最小値, by default None
        vmax : float, optional
            最大値, by default None
        """
        if use_si:
            vmin = vmin or self.valunit.reverse(self.min())
            vmax = vmax or self.valunit.reverse(self.max())
        else:
            vmin = vmin or self.min()
            vmax = vmax or self.max()
        
        updater = FrameUpdater(
            self, axis, title, notitle, offsets, use_si, vmin=vmin, vmax=vmax, **kwargs
        )

        return updater

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
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        use_si: bool = True,
        vmin: float = None,
        vmax: float = None,
        to_html: bool = False,
        return_updater: bool = False,
        **kwargs,
    ):
        """gifアニメーションを作成する.

        Parameters
        ----------
        fig : Figure
            アニメーションを描画するFigure(Noneの場合新しく作成する), by default None
        axis : int, optional
            アニメーションする軸, by default 0

        action : {'return', 'show', 'to_html', 'save', 'frames'}, optional, by default 'to_html'
            Determines the behavior of the function:

            - 'return': The plot object is returned without rendering it.
            - 'show': The plot is displayed immediately.
            - 'to_html': The plot is converted to an Ipython.display.HTML object and returned.
            - 'save': The plot is saved to a file specified by 'filename' argument.
            - 'frames': FrameUpdater object is returned without rendering it.

        filename : str, optional
            保存するファイル名(actionが'save'以外の場合やNoneの場合保存されない), by default None

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
        vmin : float, optional
            最小値, by default None
        vmax : float, optional
            最大値, by default None
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default False

        show : bool, optional
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default

            .. deprecated :: 1.2.1

               This parameter is deprecated and will be removed in version 2.0.0.
               Use the 'action'='show' instead for equivalent functionality.

        savefilename : str, optional
            保存するファイル名(Noneの場合保存しない), by default None

            .. deprecated :: 1.2.1

               This parameter is deprecated and will be removed in version 2.0.0.
               Use the plot('action'='save', filename='example.gif') instead for equivalent functionality.

        to_html : bool
            アニメーションをHTMLとして返す. (使用例: Jupyter Notebook等でアニメーションを描画する際等)

            .. deprecated :: 1.2.1

               This parameter is deprecated and will be removed in version 2.0.0.
               Use the 'action'='to_html' instead for equivalent functionality.

        return_updater : bool
            FrameUpdaterを返す場合True, by default False

            .. deprecated :: 1.2.1

               This parameter is deprecated and will be removed in version 2.0.0.
               Use the 'action'='frames' instead for equivalent functionality.

        Returns
        -------
        Depending on the selected action:

        - If 'return': Returns the tuple of the plot object (fig, animation).
        - If 'show': Does not return anything, displays the plot.
        - If 'to_html': Returns an Ipython.display.HTML object of the plot (for Jupyter).
        - If 'save': Does not return anything, saves the plot to a file.
        - If 'frames': Returns FrameUpdater object.

        Examples
        --------
        >>> fig, ani = gifplot(action="return")
        Returns the tuple of the plot object.

        >>> gifplot(action="show")
        Displays the plot.

        >>> html = gifplot(action="to_html")
        Returns the HTML representation of the plot.

        >>> gifplot(action="save", filename = "example.gif")
        Saves the plot to a file.

        >>> updater = gifplot(action="frames")
        Returns FrameUpdater object.
        """
        if return_updater:
            warnings.warn(
                "The 'return_updater' flag is deprecated. "
                "Please use gifplot(action='frames') instead.",
                DeprecationWarning,
            )
            action = "frames"

        if mode is None:
            updater = self.build_frame_updater(
                axis, title, notitle, offsets, use_si, vmin, vmax, **kwargs
            )
        else:
            updater = self.build_frame_updater(
                axis, title, notitle, offsets, use_si, vmin, vmax, mode=mode, **kwargs
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


class Data4d(Data):
    """4次元データを管理する."""

    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)

        if "xslice" not in kwargs:
            kwargs["xslice"] = slice(0, obj.shape[3], 1)
        if "yslice" not in kwargs:
            kwargs["yslice"] = slice(0, obj.shape[2], 1)
        if "zslice" not in kwargs:
            kwargs["zslice"] = slice(0, obj.shape[1], 1)
        if "tslice" not in kwargs:
            kwargs["tslice"] = slice(0, obj.shape[0], 1)
        if "slice_axes" not in kwargs:
            kwargs["slice_axes"] = [0, 1, 2, 3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(self, mode: Literal["auto"] = "auto", **kwargs):
        """4次元データをプロットする.(未実装)

        Parameters
        ----------
        mode : str, optional
            [description], by default 'auto'
        """
        if mode == "auto":
            mode = "".join(sorted(self.use_axes))
        pass


class Data3d(Data):
    """3次元データを管理する."""

    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)

        if "xslice" not in kwargs:
            kwargs["xslice"] = slice(0, obj.shape[2], 1)
        if "yslice" not in kwargs:
            kwargs["yslice"] = slice(0, obj.shape[1], 1)
        if "zslice" not in kwargs:
            kwargs["zslice"] = slice(0, obj.shape[0], 1)
        if "tslice" not in kwargs:
            kwargs["tslice"] = slice(0, 1, 1)
        if "slice_axes" not in kwargs:
            kwargs["slice_axes"] = [1, 2, 3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(self, mode: Literal["auto"] = "auto", **kwargs):
        """3次元データをプロットする.(未実装)

        Parameters
        ----------
        mode : str, optional
            [description], by default 'auto'
        """
        if mode == "auto":
            mode = "".join(sorted(self.use_axes))
        pass


class Data2d(Data):
    """2次元データの2次元面を管理する."""

    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)

        if "xslice" not in kwargs:
            kwargs["xslice"] = slice(0, obj.shape[1], 1)
        if "yslice" not in kwargs:
            kwargs["yslice"] = slice(0, obj.shape[0], 1)
        if "zslice" not in kwargs:
            kwargs["zslice"] = slice(0, 1, 1)
        if "tslice" not in kwargs:
            kwargs["tslice"] = slice(0, 1, 1)
        if "slice_axes" not in kwargs:
            kwargs["slice_axes"] = [2, 3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(
        self,
        axes: Literal["auto", "xy", "yz", "zx", "yx", "zy", "xy"] = "auto",
        show: bool = False,
        use_si: bool = True,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        mode: Literal["cm", "cm+cont", "cont"] = "cm",
        **kwargs,
    ):
        """2次元データをプロットする.

        Parameters
        ----------
        axes : str, optional
            プロットする軸('xy', 'zx', etc), by default 'auto'
        show : bool
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default False
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default True
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
        import emout.plot.basic_plot as emplt

        if self.valunit is None:
            use_si = False

        if axes == "auto":
            axes = "".join(sorted(self.use_axes))

        if not re.match(r"x[yzt]|y[xzt]|z[xyt]|t[xyz]", axes):
            raise Exception(
                'Error: axes "{axes}" cannot be used with Data2d'.format(axes=axes)
            )
        if axes[0] not in self.use_axes or axes[1] not in self.use_axes:
            raise Exception(
                'Error: axes "{axes}" cannot be used because {axes}-axis does not exist in this data.'.format(
                    axes=axes
                )
            )
        if len(self.shape) != 2:
            raise Exception(
                'Error: axes "{axes}" cannot be used because data is not 2dim shape.'.format(
                    axes=axes
                )
            )

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

            _xlabel = "{} [{}]".format(axes[0], xunit.unit)
            _ylabel = "{} [{}]".format(axes[1], yunit.unit)
            _title = "{} [{}]".format(self.name, self.valunit.unit)
        else:
            _xlabel = axes[0]
            _ylabel = axes[1]
            _title = self.name

        def _offseted(line, offset):
            if offset == "left":
                line -= line.ravel()[0]
            elif offset == "center":
                line -= line.ravel()[line.size // 2]
            elif offset == "right":
                line -= line.ravel()[-1]
            else:
                line += offset
            return line

        kwargs["xlabel"] = kwargs.get("xlabel", None) or _xlabel
        kwargs["ylabel"] = kwargs.get("ylabel", None) or _ylabel
        kwargs["title"] = kwargs.get("title", None) or _title

        if mode == "surf":
            mesh = np.meshgrid(x, y)

            kwargs["zlabel"] = kwargs.get("zlabel", None) or _title
            val = z
            if "x" not in self.use_axes:
                y, z = mesh
                x = self.x_si[0] if use_si else self.x[0]
                x = np.zeros_like(mesh[0]) + x
            elif "y" not in self.use_axes:
                x, z = mesh
                y = self.y_si[0] if use_si else self.y[0]
                y = np.zeros_like(mesh[0]) + y
            elif "z" not in self.use_axes:
                x, y = mesh
                z = self.z_si[0] if use_si else self.z[0]
                z = np.zeros_like(mesh[0]) + z

            if offsets is not None:
                x = _offseted(x, offsets[0])
                y = _offseted(y, offsets[1])
                z = _offseted(z, offsets[2])
                val = _offseted(val, offsets[3])

            imgs = [emplt.plot_surface(x, y, z, val, **kwargs)]
        else:
            if offsets is not None:
                x = _offseted(x, offsets[0])
                y = _offseted(y, offsets[1])
                z = _offseted(z, offsets[2])
            mesh = np.meshgrid(x, y)

            imgs = []
            if "cm" in mode and "cont" in mode:
                savefilename = kwargs.get("savefilename", None)
                kwargs["savefilename"] = None
                img = emplt.plot_2dmap(z, mesh=mesh, **kwargs)
                kwargs["savefilename"] = savefilename
                img2 = emplt.plot_2d_contour(z, mesh=mesh, **kwargs)
                imgs = [img, img2]
            elif "cm" in mode:
                img = emplt.plot_2dmap(z, mesh=mesh, **kwargs)
                imgs.append(img)
            elif "cont" in mode:
                img = emplt.plot_2d_contour(z, mesh=mesh, **kwargs)
                imgs.append(img)

        if show:
            plt.show()
            return None
        else:
            return imgs[0] if len(imgs) == 1 else imgs


class Data1d(Data):
    """3次元データの1次元直線を管理する."""

    def __new__(cls, input_array, **kwargs):
        obj = np.asarray(input_array).view(cls)

        if "xslice" not in kwargs:
            kwargs["xslice"] = slice(0, obj.shape[1], 1)
        if "yslice" not in kwargs:
            kwargs["yslice"] = slice(0, 1, 1)
        if "zslice" not in kwargs:
            kwargs["zslice"] = slice(0, 1, 1)
        if "tslice" not in kwargs:
            kwargs["tslice"] = slice(0, 1, 1)
        if "slice_axes" not in kwargs:
            kwargs["slice_axes"] = [3]

        return super().__new__(cls, input_array, **kwargs)

    def plot(
        self,
        show: bool = False,
        use_si: bool = True,
        offsets: Union[Tuple[Union[float, str], Union[float, str]], None] = None,
        **kwargs,
    ):
        """1次元データをプロットする.

        Parameters
        ----------
        show : bool
            プロットを表示する場合True(ファイルに保存する場合は非表示), by default False
        use_si : bool
            SI単位系を用いる場合True(そうでない場合EMSES単位系を用いる), by default True
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
        import emout.plot.basic_plot as emplt

        if self.valunit is None:
            use_si = False

        if len(self.shape) != 1:
            raise Exception("Error: cannot plot because data is not 1dim shape.")

        axis = self.slice_axes[0]
        x = np.arange(*utils.slice2tuple(self.slices[axis]))
        y = self

        # "EMSES Unit" to "Physical Unit"
        if use_si:
            xunit = self.axisunits[axis]

            x = xunit.reverse(x)
            y = self.valunit.reverse(y)

            _xlabel = "{} [{}]".format(self.use_axes[0], xunit.unit)
            _ylabel = "{} [{}]".format(self.name, self.valunit.unit)
        else:
            _xlabel = self.use_axes[0]
            _ylabel = self.name

        def _offseted(line, offset):
            if offset == "left":
                line -= line[0]
            elif offset == "center":
                line -= line[len(line) // 2]
            elif offset == "right":
                line -= line[-1]
            else:
                line += offset
            return line

        if offsets is not None:
            x = _offseted(x, offsets[0])
            y = _offseted(y, offsets[1])

        kwargs["xlabel"] = kwargs.get("xlabel", None) or _xlabel
        kwargs["ylabel"] = kwargs.get("ylabel", None) or _ylabel

        line = emplt.plot_line(y, x=x, **kwargs)

        if show:
            plt.show()
            return None
        else:
            return line
