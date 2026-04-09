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
        """インスタンスを生成する。
        
        Parameters
        ----------
        input_array : object
            元となる NumPy 配列です。
        filename : object, optional
            保存先または読み込み対象のファイル名です。
        name : object, optional
            対象データ名またはキー名です。
        xslice : object, optional
            x 軸スライスです。
        yslice : object, optional
            y 軸スライスです。
        zslice : object, optional
            z 軸スライスです。
        tslice : object, optional
            時間方向スライスです。
        slice_axes : object, optional
            配列軸が元データのどの軸に対応するかを示す index リストです
            （`0:t, 1:z, 2:y, 3:x`）。
        axisunits : object, optional
            各軸（`t,z,y,x`）に対応する単位変換器のリストです。
        valunit : object, optional
            値の単位変換器です。
        Returns
        -------
        object
            処理結果です。
        """
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
        """要素を取得する。
        
        Parameters
        ----------
        item : object
            代入または更新する値です。
        Returns
        -------
        object
            処理結果です。
        """
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
        """NumPy 配列のメタ情報を引き継ぐ。
        
        Parameters
        ----------
        obj : object
            対象オブジェクトです。
        Returns
        -------
        None
            戻り値はありません。
        """
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
        """対象軸の情報を返す。
        
        Parameters
        ----------
        ax : int
            描画先の Axes。
        
        Returns
        -------
        np.ndarray
            処理結果です。
        """
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
        return [to_axis[a] for a in self.slice_axes]

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
        """アニメーション描画を実行する。
        
        Parameters
        ----------
        fig : Union[plt.Figure, None], optional
            描画先の Figure。
        axis : int, optional
            対象軸。
        mode : str, optional
            処理モード。
        action : ANIMATER_PLOT_MODE, optional
            出力アクション種別です。
        filename : PathLike, optional
            保存先または読み込み対象のファイル名です。
        show : bool, optional
            True の場合は描画を表示。
        savefilename : PathLike, optional
            保存先ファイル名です。
        interval : int, optional
            フレーム間隔 [ms] です。
        repeat : bool, optional
            `True` の場合、アニメーションをループ再生します。
        title : Union[str, None], optional
            タイトル文字列。
        notitle : bool, optional
            `True` の場合、フレーム番号由来の自動タイトル追記を行いません。
        offsets : Union[, optional
                    Tuple[Union[float, str], Union[float, str], Union[float, str]], None
                ], optional
            軸方向のオフセット。
        use_si : bool, optional
            True の場合は SI 単位系を使用。
        vmin : float, optional
            表示範囲の最小値。
        vmax : float, optional
            表示範囲の最大値。
        to_html : bool, optional
            非推奨オプションです。`True` の場合 `action='to_html'` と同等です。
        return_updater : bool, optional
            非推奨オプションです。`True` の場合 `action='frames'` と同等です。
        **kwargs : dict
            追加のキーワード引数。内部で呼び出す関数へ渡されます。
        
        Returns
        -------
        object
            処理結果です。
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
        """インスタンスを生成する。
        
        Parameters
        ----------
        input_array : object
            元となる NumPy 配列です。
        **kwargs : dict
            追加のキーワード引数。内部で呼び出す関数へ渡されます。
        
        Returns
        -------
        object
            処理結果です。
        """
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
        """4 次元データをプロットする（未実装）。

        Parameters
        ----------
        mode : {'auto'}, optional
            プロットモードです。現在は未実装のため `'auto'` のみ受け付けます。
        **kwargs : dict
            将来拡張のためのキーワード引数です。現在は使用しません。

        Returns
        -------
        None
            戻り値はありません。
        """
        if mode == "auto":
            mode = "".join(sorted(self.use_axes))
        pass


class Data3d(Data):
    """3次元データを管理する."""

    def __new__(cls, input_array, **kwargs):
        """インスタンスを生成する。
        
        Parameters
        ----------
        input_array : object
            元となる NumPy 配列です。
        **kwargs : dict
            追加のキーワード引数。内部で呼び出す関数へ渡されます。
        
        Returns
        -------
        object
            処理結果です。
        """
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

    def plot(
        self,
        mode: Literal["auto"] = "auto",
        use_si: bool = True,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        *args,
        **kwargs,
    ):
        """3 次元データをプロットする。

        現在は `mode='cont'` のみ実装されており、内部で
        :func:`emout.plot.contour3d.contour3d` を呼び出します。

        Parameters
        ----------
        mode : {'auto', 'cont'}, optional
            プロットモードです。`'auto'` の場合は `'cont'` が選択されます。
        use_si : bool, optional
            `True` の場合は SI 単位系へ変換して描画します。
        offsets : Union[
                    Tuple[Union[float, str], Union[float, str], Union[float, str]], None
                ], optional
            描画原点のオフセット `(x, y, z)` です。
            文字列 `'left'`, `'center'`, `'right'` も指定できます。
        *args : tuple
            `contour3d` へ渡す追加の位置引数です。
            先頭要素として等値面レベル `levels`（`Sequence[float]`）を指定します。
        **kwargs : dict
            `contour3d` へ渡すキーワード引数です。
            主な引数は `ax`, `bounds_xyz`, `roi_zyx`, `opacity`, `step`,
            `title`, `save`, `show`, `xlabel`, `ylabel`, `zlabel`,
            `clabel`, `clabel_fmt`, `clabel_fontsize`, `clabel_sigfigs`,
            `clabel_shared_exponent`, `clabel_text_kwargs`,
            `clabel_exponent_pos`, `clabel_exponent_text`,
            `clabel_exponent_kwargs` です。

        Returns
        -------
        tuple(matplotlib.figure.Figure, matplotlib.axes.Axes) or None
            `mode='cont'` の場合は `(fig, ax)` を返します。
            未対応モードでは `None` を返します。
        """
        if mode == "auto":
            mode = "cont"

        def _offseted(line, offset):
            """位置指定を実座標オフセットへ変換する。
            
            Parameters
            ----------
            line : object
                オフセット適用対象の座標列です。
            offset : object
                適用するオフセット値またはキーワードです。
            Returns
            -------
            object
                処理結果です。
            """
            if offset == "left":
                line -= line.ravel()[0]
            elif offset == "center":
                line -= line.ravel()[line.size // 2]
            elif offset == "right":
                line -= line.ravel()[-1]
            else:
                line += offset
            return line

        if mode == "cont":
            from emout.plot.contour3d import contour3d

            if use_si:
                data3d = self.val_si
                dx = self.axisunits[-1].reverse(1.0)
            else:
                data3d = self
                dx = 1.0

            if offsets is not None:
                origin_xyz = (
                    _offseted(0.0, offsets[0]),
                    _offseted(0.0, offsets[1]),
                    _offseted(0.0, offsets[2]),
                )
            else:
                origin_xyz = (0.0, 0.0, 0.0)

            fig, ax = contour3d(data3d, dx, origin_xyz=origin_xyz, *args, **kwargs)

            return fig, ax

    def plot_pyvista(
        self,
        mode: Literal["box", "volume", "slice", "contour"] = "box",
        use_si: bool = True,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        show: bool = False,
        plotter=None,
        cmap: str = "viridis",
        clim: Union[Tuple[float, float], None] = None,
        opacity: Union[float, str] = "sigmoid",
        contour_levels: Union[int, np.ndarray] = 8,
        add_outline: bool = True,
        outline_color: str = "white",
        add_scalar_bar: bool = True,
        **kwargs,
    ):
        """pyvista で 3 次元データを描画する。"""
        from emout.plot.pyvista_plot import plot_scalar_volume

        if self.valunit is None:
            use_si = False

        return plot_scalar_volume(
            self,
            mode=mode,
            plotter=plotter,
            use_si=use_si,
            offsets=offsets,
            show=show,
            cmap=cmap,
            clim=clim,
            opacity=opacity,
            contour_levels=contour_levels,
            add_outline=add_outline,
            outline_color=outline_color,
            add_scalar_bar=add_scalar_bar,
            **kwargs,
        )

    def plot3d(self, *args, **kwargs):
        """`plot_pyvista` のエイリアス。"""
        return self.plot_pyvista(*args, **kwargs)

    def plot_surfaces(
        self,
        surfaces,
        *,
        ax=None,
        use_si: bool = True,
        vmin: Union[float, None] = None,
        vmax: Union[float, None] = None,
        **kwargs,
    ):
        """3D スカラー場に明示メッシュ境界を重ねて描画する。

        :func:`emout.plot.surface_cut.plot_surfaces` に ``self`` を
        :class:`emout.plot.surface_cut.Field3D` としてラップして渡します。
        ``data.phisp[-1].plot_surfaces(data.boundaries.mesh().render(), vmin=0, vmax=10)``
        のような 1 行呼び出しを意図しています。

        Parameters
        ----------
        surfaces
            :class:`emout.plot.surface_cut.RenderItem` か
            :class:`emout.plot.surface_cut.MeshSurface3D` 、または
            :class:`emout.emout.boundaries.Boundary` /
            :class:`emout.emout.boundaries.BoundaryCollection`、
            あるいはそれらのシーケンス。
            ``MeshSurface3D`` 単体が渡された場合は ``render()`` 相当の
            デフォルトスタイルで包みます。``Boundary`` /
            ``BoundaryCollection`` が渡された場合は ``mesh(use_si=use_si)``
            を呼んでから ``render()`` でラップするので、
            ``data.phisp[-1].plot_surfaces(data.boundaries)`` のように
            一行で渡せます。
        ax : matplotlib.axes.Axes, optional
            描画先 Axes です。未指定の場合は新たに 3D 軸を作成します。
        use_si : bool, optional
            `True` (デフォルト) の場合、データと格子間隔を SI 単位に
            変換してから描画します。``Boundary`` を渡した場合の境界
            メッシュ生成にもこの値が伝播します。
            単位変換キーが無い場合は自動で `False` 扱いとなります。
        vmin, vmax : float, optional
            カラーマップの範囲です。
        **kwargs : dict
            :func:`emout.plot.surface_cut.plot_surfaces` に転送される
            追加のキーワード引数です (`bounds`, `cmap_name`,
            `contour_levels` など)。

        Returns
        -------
        tuple
            ``plot_surfaces`` が返す ``(cmap, norm)`` のタプル。
        """
        import matplotlib.pyplot as plt

        from emout.emout.boundaries import Boundary, BoundaryCollection
        from emout.plot.surface_cut import (
            Field3D,
            MeshSurface3D,
            RenderItem,
            UniformCellCenteredGrid,
            plot_surfaces as _plot_surfaces,
        )

        effective_si = bool(use_si) and getattr(self, "valunit", None) is not None

        if effective_si:
            data = np.asarray(self.val_si, dtype=np.float64)
            dx = float(self.axisunits[-1].reverse(1.0))
            dy = float(self.axisunits[-2].reverse(1.0))
            dz = float(self.axisunits[-3].reverse(1.0))
        else:
            data = np.asarray(self, dtype=np.float64)
            dx = dy = dz = 1.0

        nz, ny, nx = data.shape
        grid = UniformCellCenteredGrid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
        field = Field3D(grid, data)

        def _wrap(item):
            if isinstance(item, RenderItem):
                return item
            if isinstance(item, MeshSurface3D):
                return item.render()
            if isinstance(item, (Boundary, BoundaryCollection)):
                return item.render(use_si=effective_si)
            return item

        # A bare BoundaryCollection is iterable, but we want to treat it as a
        # single composite — match the (RenderItem, MeshSurface3D) branch.
        single_types = (RenderItem, MeshSurface3D, Boundary, BoundaryCollection)
        if isinstance(surfaces, single_types):
            items = _wrap(surfaces)
        else:
            items = [_wrap(s) for s in surfaces]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        return _plot_surfaces(
            ax,
            field=field,
            surfaces=items,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )


class Data2d(Data):
    """2次元データの2次元面を管理する."""

    def __new__(cls, input_array, **kwargs):
        """インスタンスを生成する。
        
        Parameters
        ----------
        input_array : object
            元となる NumPy 配列です。
        **kwargs : dict
            追加のキーワード引数。内部で呼び出す関数へ渡されます。
        
        Returns
        -------
        object
            処理結果です。
        """
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
        **kwargs : dict
            低レベル描画関数へ渡す追加引数です。
            `mode='cm'` / `mode='cm+cont'` では `plot_2dmap`、
            `mode='cont'` では `plot_2d_contour`、
            `mode='surf'` では `plot_surface` の引数を指定できます。
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
            raise ValueError(
                f'axes "{axes}" cannot be used with Data2d'
            )
        if axes[0] not in self.use_axes or axes[1] not in self.use_axes:
            raise ValueError(
                f'axes "{axes}" cannot be used because the axis does not exist in this data'
            )
        if len(self.shape) != 2:
            raise ValueError(
                f'axes "{axes}" cannot be used because data is not 2-dimensional (shape={self.shape})'
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
            """位置指定を実座標オフセットへ変換する。
            
            Parameters
            ----------
            line : object
                オフセット適用対象の座標列です。
            offset : object
                適用するオフセット値またはキーワードです。
            Returns
            -------
            object
                処理結果です。
            """
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

    def cmap(self, **kwargs):
        """2次元データをカラーマップとして描画する。

        :py:meth:`plot` の ``mode='cm'`` と等価なショートカット。
        ``data.cmap(...)`` の形で直接呼び出せるように用意されている。
        引数は :py:meth:`plot` とまったく同じで、 ``mode`` だけは
        受け付けない（指定すると :class:`TypeError`）。

        Returns
        -------
        matplotlib.image.AxesImage or list or None
            :py:meth:`plot` と同じ返値。
        """
        if "mode" in kwargs:
            raise TypeError(
                "Data2d.cmap() does not accept 'mode'; call plot(mode=...) directly instead"
            )
        return self.plot(mode="cm", **kwargs)

    def contour(self, **kwargs):
        """2次元データを等高線として描画する。

        :py:meth:`plot` の ``mode='cont'`` と等価なショートカット。
        ``data.contour(...)`` の形で直接呼び出せるように用意されている。
        引数は :py:meth:`plot` とまったく同じで、 ``mode`` だけは
        受け付けない（指定すると :class:`TypeError`）。

        Returns
        -------
        matplotlib.contour.QuadContourSet or list or None
            :py:meth:`plot` と同じ返値。
        """
        if "mode" in kwargs:
            raise TypeError(
                "Data2d.contour() does not accept 'mode'; call plot(mode=...) directly instead"
            )
        return self.plot(mode="cont", **kwargs)

    def plot_pyvista(
        self,
        use_si: bool = True,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        show: bool = False,
        plotter=None,
        cmap: str = "viridis",
        clim: Union[Tuple[float, float], None] = None,
        show_edges: bool = False,
        add_scalar_bar: bool = True,
        **kwargs,
    ):
        """pyvista で 2 次元データを 3D 空間上の平面として描画する。"""
        from emout.plot.pyvista_plot import plot_scalar_plane

        if self.valunit is None:
            use_si = False

        return plot_scalar_plane(
            self,
            plotter=plotter,
            use_si=use_si,
            offsets=offsets,
            show=show,
            cmap=cmap,
            clim=clim,
            show_edges=show_edges,
            add_scalar_bar=add_scalar_bar,
            **kwargs,
        )

    def plot3d(self, *args, **kwargs):
        """`plot_pyvista` のエイリアス。"""
        return self.plot_pyvista(*args, **kwargs)


class Data1d(Data):
    """3次元データの1次元直線を管理する."""

    def __new__(cls, input_array, **kwargs):
        """インスタンスを生成する。
        
        Parameters
        ----------
        input_array : object
            元となる NumPy 配列です。
        **kwargs : dict
            追加のキーワード引数。内部で呼び出す関数へ渡されます。
        
        Returns
        -------
        object
            処理結果です。
        """
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
            raise ValueError("cannot plot because data is not 1-dimensional")

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
            """位置指定を実座標オフセットへ変換する。
            
            Parameters
            ----------
            line : object
                オフセット適用対象の座標列です。
            offset : object
                適用するオフセット値またはキーワードです。
            Returns
            -------
            object
                処理結果です。
            """
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
