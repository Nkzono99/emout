import re
import warnings
from os import PathLike
from typing import Any, List, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import emout.plot.basic_plot as emplt
import emout.utils as utils
from emout.plot.animation_plot import ANIMATER_PLOT_MODE, FrameUpdater
from emout.utils import UnitTranslator


class VectorData(utils.Group):
    """VectorData クラス。
    """
    def __init__(self, objs: List[Any], name=None, attrs=None):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        objs : List[Any]
            ベクトル成分データのリストです（2成分または3成分）。
        name : object, optional
            対象データ名またはキー名です。
        attrs : object, optional
            生成される `VectorData` に引き継ぐ属性辞書です。
        """
        if len(objs) not in (2, 3):
            raise ValueError("VectorData requires 2 or 3 components.")
        x_data = objs[0]
        y_data = objs[1]
        z_data = objs[2] if len(objs) == 3 else None

        if attrs is None:
            attrs = dict()

        if name:
            attrs["name"] = name
        elif "name" in attrs:
            pass
        elif hasattr(x_data, "name"):
            attrs["name"] = name
        else:
            attrs["name"] = ""

        super().__init__(list(objs), attrs=attrs)
        self.x_data = x_data
        self.y_data = y_data
        if z_data is not None:
            self.z_data = z_data

    def __setattr__(self, key, value):
        """属性を設定する。
        
        Parameters
        ----------
        key : object
            取得・設定対象のキーです。
        value : object
            値。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        if key in ("x_data", "y_data", "z_data"):
            super().__dict__[key] = value
            return
        super().__setattr__(key, value)

    @property
    def name(self) -> str:
        """データ名を返す。
        
        Returns
        -------
        str
            文字列表現です。
        """
        return self.attrs["name"]

    @property
    def valunit(self) -> UnitTranslator:
        """値の単位変換器を返す。
        
        Returns
        -------
        UnitTranslator
            処理結果です。
        """
        return self.objs[0].valunit

    @property
    def axisunits(self) -> UnitTranslator:
        """軸ごとの単位変換器を返す。
        
        Returns
        -------
        UnitTranslator
            処理結果です。
        """
        return self.objs[0].axisunits

    @property
    def slice_axes(self) -> np.ndarray:
        """各配列軸に対応する元データ軸を返す。
        
        Returns
        -------
        np.ndarray
            処理結果です。
        """
        return self.objs[0].slice_axes

    @property
    def slices(self) -> np.ndarray:
        """各軸のスライス範囲を返す。
        
        Returns
        -------
        np.ndarray
            処理結果です。
        """
        return self.objs[0].slices

    @property
    def shape(self) -> np.ndarray:
        """ベクトル成分を除いたデータ形状を返す。
        
        Returns
        -------
        np.ndarray
            処理結果です。
        """
        return self.objs[0].shape

    @property
    def ndim(self) -> int:
        """ベクトル成分を除いたデータ次元数を返す。"""
        return self.objs[0].ndim

    def build_frame_updater(
        self,
        axis: int = 0,
        title: Union[str, None] = None,
        notitle: bool = False,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        use_si: bool = True,
        **kwargs,
    ):
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
        """
        updater = FrameUpdater(
            self, axis, title, notitle, offsets, use_si, **kwargs
        )

        return updater

    def gifplot(
        self,
        fig: Union[plt.Figure, None] = None,
        axis: int = 0,
        action: ANIMATER_PLOT_MODE = "to_html",
        filename: PathLike = None,
        interval: int = 200,
        repeat: bool = True,
        title: Union[str, None] = None,
        notitle: bool = False,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        use_si: bool = True,
        show: bool = False,
        savefilename: PathLike = None,
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
        action : ANIMATER_PLOT_MODE, optional
            出力アクション種別です。
        filename : PathLike, optional
            保存先または読み込み対象のファイル名です。
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
        show : bool, optional
            True の場合は描画を表示。
        savefilename : PathLike, optional
            保存先ファイル名です。
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

        updater = self.build_frame_updater(
            axis, title, notitle, offsets, use_si, **kwargs
        )

        if action == "frames":
            return updater

        animator = updater.to_animator([[[updater]]])

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

    def plot(
        self,
        *args,
        **kwargs,
    ):
        """ベクトルデータをプロットする。

        2 次元データの場合は :meth:`plot2d`、
        3 次元データの場合は :meth:`plot3d` を呼び出します。

        Parameters
        ----------
        *args : tuple
            :meth:`plot2d` へ渡す位置引数です。
            利用可能な主な引数は `mode`, `axes`, `show`, `use_si`, `offsets` です。
        **kwargs : dict
            :meth:`plot2d` へ渡すキーワード引数です。
            さらに内部で `plot_2d_vector` または `plot_2d_streamline` に委譲されるため、
            `mesh`, `savefilename`, `scale`, `scaler`, `skip`, `easy_to_read`,
            `color`, `cmap`, `norm`, `vmin`, `vmax`, `density`, `figsize`,
            `xlabel`, `ylabel`, `title`, `dpi` などを指定できます。

        Returns
        -------
        object
            委譲先の描画メソッドが返すオブジェクトです。
        """
        if self.x_data.ndim == 2:
            return self.plot2d(
                *args,
                **kwargs,
            )
        if self.x_data.ndim == 3:
            return self.plot3d(
                *args,
                **kwargs,
            )
        raise NotImplementedError(
            f"VectorData.plot is not implemented for ndim={self.x_data.ndim}."
        )

    def plot2d(
        self,
        mode: Literal["stream", "vec"] = "stream",
        axes: Literal["auto", "xy", "yz", "zx", "yx", "zy", "xy"] = "auto",
        show: bool = False,
        use_si: bool = True,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        **kwargs,
    ):
        """2次元データをプロットする.

        Parameters
        ----------
        mode : str
            プロットの種類('vec': quiver plot, 'stream': streamline plot), by default 'stream'
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
        if self.objs[0].valunit is None:
            use_si = False

        if axes == "auto":
            axes = "".join(sorted(self.objs[0].use_axes))

        if not re.match(r"x[yzt]|y[xzt]|z[xyt]|t[xyz]", axes):
            raise ValueError(
                f'axes "{axes}" cannot be used with 2D vector data'
            )
        if axes[0] not in self.objs[0].use_axes or axes[1] not in self.objs[0].use_axes:
            raise ValueError(
                f'axes "{axes}" cannot be used because the axis does not exist in this data'
            )
        if len(self.objs[0].shape) != 2:
            raise ValueError(
                f'axes "{axes}" cannot be used because data is not 2-dimensional'
            )

        # x: 3, y: 2, z:1 t:0
        axis1 = self.objs[0].slice_axes[self.objs[0].use_axes.index(axes[0])]
        axis2 = self.objs[0].slice_axes[self.objs[0].use_axes.index(axes[1])]

        x = np.arange(*utils.slice2tuple(self.objs[0].slices[axis1]), dtype=float)
        y = np.arange(*utils.slice2tuple(self.objs[0].slices[axis2]), dtype=float)

        if use_si:
            xunit = self.objs[0].axisunits[axis1]
            yunit = self.objs[0].axisunits[axis2]
            valunit = self.objs[0].valunit

            x = xunit.reverse(x)
            y = yunit.reverse(y)

            _xlabel = "{} [{}]".format(axes[0], xunit.unit)
            _ylabel = "{} [{}]".format(axes[1], yunit.unit)
            _title = "{} [{}]".format(self.name, valunit.unit)

            x_data = self.x_data.val_si
            y_data = self.y_data.val_si
        else:
            _xlabel = axes[0]
            _ylabel = axes[1]
            _title = self.name

            x_data = self.x_data
            y_data = self.y_data

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
            line = line.astype(float)
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
        kwargs["title"] = kwargs.get("title", None) or _title

        mesh = np.meshgrid(x, y)
        if "vec" in mode:
            img = emplt.plot_2d_vector(x_data, y_data, mesh=mesh, **kwargs)
        elif "stream" in mode:
            img = emplt.plot_2d_streamline(x_data, y_data, mesh=mesh, **kwargs)

        if show:
            plt.show()
            return None
        else:
            return img

    def plot_pyvista(
        self,
        mode: Literal["stream", "streamline", "vec", "quiver"] = "stream",
        show: bool = False,
        use_si: bool = True,
        offsets: Union[
            Tuple[Union[float, str], Union[float, str], Union[float, str]], None
        ] = None,
        plotter=None,
        **kwargs,
    ):
        """pyvista で 3 次元ベクトル場を描画する。"""
        if self.x_data.ndim != 3:
            raise ValueError(
                "plot_pyvista on VectorData requires 3D component data."
            )
        if len(self.objs) < 3 or not hasattr(self, "z_data"):
            raise ValueError(
                "plot_pyvista on VectorData requires 3 components (x, y, z)."
            )

        if self.objs[0].valunit is None:
            use_si = False

        if mode in ("vec", "quiver"):
            from emout.plot.pyvista_plot import plot_vector_quiver3d

            return plot_vector_quiver3d(
                self.x_data,
                self.y_data,
                self.z_data,
                plotter=plotter,
                use_si=use_si,
                offsets=offsets,
                show=show,
                **kwargs,
            )

        if mode in ("stream", "streamline"):
            from emout.plot.pyvista_plot import plot_vector_streamlines3d

            return plot_vector_streamlines3d(
                self.x_data,
                self.y_data,
                self.z_data,
                plotter=plotter,
                use_si=use_si,
                offsets=offsets,
                show=show,
                **kwargs,
            )

        raise ValueError(f'Unsupported mode "{mode}" for VectorData.plot_pyvista.')

    def plot3d(
        self,
        mode: Literal["stream", "streamline", "vec", "quiver"] = "stream",
        **kwargs,
    ):
        """`plot_pyvista` のエイリアス。"""
        return self.plot_pyvista(mode=mode, **kwargs)


VectorData2d = VectorData
VectorData3d = VectorData
