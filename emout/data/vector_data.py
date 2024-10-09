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
    def __init__(self, objs: List[Any], name=None, attrs=None):
        x_data, y_data = objs

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

        super().__init__([x_data, y_data], attrs=attrs)
        self.x_data = x_data
        self.y_data = y_data

    def __setattr__(self, key, value):
        if key in ("x_data", "y_data"):
            super().__dict__[key] = value
            return
        super().__setattr__(key, value)

    @property
    def name(self) -> str:
        return self.attrs["name"]

    @property
    def valunit(self) -> UnitTranslator:
        return self.objs[0].valunit

    @property
    def axisunits(self) -> UnitTranslator:
        return self.objs[0].axisunits

    @property
    def slice_axes(self) -> np.ndarray:
        return self.objs[0].slice_axes

    @property
    def slices(self) -> np.ndarray:
        return self.objs[0].slices

    @property
    def shape(self) -> np.ndarray:
        return self.objs[0].shape

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
        vmin : float, optional
            最小値, by default None
        vmax : float, optional
            最大値, by default None
        """
        if vmin is None:
            vmin = min(self.objs[0].min(), self.objs[1].min())
        if vmax is None:
            vmax = max(self.objs[0].max(), self.objs[1].max())

        updater = FrameUpdater(
            self, axis, title, notitle, offsets, use_si, vmin, vmax, **kwargs
        )

        return updater

    def gifplot(
        self,
        fig: Union[plt.Figure, None] = None,
        axis: int = 0,
        action: ANIMATER_PLOT_MODE = "return",
        filename: PathLike = None,
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
        show: bool = False,
        savefilename: PathLike = None,
        to_html: bool = False,
        return_updater: bool = False,
        **kwargs,
    ):
        """gifアニメーションを作成する

        Parameters
        ----------
        fig : Figure
            アニメーションを描画するFigure(Noneの場合新しく作成する), by default None
        axis : int, optional
            アニメーションする軸, by default 0

        action : {'return', 'show', 'to_html', 'save', 'frames'}, optional
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

        print(f"vector: {action}")

        updater = self.build_frame_updater(
            axis, title, notitle, offsets, use_si, vmin, vmax, **kwargs
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
            **kwargs,
        )

    def plot(
        self,
        *args,
        **kwargs,
    ):
        if self.x_data.ndim == 2:
            self.plot2d(
                *args,
                **kwargs,
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
            raise Exception(
                'Error: axes "{axes}" cannot be used with Data2d'.format(axes=axes)
            )
        if axes[0] not in self.objs[0].use_axes or axes[1] not in self.objs[0].use_axes:
            raise Exception(
                'Error: axes "{axes}" cannot be used because {axes}-axis does not exist in this data.'.format(
                    axes=axes
                )
            )
        if len(self.objs[0].shape) != 2:
            raise Exception(
                'Error: axes "{axes}" cannot be used because data is not 2dim shape.'.format(
                    axes=axes
                )
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


VectorData2d = VectorData
