import collections
import warnings
from os import PathLike
from typing import Callable, List, Tuple, Union, Literal

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import emout.utils as utils


def flatten_list(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(
            el, (str, bytes)
        ):
            yield from flatten_list(el)
        else:
            yield el


ANIMATER_PLOT_MODE = Literal["return", "show", "to_html", "save"]


class Animator:
    def __init__(
        self,
        layout: List[List[List[Union["FrameUpdater", Callable[[int], None], None]]]],
    ):
        self._layout = layout

    def plot(
        self,
        fig: Union[plt.Figure, None] = None,
        action: ANIMATER_PLOT_MODE = "return",
        filename: PathLike = None,
        interval: int = 200,
        repeat: bool = True,
        show: bool = False,
        savefilename: PathLike = None,
        to_html: bool = False,
    ):
        """gifアニメーションを作成する

        Parameters
        ----------
        fig : Figure
            アニメーションを描画するFigure(Noneの場合新しく作成する), by default None

        action : {'return', 'show', 'to_html', 'save'}, optional, by default 'return'
            Determines the behavior of the function:

            - 'return': The plot object (fig, ani) is returned without rendering it.
            - 'show': The plot is displayed immediately.
            - 'to_html': The plot is converted to an Ipython.display.HTML object and returned.
            - 'save': The plot is saved to a file specified by 'filename' argument.

        filename : str, optional
            保存するファイル名(actionが'save'以外の場合やNoneの場合保存されない), by default None

        interval : int, optional
            フレーム間のインターバル(ミリ秒), by default 400
        repeat : bool
            アニメーションをループするならTrue, by default True

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

        Returns
        -------
        Depending on the selected action:

        - If 'return': Returns the tuple of the plot object (fig, animation).
        - If 'show': Does not return anything, displays the plot.
        - If 'to_html': Returns an Ipython.display.HTML object of the plot (for Jupyter).
        - If 'save': Does not return anything, saves the plot to a file.

        Examples
        --------
        >>> fig, ani = plot(action="return")
        Returns the tuple of the plot object.

        >>> plot(action="show")
        Displays the plot.

        >>> html = plot(action="to_html")
        Returns the HTML representation of the plot.

        >>> plot(action="save", filename = "example.gif")
        Saves the plot to a file.
        """
        if show:
            warnings.warn(
                "The 'show' flag is deprecated. "
                "Please use gifplot(action='show') instead.",
                DeprecationWarning,
            )
            action = "show"

        if to_html:
            warnings.warn(
                "The 'to_html' flag is deprecated. "
                "Please use gifplot(action='to_html') instead.",
                DeprecationWarning,
            )
            action = "to_html"

        if savefilename:
            warnings.warn(
                "The 'savefilename' argument is scheduled to change. "
                "Please use gifplot(action='save', filename='example.gif'), instead",
                DeprecationWarning,
            )
            action = "save"
            filename = savefilename

        if fig is None:
            fig = plt.gcf()

        def _update_all(i):
            plt.clf()
            j = 0
            shape = self.shape
            for line in self._layout:
                for plot in line:
                    j += 1

                    if plot[0] is None:
                        continue

                    plt.subplot(shape[0], shape[1], j)
                    for updater in plot:
                        if updater is None:
                            continue
                        updater(i)

        frames = self.frames

        ani = animation.FuncAnimation(
            fig,
            _update_all,
            interval=interval,
            frames=frames,
            repeat=repeat,
        )

        if action == "to_html":
            from IPython.display import HTML

            return HTML(ani.to_jshtml())
        elif action == "save" and (filename is not None):
            ani.save(filename, writer="quantized-pillow")
        elif action == "show":
            plt.show()
        else:
            return fig, ani

    @property
    def frames(self):
        """管理いているFrameUpdaterの最小フレーム数."""
        updaters = list(flatten_list(self._layout))
        if not updaters:
            raise ValueError("Updaters have no elements")

        # フレーム数の最小値を返す
        frames = min(
            len(updater) for updater in updaters if isinstance(updater, FrameUpdater)
        )
        return frames

    @property
    def shape(self):
        """レイアウトの形状."""
        nrows = len(self._layout)

        ncols = 1
        for l in self._layout:
            ncols = max(ncols, len(l))

        return (nrows, ncols)


class FrameUpdater:
    def __init__(
        self,
        data,
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
        if data.valunit is None:
            use_si = False

        if title is None:
            title = data.name

        if use_si:
            vmin = vmin or data.valunit.reverse(data.min())
            vmax = vmax or data.valunit.reverse(data.max())
        else:
            vmin = vmin or data.min()
            vmax = vmax or data.max()

        self.data = data
        self.axis = axis
        self.title = title
        self.notitle = notitle
        self.offsets = offsets
        self.use_si = use_si
        self.vmin = vmin
        self.vmax = vmax
        self.kwargs = kwargs

    def __call__(self, i: int):
        self.update(i)

    def update(self, i: int):
        data = self.data
        axis = self.axis
        title = self.title
        notitle = self.notitle
        offsets = self.offsets
        use_si = self.use_si
        vmin = self.vmin
        vmax = self.vmax
        kwargs = self.kwargs

        # 指定した軸でスライス
        slices = [slice(None)] * len(data.shape)
        slices[axis] = i
        val = data[tuple(slices)]

        # タイトルの設定
        if notitle:
            _title = title if len(title) > 0 else None
        else:
            ax = data.slice_axes[axis]
            slc = data.slices[ax]
            maxlen = data.shape[axis]

            line = np.array(utils.range_with_slice(slc, maxlen=maxlen), dtype=float)

            if offsets is not None:
                line = self._offseted(line, offsets[0])

            index = line[i]

            if use_si:  # SI単位系を用いる場合
                axisunit = data.axisunits[ax]
                _title = f"{title}({axisunit.reverse(index):.4e} {axisunit.unit}"

            else:  # EMSES単位系を用いる場合
                _title = f"{title}({index})"

        if offsets is not None:
            offsets2d = offsets[1:]
        else:
            offsets2d = None

        val.plot(
            vmin=vmin,
            vmax=vmax,
            title=_title,
            use_si=use_si,
            offsets=offsets2d,
            **kwargs,
        )

    def _offseted(self, line: List, offset: Union[str, float]):
        if offset == "left":
            line -= line[0]
        elif offset == "center":
            line -= line[len(line) // 2]
        elif offset == "right":
            line -= line[-1]
        else:
            line += offset
        return line

    def to_animator(self, layout=None):
        """アニメーターに変換する.

        Parameters
        ----------
        layout: List[List[List[FrameUpdater]]]
            アニメーションプロットのレイアウト
        """
        if layout is None:
            layout = [[[self]]]

        return Animator(layout=layout)

    def __len__(self):
        return self.data.shape[self.axis]
