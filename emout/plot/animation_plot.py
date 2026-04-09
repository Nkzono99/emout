"""Animation framework for time-series EMSES data.

:class:`FrameUpdater` wraps a single data series and produces per-frame
plots, while :class:`Animator` orchestrates one or more updaters into a
``matplotlib.animation.FuncAnimation``.
"""

import collections
import warnings
from os import PathLike
from typing import Callable, List, Tuple, Union, Literal

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import emout.utils as utils


def flatten_list(l):
    """入れ子のイテラブルを 1 次元に平坦化して順に返す。

    Parameters
    ----------
    l : object
        任意に入れ子になったイテラブルです。
        文字列と bytes は 1 要素として扱います。

    Returns
    -------
    Iterator
        平坦化された要素を順に返すイテレータです。
    """
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(
            el, (str, bytes)
        ):
            yield from flatten_list(el)
        else:
            yield el


ANIMATER_PLOT_MODE = Literal["return", "show", "to_html", "save"]


class Animator:
    """複数の FrameUpdater を束ねてアニメーション描画を行うクラス。"""
    def __init__(
        self,
        layout: List[List[List[Union["FrameUpdater", Callable[[int], None], None]]]],
    ):
        """インスタンスを初期化する。

        Parameters
        ----------
        layout : List[List[List[Union["FrameUpdater", Callable[[int], None], None]]]]
            3 重リストで定義したレイアウトです。
            `layout[row][col]` に複数 updater を配置できます。
        """
        self._layout = layout

    def plot(
        self,
        fig: Union[plt.Figure, None] = None,
        action: ANIMATER_PLOT_MODE = "to_html",
        filename: PathLike = None,
        interval: int = 200,
        repeat: bool = True,
        show: bool = False,
        savefilename: PathLike = None,
        to_html: bool = False,
    ):
        """GIF アニメーションを作成する。

        Parameters
        ----------
        fig : Union[plt.Figure, None], optional
            描画に使用する Figure。`None` の場合は現在の Figure を使います。
        action : ANIMATER_PLOT_MODE, optional
            アニメーション生成後の処理モードです。
            `'return'` は `(fig, animation)` を返し、
            `'show'` は表示、
            `'to_html'` は HTML を返し、
            `'save'` は `filename` に保存します。
        filename : PathLike, optional
            `action='save'` のときの保存先ファイル名です。
        interval : int, optional
            フレーム間隔 [ms]。
        repeat : bool, optional
            ループ再生するかどうか。
        show : bool, optional
            非推奨。`True` の場合は `action='show'` と同等に扱います。
        savefilename : PathLike, optional
            非推奨。指定時は `action='save'` として扱います。
        to_html : bool, optional
            非推奨。`True` の場合は `action='to_html'` と同等に扱います。

        Returns
        -------
        object
            `action` に応じた描画結果を返します。
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
            """全 updater を 1 フレーム分更新する。
            
            Parameters
            ----------
            i : object
                反復 index です。
            Returns
            -------
            None
                戻り値はありません。
            """
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
    """FrameUpdater クラス。
    """
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
        **kwargs,
    ):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        data : object
            フレームごとにスライスして描画するデータ。
        axis : int, optional
            アニメーションさせる軸 index。
        title : Union[str, None], optional
            タイトルのプレフィックス。
        notitle : bool, optional
            `True` の場合、フレーム位置をタイトルに追記しません。
        offsets : Union[, optional
                    Tuple[Union[float, str], Union[float, str], Union[float, str]], None
                ], optional
            座標オフセット。`'left'` / `'center'` / `'right'` も指定できます。
        use_si : bool, optional
            `True` の場合は SI 単位系で表示します。
        **kwargs : dict
            `val.plot(...)` に渡す追加引数です。
        """
        if data.valunit is None:
            use_si = False

        if title is None:
            title = data.name

        self.data = data
        self.axis = axis
        self.title = title
        self.notitle = notitle
        self.offsets = offsets
        self.use_si = use_si
        self.kwargs = kwargs

    def __call__(self, i: int):
        """呼び出し可能オブジェクトとして実行する。
        
        Parameters
        ----------
        i : int
            フレーム番号。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        self.update(i)

    def update(self, i: int):
        """指定フレームのスライスを描画する。

        Parameters
        ----------
        i : int
            フレーム番号。
        
        Returns
        -------
        None
            戻り値はありません。
        """
        data = self.data
        axis = self.axis
        title = self.title
        notitle = self.notitle
        offsets = self.offsets
        use_si = self.use_si
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
            title=_title,
            use_si=use_si,
            offsets=offsets2d,
            **kwargs,
        )

    def _offseted(self, line: List, offset: Union[str, float]):
        """配列にオフセットを適用した結果を返す。

        Parameters
        ----------
        line : List
            座標配列。
        offset : Union[str, float]
            オフセット指定。`'left'` / `'center'` / `'right'` または数値。

        Returns
        -------
        object
            オフセット適用後の配列。
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
        """要素数を返す。
        
        Returns
        -------
        int
            要素数。
        """
        return self.data.shape[self.axis]
