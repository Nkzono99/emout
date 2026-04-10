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
    """Flatten a nested iterable into a 1-D sequence.

    Parameters
    ----------
    l : object
        Arbitrarily nested iterable.
        Strings and bytes are treated as single elements.

    Returns
    -------
    Iterator
        Iterator that yields elements in flattened order.
    """
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten_list(el)
        else:
            yield el


ANIMATER_PLOT_MODE = Literal["return", "show", "to_html", "save"]


class Animator:
    """Orchestrate multiple FrameUpdaters into an animated plot."""

    def __init__(
        self,
        layout: List[List[List[Union["FrameUpdater", Callable[[int], None], None]]]],
    ):
        """Initialize the animator.

        Parameters
        ----------
        layout : List[List[List[Union["FrameUpdater", Callable[[int], None], None]]]]
            Layout defined as a triply-nested list.
            `layout[row][col]` can hold multiple updaters.
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
        """Create a GIF animation.

        Parameters
        ----------
        fig : Union[plt.Figure, None], optional
            Figure to use for rendering. If `None`, the current figure is used.
        action : ANIMATER_PLOT_MODE, optional
            Post-generation action mode.
            `'return'` returns `(fig, animation)`,
            `'show'` displays the animation,
            `'to_html'` returns HTML,
            `'save'` saves to `filename`.
        filename : PathLike, optional
            Output file path when `action='save'`.
        interval : int, optional
            Frame interval in milliseconds.
        repeat : bool, optional
            Whether to loop the animation.
        show : bool, optional
            Deprecated. If `True`, treated as `action='show'`.
        savefilename : PathLike, optional
            Deprecated. When given, treated as `action='save'`.
        to_html : bool, optional
            Deprecated. If `True`, treated as `action='to_html'`.

        Returns
        -------
        object
            Rendering result depending on `action`.
        """
        if show:
            warnings.warn(
                "The 'show' flag is deprecated. Please use gifplot(action='show') instead.",
                DeprecationWarning,
            )
            action = "show"

        if to_html:
            warnings.warn(
                "The 'to_html' flag is deprecated. Please use gifplot(action='to_html') instead.",
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
            """Update all updaters for one frame.

            Parameters
            ----------
            i : object
                Iteration index.
            Returns
            -------
            None
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
        """Minimum frame count among managed FrameUpdaters."""
        updaters = list(flatten_list(self._layout))
        if not updaters:
            raise ValueError("Updaters have no elements")

        # Return the minimum frame count
        frames = min(len(updater) for updater in updaters if isinstance(updater, FrameUpdater))
        return frames

    @property
    def shape(self):
        """Layout shape as (nrows, ncols)."""
        nrows = len(self._layout)

        ncols = 1
        for l in self._layout:
            ncols = max(ncols, len(l))

        return (nrows, ncols)


class FrameUpdater:
    """Single data series frame updater for animation."""

    def __init__(
        self,
        data,
        axis: int = 0,
        title: Union[str, None] = None,
        notitle: bool = False,
        offsets: Union[Tuple[Union[float, str], Union[float, str], Union[float, str]], None] = None,
        use_si: bool = True,
        **kwargs,
    ):
        """Initialize the frame updater.

        Parameters
        ----------
        data : object
            Data to slice per frame for rendering.
        axis : int, optional
            Axis index to animate along.
        title : Union[str, None], optional
            Title prefix.
        notitle : bool, optional
            If `True`, do not append the frame position to the title.
        offsets : Union[, optional
                    Tuple[Union[float, str], Union[float, str], Union[float, str]], None
                ], optional
            Coordinate offsets. `'left'` / `'center'` / `'right'` are also accepted.
        use_si : bool, optional
            If `True`, display in SI units.
        **kwargs : dict
            Extra arguments forwarded to `val.plot(...)`.
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
        """Invoke the updater as a callable.

        Parameters
        ----------
        i : int
            Frame number.

        Returns
        -------
        None
        """
        self.update(i)

    def update(self, i: int):
        """Render the slice for the specified frame.

        Parameters
        ----------
        i : int
            Frame number.

        Returns
        -------
        None
        """
        data = self.data
        axis = self.axis
        title = self.title
        notitle = self.notitle
        offsets = self.offsets
        use_si = self.use_si
        kwargs = self.kwargs

        # Slice along the specified axis
        slices = [slice(None)] * len(data.shape)
        slices[axis] = i
        val = data[tuple(slices)]

        # Set the title
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

            if use_si:  # Use SI units
                axisunit = data.axisunits[ax]
                _title = f"{title}({axisunit.reverse(index):.4e} {axisunit.unit}"

            else:  # Use EMSES units
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
        """Return the array with an offset applied.

        Parameters
        ----------
        line : List
            Coordinate array.
        offset : Union[str, float]
            Offset specification. `'left'` / `'center'` / `'right'` or a numeric value.

        Returns
        -------
        object
            Offset-adjusted array.
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
        """Convert to an Animator.

        Parameters
        ----------
        layout: List[List[List[FrameUpdater]]]
            Layout for the animation plot.
        """
        if layout is None:
            layout = [[[self]]]

        return Animator(layout=layout)

    def __len__(self):
        """Return the number of elements.

        Returns
        -------
        int
            Number of elements.
        """
        return self.data.shape[self.axis]
