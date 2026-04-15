"""Context manager that transparently executes pyplot calls on the server side.

Inside a ``with remote_figure():`` block, all ``data.plot()`` and
``plt.xlabel()`` calls are recorded as commands. When the block exits,
the commands are replayed on the Dask worker and only the PNG bytes are
returned to the client.

No memory beyond the image data is allocated locally.

The ``RemoteFigure`` class offers the same functionality via an
``open()`` / ``close()`` API, which is convenient when a ``with`` block
is hard to introduce into existing code::

    rf = RemoteFigure()
    rf.open()
    data.phisp[-1, :, 100, :].plot()
    plt.xlabel("x [m]")
    rf.close()   # <- PNG is displayed
"""

from __future__ import annotations

import contextlib
import itertools
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Global recording state
# ---------------------------------------------------------------------------

_recording: bool = False
_commands: List[Tuple[str, Any, ...]] = []
_session: Any = None  # the shared RemoteSession bound during recording
_proxy_counter = itertools.count()
_current_figure_id: Optional[str] = None
_current_axes_id: Optional[str] = None

_PLOT_PROXY_MARKER = "__remote_plot_proxy__"


def _next_proxy_id(prefix: str) -> str:
    return f"{prefix}_{next(_proxy_counter)}"


def _set_current(figure_id: Optional[str] = None, axes_id: Optional[str] = None) -> None:
    global _current_figure_id, _current_axes_id
    if figure_id is not None:
        _current_figure_id = figure_id
    _current_axes_id = axes_id


def _encode_command_value(value):
    proxy_kind = getattr(value, "_remote_plot_kind", None)
    proxy_id = getattr(value, "_remote_plot_id", None)
    if proxy_kind is not None and proxy_id is not None:
        payload = {"kind": proxy_kind, "id": proxy_id}
        figure_id = getattr(value, "_figure_id", None)
        axis_name = getattr(value, "_axis_name", None)
        spine_name = getattr(value, "_spine_name", None)
        if figure_id is not None:
            payload["figure_id"] = figure_id
        if axis_name is not None:
            payload["axis_name"] = axis_name
        if spine_name is not None:
            payload["spine_name"] = spine_name
        return {_PLOT_PROXY_MARKER: payload}
    if isinstance(value, dict):
        return {k: _encode_command_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_command_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_encode_command_value(v) for v in value)
    return value


def _encode_target_spec(target):
    if target is None:
        return None
    if isinstance(target, list):
        return [_encode_target_spec(item) for item in target]
    if isinstance(target, tuple):
        return tuple(_encode_target_spec(item) for item in target)
    return _encode_command_value(target)


def _reset_recording_state() -> None:
    global _recording, _commands, _session, _proxy_counter, _current_figure_id, _current_axes_id
    _recording = False
    _commands = []
    _session = None
    _proxy_counter = itertools.count()
    _current_figure_id = None
    _current_axes_id = None


def is_recording() -> bool:
    """Return ``True`` inside a ``remote_figure`` block."""
    return _recording


def record_field_plot(
    attr_name: str,
    recipe_index: tuple,
    plot_kwargs: dict,
    emout_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Record a ``Data*.plot()`` command with the Emout identity."""
    _commands.append(
        (
            "field_plot",
            attr_name,
            recipe_index,
            _encode_command_value(plot_kwargs),
            _encode_command_value(emout_kwargs),
        )
    )


def record_plot_surfaces(
    attr_name: str,
    recipe_index: tuple,
    surfaces,
    plot_kwargs: dict,
    emout_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Record a ``Data3d.plot_surfaces()`` command with the Emout identity."""
    _commands.append(
        (
            "plot_surfaces",
            attr_name,
            recipe_index,
            _encode_command_value(surfaces),
            _encode_command_value(plot_kwargs),
            _encode_command_value(emout_kwargs),
        )
    )


def record_plt_call(method: str, args: tuple, kwargs: dict) -> None:
    """Record a ``plt.*`` call."""
    _commands.append(("plt", method, _encode_command_value(args), _encode_command_value(kwargs)))


def record_figure_call(method: str, args: tuple, kwargs: dict, target=None, figure_id: Optional[str] = None) -> None:
    """Record a figure-creation or figure-level call."""
    _commands.append(
        (
            "figure_call",
            figure_id,
            method,
            _encode_command_value(args),
            _encode_command_value(kwargs),
            _encode_target_spec(target),
        )
    )


def record_axes_call(ax_id: str, method: str, args: tuple, kwargs: dict) -> None:
    """Record an ``Axes`` method call."""
    _commands.append(("axes_call", ax_id, method, _encode_command_value(args), _encode_command_value(kwargs)))


def record_axis_call(ax_id: str, axis_name: str, method: str, args: tuple, kwargs: dict) -> None:
    """Record an ``Axis`` method call."""
    _commands.append(
        (
            "axis_call",
            ax_id,
            axis_name,
            method,
            _encode_command_value(args),
            _encode_command_value(kwargs),
        )
    )


def record_spine_call(ax_id: str, spine_name: str, method: str, args: tuple, kwargs: dict) -> None:
    """Record a spine method call."""
    _commands.append(
        (
            "spine_call",
            ax_id,
            spine_name,
            method,
            _encode_command_value(args),
            _encode_command_value(kwargs),
        )
    )


def record_proxy_attr_set(kind: str, proxy_id: str, name: str, value) -> None:
    """Record attribute assignment on a proxy object."""
    _commands.append(("proxy_setattr", kind, proxy_id, name, _encode_command_value(value)))


def record_colorbar_call(colorbar_id: str, method: str, args: tuple, kwargs: dict) -> None:
    """Record a colorbar method call."""
    _commands.append(
        (
            "colorbar_call",
            colorbar_id,
            method,
            _encode_command_value(args),
            _encode_command_value(kwargs),
        )
    )


def record_cached_plot(cache_key: str, plot_kwargs: dict) -> None:
    """Record ``RemoteRef.plot()`` for replay on the worker."""
    _commands.append(("cached_plot", cache_key, _encode_command_value(plot_kwargs)))


def record_cached_plot_surfaces(cache_key: str, surfaces, plot_kwargs: dict) -> None:
    """Record ``RemoteRef.plot_surfaces()`` for replay on the worker."""
    _commands.append(
        (
            "cached_plot_surfaces",
            cache_key,
            _encode_command_value(surfaces),
            _encode_command_value(plot_kwargs),
        )
    )


def record_boundary_plot(
    plot_kwargs: dict,
    emout_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """Record a ``BoundaryCollection.plot()`` command."""
    _commands.append(("boundary_plot", _encode_command_value(plot_kwargs), _encode_command_value(emout_kwargs)))


def record_backtrace_render(cache_key: str, var1: str, var2: str, plot_kwargs: dict) -> None:
    """Record a ``RemoteHeatmap.plot()`` / ``RemoteXYData.plot()`` command."""
    _commands.append(("backtrace_render", cache_key, var1, var2, _encode_command_value(plot_kwargs)))


def record_energy_spectrum(cache_key: str, spec_kwargs: dict) -> None:
    """Record a ``RemoteProbabilityResult.plot_energy_spectrum()`` command."""
    _commands.append(("energy_spectrum", cache_key, _encode_command_value(spec_kwargs)))


def bind_session(session) -> None:
    """Set *session* as the replay target for the current ``remote_figure`` block."""
    global _session
    if session is not None:
        _session = session


def request_session(emout_kwargs: Optional[dict[str, Any]]) -> None:
    """Resolve and bind the shared session.

    Called by ``Data._try_remote_plot()`` and
    ``BoundaryCollection.plot()`` which don't have a session object yet.
    """
    global _session
    if emout_kwargs is None:
        return
    from .remote_render import get_or_create_session

    session = get_or_create_session(emout_kwargs=emout_kwargs)
    if session is not None:
        _session = session


# ---------------------------------------------------------------------------
# Monkey-patch target list (shared between RemoteFigure and remote_figure)
# ---------------------------------------------------------------------------

_PLT_METHODS = [
    "xlabel",
    "ylabel",
    "title",
    "suptitle",
    "xlim",
    "ylim",
    "clim",
    "axhline",
    "axvline",
    "legend",
    "colorbar",
    "tight_layout",
    "grid",
    "text",
    "annotate",
    "xticks",
    "yticks",
    "subplot",
    "subplots",
    "figure",
    "savefig",
]

_PLT_PROXY_METHODS = [
    "figure",
    "subplot",
    "subplots",
    "gca",
    "gcf",
    "sca",
]


class _RemotePlotProxyBase:
    _remote_plot_kind: str
    _remote_plot_id: str

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        record_proxy_attr_set(self._remote_plot_kind, self._remote_plot_id, name, value)

    def __repr__(self):
        return f"<{type(self).__name__} id={self._remote_plot_id!r}>"


class FigureProxy(_RemotePlotProxyBase):
    """Client-side proxy for a remote matplotlib Figure."""

    def __init__(self, figure_id: str):
        object.__setattr__(self, "_remote_plot_kind", "figure")
        object.__setattr__(self, "_remote_plot_id", figure_id)

    def add_axes(self, *args, **kwargs):
        ax = AxesProxy(_next_proxy_id("ax"), self._remote_plot_id)
        record_figure_call("add_axes", args, kwargs, target=ax, figure_id=self._remote_plot_id)
        _set_current(self._remote_plot_id, ax._remote_plot_id)
        return ax

    def add_subplot(self, *args, **kwargs):
        ax = AxesProxy(_next_proxy_id("ax"), self._remote_plot_id)
        record_figure_call("add_subplot", args, kwargs, target=ax, figure_id=self._remote_plot_id)
        _set_current(self._remote_plot_id, ax._remote_plot_id)
        return ax

    def colorbar(self, *args, **kwargs):
        cax = kwargs.get("cax")
        cax_proxy = cax if isinstance(cax, AxesProxy) else AxesProxy(_next_proxy_id("ax"), self._remote_plot_id)
        cbar = ColorbarProxy(_next_proxy_id("cbar"), cax_proxy)
        record_figure_call(
            "colorbar",
            args,
            kwargs,
            target={"colorbar": cbar, "cax": cax_proxy},
            figure_id=self._remote_plot_id,
        )
        return cbar

    def __getattr__(self, name: str):
        def _recorder(*args, **kwargs):
            record_figure_call(name, args, kwargs, figure_id=self._remote_plot_id)
            return None

        return _recorder


class AxesProxy(_RemotePlotProxyBase):
    """Client-side proxy for a remote matplotlib Axes."""

    def __init__(self, axes_id: str, figure_id: Optional[str] = None):
        object.__setattr__(self, "_remote_plot_kind", "axes")
        object.__setattr__(self, "_remote_plot_id", axes_id)
        object.__setattr__(self, "_figure_id", figure_id)
        object.__setattr__(self, "azim", -60.0)
        object.__setattr__(self, "elev", 30.0)

    def __setattr__(self, name: str, value) -> None:
        if name in {"azim", "elev", "computed_zorder"}:
            object.__setattr__(self, name, value)
            record_proxy_attr_set(self._remote_plot_kind, self._remote_plot_id, name, value)
            return
        super().__setattr__(name, value)

    @property
    def figure(self):
        if self._figure_id is None:
            raise AttributeError("This AxesProxy is not bound to a figure")
        return FigureProxy(self._figure_id)

    @property
    def xaxis(self):
        return AxisProxy(self._remote_plot_id, "xaxis", self._figure_id)

    @property
    def yaxis(self):
        return AxisProxy(self._remote_plot_id, "yaxis", self._figure_id)

    @property
    def zaxis(self):
        return AxisProxy(self._remote_plot_id, "zaxis", self._figure_id)

    @property
    def spines(self):
        return SpinesProxy(self._remote_plot_id, self._figure_id)

    def __getattr__(self, name: str):
        def _recorder(*args, **kwargs):
            _set_current(self._figure_id, self._remote_plot_id)
            if name == "view_init":
                elev = kwargs.get("elev", args[0] if len(args) >= 1 else None)
                azim = kwargs.get("azim", args[1] if len(args) >= 2 else None)
                if elev is not None:
                    object.__setattr__(self, "elev", elev)
                if azim is not None:
                    object.__setattr__(self, "azim", azim)
            record_axes_call(self._remote_plot_id, name, args, kwargs)
            return None

        return _recorder


class AxisProxy(_RemotePlotProxyBase):
    """Client-side proxy for ``Axes.xaxis`` / ``yaxis`` / ``zaxis``."""

    def __init__(self, axes_id: str, axis_name: str, figure_id: Optional[str] = None):
        object.__setattr__(self, "_remote_plot_kind", "axis")
        object.__setattr__(self, "_remote_plot_id", axes_id)
        object.__setattr__(self, "_axis_name", axis_name)
        object.__setattr__(self, "_figure_id", figure_id)

    def __getattr__(self, name: str):
        def _recorder(*args, **kwargs):
            _set_current(self._figure_id, self._remote_plot_id)
            record_axis_call(self._remote_plot_id, self._axis_name, name, args, kwargs)
            return None

        return _recorder


class SpineProxy(_RemotePlotProxyBase):
    """Client-side proxy for a single axes spine."""

    def __init__(self, axes_id: str, spine_name: str, figure_id: Optional[str] = None):
        object.__setattr__(self, "_remote_plot_kind", "spine")
        object.__setattr__(self, "_remote_plot_id", axes_id)
        object.__setattr__(self, "_spine_name", spine_name)
        object.__setattr__(self, "_figure_id", figure_id)

    def __getattr__(self, name: str):
        def _recorder(*args, **kwargs):
            _set_current(self._figure_id, self._remote_plot_id)
            record_spine_call(self._remote_plot_id, self._spine_name, name, args, kwargs)
            return None

        return _recorder


class SpinesProxy:
    """Container proxy for ``Axes.spines``."""

    _ORDER = ("left", "right", "bottom", "top")

    def __init__(self, axes_id: str, figure_id: Optional[str] = None):
        self._axes_id = axes_id
        self._figure_id = figure_id

    def __getitem__(self, name: str):
        return SpineProxy(self._axes_id, name, self._figure_id)

    def values(self):
        return [self[name] for name in self._ORDER]


class ColorbarProxy(_RemotePlotProxyBase):
    """Client-side proxy for a remote matplotlib Colorbar."""

    def __init__(self, colorbar_id: str, cax_proxy: AxesProxy):
        object.__setattr__(self, "_remote_plot_kind", "colorbar")
        object.__setattr__(self, "_remote_plot_id", colorbar_id)
        object.__setattr__(self, "_ax_proxy", cax_proxy)
        object.__setattr__(self, "_figure_id", getattr(cax_proxy, "_figure_id", None))

    @property
    def ax(self):
        return self._ax_proxy

    def __getattr__(self, name: str):
        def _recorder(*args, **kwargs):
            record_colorbar_call(self._remote_plot_id, name, args, kwargs)
            return None

        return _recorder


def _infer_subplots_shape(args: tuple, kwargs: dict) -> tuple[int, int, bool]:
    nrows = kwargs.get("nrows", 1)
    ncols = kwargs.get("ncols", 1)
    if len(args) >= 1:
        nrows = args[0]
    if len(args) >= 2:
        ncols = args[1]
    return int(nrows), int(ncols), bool(kwargs.get("squeeze", True))


def _build_subplots_proxies(figure_id: str, nrows: int, ncols: int, squeeze: bool):
    import numpy as np

    grid: list[list[AxesProxy]] = []
    for _row in range(nrows):
        row = []
        for _col in range(ncols):
            row.append(AxesProxy(_next_proxy_id("ax"), figure_id))
        grid.append(row)

    arr = np.array(grid, dtype=object)
    if squeeze:
        arr = np.squeeze(arr)
        if nrows == 1 and ncols == 1:
            return grid[0][0], grid
    return arr, grid


def _resolve_output_format(fmt: Optional[str], savefilepath: Optional[Path]) -> str:
    if fmt is not None:
        return fmt
    if savefilepath is None:
        return "png"
    suffix = savefilepath.suffix.lower().lstrip(".")
    if not suffix:
        return "png"
    if suffix == "jpg":
        return "jpeg"
    return suffix


def _save_image_bytes(img_bytes: bytes, savefilepath: Path) -> None:
    savefilepath.parent.mkdir(parents=True, exist_ok=True)
    savefilepath.write_bytes(img_bytes)


def _has_active_ipython() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def _can_display_image(fmt: str) -> bool:
    return fmt.lower() in {"png", "jpg", "jpeg"}


# ---------------------------------------------------------------------------
# RemoteFigure class -- open/close and context manager
# ---------------------------------------------------------------------------


class RemoteFigure:
    """Execute matplotlib operations on the server side.

    Can be used as a ``with`` statement or via ``open()`` / ``close()``.

    Parameters
    ----------
    session : RemoteSession, optional
        Actor to use. Auto-detected from *emout_dir* if omitted.
    emout_dir : str, optional
        Directory used to look up the Actor when *session* is not given.
    emout_kwargs : dict, optional
        Argument set for reconstructing ``Emout(...)``.
    fmt : str
        Output image format. Defaults to ``png`` unless inferred from
        *savefilepath*.
    dpi : int
        Output resolution.
    figsize : tuple, optional
        Figure size (width, height).
    savefilepath : str or Path, optional
        Save rendered image bytes to this path. When provided, the image
        is still displayed in active IPython sessions for displayable
        formats.

    Usage::

        # with statement
        with RemoteFigure() as rf:
            data.phisp[-1, :, 100, :].plot()
            plt.xlabel("x [m]")

        # open/close
        rf = RemoteFigure()
        rf.open()
        data.phisp[-1, :, 100, :].plot()
        plt.xlabel("x [m]")
        rf.close()

    After ``open()`` the ``fig`` attribute holds a :class:`FigureProxy`
    that forwards calls to the matplotlib ``Figure`` constructed on the
    worker, so callers can issue ``rf.fig.add_axes(...)`` or use the
    ``remote_figure(...) as fig`` contextmanager form without going
    through ``plt.figure()``.
    """

    def __init__(
        self,
        session=None,
        emout_dir: Optional[str] = None,
        emout_kwargs: Optional[dict[str, Any]] = None,
        fmt: Optional[str] = None,
        dpi: int = 150,
        figsize: Optional[Tuple[float, float]] = None,
        savefilepath: Optional[Union[Path, str]] = None,
    ):
        self._init_session = session
        self._emout_dir = emout_dir
        self._emout_kwargs = emout_kwargs
        self.savefilepath = Path(savefilepath).expanduser() if savefilepath is not None else None
        self.fmt = _resolve_output_format(fmt, self.savefilepath)
        self.dpi = dpi
        self.figsize = figsize
        self.fig: Optional[FigureProxy] = None
        self._originals: dict[str, Any] = {}
        self._opened = False

    @property
    def is_open(self) -> bool:
        """Whether the figure is currently in the open (recording) state."""
        return self._opened

    def open(self) -> "RemoteFigure":
        """Start recording mode.

        Returns
        -------
        self
            For method chaining or ``with`` statement usage.
        """
        global _recording, _commands, _session

        if self._opened:
            raise RuntimeError("RemoteFigure is already open")

        session = self._init_session
        if session is None:
            from .remote_render import get_or_create_session

            if self._emout_kwargs is not None or self._emout_dir is not None:
                session = get_or_create_session(
                    emout_dir=self._emout_dir,
                    emout_kwargs=self._emout_kwargs,
                )

        _recording = True
        _commands = []
        _session = session

        fig_kwargs: dict[str, Any] = {}
        if self.figsize is not None:
            fig_kwargs["figsize"] = self.figsize
        self.fig = FigureProxy(_next_proxy_id("fig"))
        record_figure_call("figure", (), fig_kwargs, target=self.fig)
        _set_current(self.fig._remote_plot_id, None)

        # Monkey-patch plt functions to record instead of execute
        import matplotlib.pyplot as plt

        self._originals = {}
        for name in sorted(set(_PLT_METHODS + _PLT_PROXY_METHODS)):
            orig = getattr(plt, name, None)
            if orig is not None:
                self._originals[name] = orig

                if name == "figure":

                    def _record_figure(*args, **kwargs):
                        fig = FigureProxy(_next_proxy_id("fig"))
                        record_figure_call("figure", args, kwargs, target=fig)
                        _set_current(fig._remote_plot_id, None)
                        return fig

                    setattr(plt, name, _record_figure)

                elif name == "subplot":

                    def _record_subplot(*args, **kwargs):
                        figure_id = _current_figure_id or _next_proxy_id("fig")
                        ax = AxesProxy(_next_proxy_id("ax"), figure_id)
                        record_figure_call(
                            "subplot", args, kwargs, target={"figure": FigureProxy(figure_id), "axes": ax}
                        )
                        _set_current(figure_id, ax._remote_plot_id)
                        return ax

                    setattr(plt, name, _record_subplot)

                elif name == "subplots":

                    def _record_subplots(*args, **kwargs):
                        figure_id = _next_proxy_id("fig")
                        nrows, ncols, squeeze = _infer_subplots_shape(args, kwargs)
                        axes_return, axes_grid = _build_subplots_proxies(figure_id, nrows, ncols, squeeze)
                        record_figure_call(
                            "subplots",
                            args,
                            kwargs,
                            target={"figure": FigureProxy(figure_id), "axes_grid": axes_grid},
                        )
                        first_ax = axes_grid[0][0]
                        _set_current(figure_id, first_ax._remote_plot_id)
                        return FigureProxy(figure_id), axes_return

                    setattr(plt, name, _record_subplots)

                elif name == "gcf":

                    def _record_gcf(*args, **kwargs):
                        if _current_figure_id is None:
                            fig = FigureProxy(_next_proxy_id("fig"))
                            record_figure_call("figure", args, kwargs, target=fig)
                            _set_current(fig._remote_plot_id, None)
                            return fig
                        return FigureProxy(_current_figure_id)

                    setattr(plt, name, _record_gcf)

                elif name == "gca":

                    def _record_gca(*args, **kwargs):
                        if _current_axes_id is not None:
                            return AxesProxy(_current_axes_id, _current_figure_id)
                        figure_id = _current_figure_id or _next_proxy_id("fig")
                        ax = AxesProxy(_next_proxy_id("ax"), figure_id)
                        record_figure_call("gca", args, kwargs, target={"figure": FigureProxy(figure_id), "axes": ax})
                        _set_current(figure_id, ax._remote_plot_id)
                        return ax

                    setattr(plt, name, _record_gca)

                elif name == "sca":

                    def _record_sca(ax, *args, **kwargs):
                        if isinstance(ax, AxesProxy):
                            _set_current(ax._figure_id, ax._remote_plot_id)
                        record_plt_call("sca", (ax, *args), kwargs)
                        return ax

                    setattr(plt, name, _record_sca)

                elif name == "colorbar":

                    def _record_colorbar(*args, **kwargs):
                        figure_id = _current_figure_id
                        if figure_id is None:
                            fig = FigureProxy(_next_proxy_id("fig"))
                            record_figure_call("figure", (), {}, target=fig)
                            figure_id = fig._remote_plot_id
                            _set_current(figure_id, None)
                        cax = kwargs.get("cax")
                        cax_proxy = cax if isinstance(cax, AxesProxy) else AxesProxy(_next_proxy_id("ax"), figure_id)
                        cbar = ColorbarProxy(_next_proxy_id("cbar"), cax_proxy)
                        record_figure_call(
                            "colorbar",
                            args,
                            kwargs,
                            target={"colorbar": cbar, "cax": cax_proxy},
                            figure_id=figure_id,
                        )
                        return cbar

                    setattr(plt, name, _record_colorbar)

                else:

                    def _make_recorder(n):
                        def _recorder(*args, **kwargs):
                            record_plt_call(n, args, kwargs)

                        return _recorder

                    setattr(plt, name, _make_recorder(name))

        self._opened = True
        return self

    def close(self) -> None:
        """Stop recording and replay commands on the shared worker."""
        if not self._opened:
            return

        global _recording
        _recording = False

        # Restore plt
        import matplotlib.pyplot as plt

        for name, orig in self._originals.items():
            setattr(plt, name, orig)
        self._originals = {}
        self.fig = None

        # Replay all commands on the shared session
        replay_session = self._init_session or _session
        if replay_session is not None and _commands:
            img_bytes = replay_session.replay_figure(
                _commands,
                fmt=self.fmt,
                dpi=self.dpi,
            ).result()

            if self.savefilepath is not None:
                _save_image_bytes(img_bytes, self.savefilepath)

            if _can_display_image(self.fmt):
                if self.savefilepath is None or _has_active_ipython():
                    from .remote_render import display_image

                    display_image(img_bytes)
            elif self.savefilepath is None:
                warnings.warn(
                    f"Format {self.fmt!r} cannot be displayed automatically; use savefilepath to keep the output.",
                    UserWarning,
                    stacklevel=2,
                )

        _reset_recording_state()
        self._opened = False

    def __enter__(self) -> "RemoteFigure":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self):
        if self._opened:
            warnings.warn(
                "RemoteFigure was not closed. Call .close() or use 'with' statement.",
                ResourceWarning,
                stacklevel=2,
            )
            try:
                self.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Context manager (convenience wrapper)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def remote_figure(
    session=None,
    emout_dir: Optional[str] = None,
    emout_kwargs: Optional[dict[str, Any]] = None,
    fmt: Optional[str] = None,
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
    savefilepath: Optional[Union[Path, str]] = None,
):
    """Context manager that executes matplotlib operations on the server side.

    Parameters
    ----------
    session : RemoteSession, optional
        Actor to use. Auto-detected from *emout_dir* if omitted.
    emout_dir : str, optional
        Directory used to look up the Actor when *session* is not given.
    emout_kwargs : dict, optional
        Argument set for reconstructing ``Emout(...)``.
        Used when ``input_path`` / ``output_directory`` are involved.
    fmt : str
        Output image format. Defaults to ``png`` unless inferred from
        *savefilepath*.
    dpi : int
        Output resolution.
    figsize : tuple, optional
        Figure size (width, height).
    savefilepath : str or Path, optional
        Save rendered image bytes to this path. In CLI/batch usage this
        suppresses local display; in IPython, PNG/JPEG output is also
        displayed inline.

    Usage::

        with remote_figure():
            data.phisp[-1, :, 100, :].plot()
            plt.axhline(y=50, color="red")
            plt.xlabel("x [m]")
        # <- PNG is displayed in Jupyter

    The context manager yields a :class:`FigureProxy` bound to a
    ``Figure`` that is created on the worker, which lets callers build
    the layout without going through ``plt.figure()``::

        with remote_figure(figsize=(13, 6), dpi=300) as fig:
            ax = fig.add_axes([0.1, 0.1, 0.6, 0.8], projection="3d")
            cax = fig.add_axes([0.78, 0.15, 0.04, 0.7])
            data.phisp[-1].plot_surfaces(ax=ax, surfaces=data.boundaries)
    """
    rf = RemoteFigure(
        session=session,
        emout_dir=emout_dir,
        emout_kwargs=emout_kwargs,
        fmt=fmt,
        dpi=dpi,
        figsize=figsize,
        savefilepath=savefilepath,
    )
    rf.open()
    try:
        yield rf.fig
    finally:
        rf.close()


# ---------------------------------------------------------------------------
# IPython / Jupyter cell magic
# ---------------------------------------------------------------------------


def _parse_magic_line(line: str) -> dict[str, Any]:
    """Parse the argument line of ``%%remote_figure``."""
    import shlex

    tokens = shlex.split(line) if line.strip() else []
    kwargs: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("--dpi", "-d") and i + 1 < len(tokens):
            kwargs["dpi"] = int(tokens[i + 1])
            i += 2
        elif tok in ("--fmt", "-f") and i + 1 < len(tokens):
            kwargs["fmt"] = tokens[i + 1]
            i += 2
        elif tok == "--figsize" and i + 1 < len(tokens):
            parts = tokens[i + 1].split(",")
            kwargs["figsize"] = (float(parts[0]), float(parts[1]))
            i += 2
        elif tok == "--emout-dir" and i + 1 < len(tokens):
            kwargs["emout_dir"] = tokens[i + 1]
            i += 2
        elif tok == "--savefilepath" and i + 1 < len(tokens):
            kwargs["savefilepath"] = tokens[i + 1]
            i += 2
        else:
            i += 1
    return kwargs


def register_magics(ipython=None) -> None:
    """Register the ``%%remote_figure`` cell magic in IPython / Jupyter.

    Usage::

        from emout.distributed.remote_figure import register_magics
        register_magics()

    or::

        %load_ext emout.distributed.remote_figure

    After registration, prepend ``%%remote_figure`` to a cell to have all
    ``data.plot()`` / ``plt.*`` calls executed on the Dask worker::

        %%remote_figure
        data.phisp[-1, :, 100, :].plot()
        plt.xlabel("x [m]")

        %%remote_figure --dpi 300 --fmt svg --figsize 12,6
        data.phisp[-1, :, 100, :].plot()
    """
    try:
        from IPython.core.magic import register_cell_magic
    except ImportError:
        raise ImportError("IPython is required to register magics")

    if ipython is None:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            raise RuntimeError("No active IPython session found")

    @register_cell_magic
    def remote_figure(line, cell):  # noqa: F811 -- shadows module-level name intentionally
        kwargs = _parse_magic_line(line)
        rf = RemoteFigure(**kwargs)
        rf.open()
        try:
            ipython.ex(cell)
        finally:
            rf.close()


def load_ipython_extension(ipython) -> None:
    """Entry point for ``%load_ext emout.distributed.remote_figure``."""
    register_magics(ipython)
