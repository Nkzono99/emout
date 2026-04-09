"""pyplot を透過的にサーバー側で実行するコンテキストマネージャ.

``with remote_figure():`` ブロック内で発行された ``data.plot()`` や
``plt.xlabel()`` などの呼び出しを全てコマンドとして記録し、ブロック終了時に
Dask worker 上で一括再生 → PNG bytes だけをクライアントに返す。

ローカルには画像データ以外のメモリを確保しない。

``RemoteFigure`` クラスを使えば ``open()`` / ``close()`` 形式でも同じ機能を
使える。既存コードに ``with`` ブロックを導入しづらい場合に便利::

    rf = RemoteFigure()
    rf.open()
    data.phisp[-1, :, 100, :].plot()
    plt.xlabel("x [m]")
    rf.close()   # ← PNG が表示される
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Global recording state
# ---------------------------------------------------------------------------

_recording: bool = False
_commands: List[Tuple[str, Any, ...]] = []
_command_sessions: List[Any] = []  # parallel to _commands: session per command
_last_bound_session: Any = None    # most recently bound session
_session_request: Optional[dict[str, Any]] = None


def _reset_recording_state() -> None:
    global _recording, _commands, _command_sessions
    global _last_bound_session, _session_request
    _recording = False
    _commands = []
    _command_sessions = []
    _last_bound_session = None
    _session_request = None


def is_recording() -> bool:
    """``remote_figure`` ブロック内かどうかを返す。"""
    return _recording


def _record(cmd: Tuple[str, Any, ...]) -> None:
    """Record *cmd* and tag it with the most recently bound session."""
    _commands.append(cmd)
    _command_sessions.append(_last_bound_session)


def record_field_plot(attr_name: str, recipe_index: tuple, plot_kwargs: dict) -> None:
    """``Data*.plot()`` からコマンドを記録する。"""
    _record(("field_plot", attr_name, recipe_index, plot_kwargs))


def record_plt_call(method: str, args: tuple, kwargs: dict) -> None:
    """``plt.*`` 呼び出しを記録する。"""
    _commands.append(("plt", method, args, kwargs))
    _command_sessions.append(None)  # plt calls have no owning session


def record_boundary_plot(plot_kwargs: dict) -> None:
    """``BoundaryCollection.plot()`` を記録する。"""
    _record(("boundary_plot", plot_kwargs))


def record_backtrace_render(cache_key: str, var1: str, var2: str, plot_kwargs: dict) -> None:
    """``RemoteHeatmap.plot()`` / ``RemoteXYData.plot()`` を記録する。"""
    _record(("backtrace_render", cache_key, var1, var2, plot_kwargs))


def record_energy_spectrum(cache_key: str, spec_kwargs: dict) -> None:
    """``RemoteProbabilityResult.plot_energy_spectrum()`` を記録する。"""
    _record(("energy_spectrum", cache_key, spec_kwargs))


def bind_session(session) -> None:
    """Register *session* as the owner of the next recorded data command.

    Multiple sessions are allowed — when the ``remote_figure`` block
    closes and more than one session is detected, the framework
    automatically falls back to *fetch + local render* mode.
    """
    global _last_bound_session
    if session is None:
        return
    _last_bound_session = session


def request_session(emout_kwargs: Optional[dict[str, Any]]) -> None:
    """Eagerly resolve a session from *emout_kwargs* and bind it.

    Called by ``Data._try_remote_plot()`` which doesn't have a session
    object yet.  The session is resolved here so that per-command
    tracking works for multi-session figures.
    """
    global _last_bound_session, _session_request
    if emout_kwargs is None:
        return
    # Eagerly resolve the session for per-command tracking
    from .remote_render import get_or_create_session
    session = get_or_create_session(emout_kwargs=emout_kwargs)
    if session is not None:
        _last_bound_session = session
    # Keep first request for single-session fallback
    if _session_request is None:
        _session_request = dict(emout_kwargs)


# ---------------------------------------------------------------------------
# Monkey-patch target list (shared between RemoteFigure and remote_figure)
# ---------------------------------------------------------------------------

_PLT_METHODS = [
    "xlabel", "ylabel", "title", "suptitle",
    "xlim", "ylim", "clim",
    "axhline", "axvline",
    "legend", "colorbar",
    "tight_layout", "grid",
    "text", "annotate",
    "xticks", "yticks",
    "subplot", "subplots",
    "figure",
    "savefig",
]


# ---------------------------------------------------------------------------
# RemoteFigure class — open/close and context manager
# ---------------------------------------------------------------------------


class RemoteFigure:
    """matplotlib 操作をサーバー側で実行するオブジェクト.

    ``with`` 文でも ``open()`` / ``close()`` でも使える。

    Parameters
    ----------
    session : RemoteSession, optional
        使用する Actor。省略時は ``emout_dir`` から自動取得。
    emout_dir : str, optional
        ``session`` が未指定のときに Actor を検索するディレクトリ。
    emout_kwargs : dict, optional
        ``Emout(...)`` を再構成するための引数セット。
    fmt : str
        出力画像フォーマット。
    dpi : int
        出力解像度。
    figsize : tuple, optional
        Figure サイズ (width, height)。

    Usage::

        # with 文
        with RemoteFigure() as rf:
            data.phisp[-1, :, 100, :].plot()
            plt.xlabel("x [m]")

        # open/close
        rf = RemoteFigure()
        rf.open()
        data.phisp[-1, :, 100, :].plot()
        plt.xlabel("x [m]")
        rf.close()
    """

    def __init__(
        self,
        session=None,
        emout_dir: Optional[str] = None,
        emout_kwargs: Optional[dict[str, Any]] = None,
        fmt: str = "png",
        dpi: int = 150,
        figsize: Optional[Tuple[float, float]] = None,
    ):
        self._init_session = session
        self._emout_dir = emout_dir
        self._emout_kwargs = emout_kwargs
        self.fmt = fmt
        self.dpi = dpi
        self.figsize = figsize
        self._originals: dict[str, Any] = {}
        self._opened = False

    @property
    def is_open(self) -> bool:
        """現在 open 状態かどうか。"""
        return self._opened

    def open(self) -> "RemoteFigure":
        """記録モードを開始する。

        Returns
        -------
        self
            メソッドチェーンまたは ``with`` 文向け。
        """
        global _recording, _commands, _command_sessions
        global _last_bound_session, _session_request

        if self._opened:
            raise RuntimeError("RemoteFigure is already open")

        session = self._init_session
        if session is None:
            from .remote_render import get_or_create_session
            if self._emout_kwargs is not None or self._emout_dir is not None:
                session = get_or_create_session(
                    emout_dir=self._emout_dir, emout_kwargs=self._emout_kwargs,
                )

        _recording = True
        _commands = []
        _command_sessions = []
        _last_bound_session = session
        _session_request = None

        if self.figsize is not None:
            _commands.append(("plt", "figure", (), {"figsize": self.figsize}))
            _command_sessions.append(None)

        # Monkey-patch plt functions to record instead of execute
        import matplotlib.pyplot as plt
        self._originals = {}
        for name in _PLT_METHODS:
            orig = getattr(plt, name, None)
            if orig is not None:
                self._originals[name] = orig

                def _make_recorder(n):
                    def _recorder(*args, **kwargs):
                        record_plt_call(n, args, kwargs)
                    return _recorder
                setattr(plt, name, _make_recorder(name))

        self._opened = True
        return self

    def close(self) -> None:
        """記録を停止し、コマンドを worker で再生して画像を表示する。

        単一セッションの場合は worker 上で一括 replay（高効率）。
        複数セッションが検出された場合は各セッションからデータを fetch し、
        ローカルで合成描画にフォールバックする。
        """
        global _recording

        if not self._opened:
            return

        _recording = False

        # Restore plt
        import matplotlib.pyplot as plt
        for name, orig in self._originals.items():
            setattr(plt, name, orig)
        self._originals = {}

        # Determine unique sessions
        unique_sessions = {id(s) for s in _command_sessions if s is not None}

        if len(unique_sessions) <= 1:
            # --- Single session: replay on worker (efficient) ---
            replay_session = self._init_session or _last_bound_session
            if replay_session is None and _session_request is not None:
                from .remote_render import get_or_create_session
                replay_session = get_or_create_session(emout_kwargs=_session_request)

            if replay_session is not None and _commands:
                img_bytes = replay_session.replay_figure(
                    _commands, fmt=self.fmt, dpi=self.dpi,
                ).result()
                from .remote_render import display_image
                display_image(img_bytes)
        else:
            # --- Multi-session: fetch data and render locally ---
            if _commands:
                _replay_locally(
                    _commands, _command_sessions,
                    fmt=self.fmt, dpi=self.dpi, figsize=self.figsize,
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
    fmt: str = "png",
    dpi: int = 150,
    figsize: Optional[Tuple[float, float]] = None,
):
    """matplotlib 操作をサーバー側で実行するコンテキストマネージャ.

    Parameters
    ----------
    session : RemoteSession, optional
        使用する Actor。省略時は ``emout_dir`` から自動取得。
    emout_dir : str, optional
        ``session`` が未指定のときに Actor を検索するディレクトリ。
    emout_kwargs : dict, optional
        ``Emout(...)`` を再構成するための引数セット。
        `input_path` / `output_directory` を含むケースで使われる。
    fmt : str
        出力画像フォーマット。
    dpi : int
        出力解像度。
    figsize : tuple, optional
        Figure サイズ (width, height)。

    Usage::

        with remote_figure():
            data.phisp[-1, :, 100, :].plot()
            plt.axhline(y=50, color="red")
            plt.xlabel("x [m]")
        # ← PNG が Jupyter に表示される
    """
    rf = RemoteFigure(
        session=session, emout_dir=emout_dir, emout_kwargs=emout_kwargs,
        fmt=fmt, dpi=dpi, figsize=figsize,
    )
    rf.open()
    try:
        yield
    finally:
        rf.close()


# ---------------------------------------------------------------------------
# IPython / Jupyter cell magic
# ---------------------------------------------------------------------------


def _parse_magic_line(line: str) -> dict[str, Any]:
    """``%%remote_figure`` の引数行をパースする。"""
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
        else:
            i += 1
    return kwargs


def register_magics(ipython=None) -> None:
    """``%%remote_figure`` セルマジックを IPython / Jupyter に登録する.

    Usage::

        from emout.distributed.remote_figure import register_magics
        register_magics()

    または::

        %load_ext emout.distributed.remote_figure

    登録後はセル先頭に ``%%remote_figure`` と書くだけで、セル内の
    ``data.plot()`` / ``plt.*`` が Dask worker 上で実行される::

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
    def remote_figure(line, cell):  # noqa: F811 — shadows module-level name intentionally
        kwargs = _parse_magic_line(line)
        rf = RemoteFigure(**kwargs)
        rf.open()
        try:
            ipython.ex(cell)
        finally:
            rf.close()


def load_ipython_extension(ipython) -> None:
    """``%load_ext emout.distributed.remote_figure`` 用エントリポイント。"""
    register_magics(ipython)
