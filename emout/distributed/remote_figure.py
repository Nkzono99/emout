"""pyplot を透過的にサーバー側で実行するコンテキストマネージャ.

``with remote_figure():`` ブロック内で発行された ``data.plot()`` や
``plt.xlabel()`` などの呼び出しを全てコマンドとして記録し、ブロック終了時に
Dask worker 上で一括再生 → PNG bytes だけをクライアントに返す。

ローカルには画像データ以外のメモリを確保しない。
"""

from __future__ import annotations

import contextlib
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Global recording state
# ---------------------------------------------------------------------------

_recording: bool = False
_commands: List[Tuple[str, Any, ...]] = []
_recording_session = None
_session_request: Optional[dict[str, Any]] = None


def is_recording() -> bool:
    """``remote_figure`` ブロック内かどうかを返す。"""
    return _recording


def record_field_plot(attr_name: str, recipe_index: tuple, plot_kwargs: dict) -> None:
    """``Data*.plot()`` からコマンドを記録する。"""
    _commands.append(("field_plot", attr_name, recipe_index, plot_kwargs))


def record_plt_call(method: str, args: tuple, kwargs: dict) -> None:
    """``plt.*`` 呼び出しを記録する。"""
    _commands.append(("plt", method, args, kwargs))


def record_boundary_plot(plot_kwargs: dict) -> None:
    """``BoundaryCollection.plot()`` を記録する。"""
    _commands.append(("boundary_plot", plot_kwargs))


def record_backtrace_render(cache_key: str, var1: str, var2: str, plot_kwargs: dict) -> None:
    """``RemoteHeatmap.plot()`` / ``RemoteXYData.plot()`` を記録する。"""
    _commands.append(("backtrace_render", cache_key, var1, var2, plot_kwargs))


def record_energy_spectrum(cache_key: str, spec_kwargs: dict) -> None:
    """``RemoteProbabilityResult.plot_energy_spectrum()`` を記録する。"""
    _commands.append(("energy_spectrum", cache_key, spec_kwargs))


def bind_session(session) -> None:
    """記録済みコマンドの再生先セッションを固定する。"""
    global _recording_session
    if session is None:
        return
    if _recording_session is None:
        _recording_session = session
        return
    if _recording_session is not session:
        raise RuntimeError("remote_figure() cannot mix commands from different remote sessions")


def request_session(emout_kwargs: Optional[dict[str, Any]]) -> None:
    """記録されたコマンドからセッション生成条件を登録する。"""
    global _session_request
    if emout_kwargs is None:
        return
    if _session_request is None:
        _session_request = dict(emout_kwargs)
        return
    if _session_request != emout_kwargs:
        raise RuntimeError("remote_figure() cannot mix commands from different Emout sources")


# ---------------------------------------------------------------------------
# Context manager
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
    global _recording, _commands, _recording_session, _session_request

    if session is None:
        from .remote_render import get_or_create_session
        if emout_kwargs is not None or emout_dir is not None:
            session = get_or_create_session(emout_dir=emout_dir, emout_kwargs=emout_kwargs)

    _recording = True
    _commands = []
    _recording_session = session
    _session_request = None
    if figsize is not None:
        _commands.append(("plt", "figure", (), {"figsize": figsize}))

    # Monkey-patch plt functions to record instead of execute
    import matplotlib.pyplot as plt
    _originals = {}
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
    for name in _PLT_METHODS:
        orig = getattr(plt, name, None)
        if orig is not None:
            _originals[name] = orig

            def _make_recorder(n):
                def _recorder(*args, **kwargs):
                    record_plt_call(n, args, kwargs)
                return _recorder
            setattr(plt, name, _make_recorder(name))

    try:
        yield
    finally:
        _recording = False

        # Restore plt
        for name, orig in _originals.items():
            setattr(plt, name, orig)

        # Flush commands to worker
        replay_session = session or _recording_session
        if replay_session is None and _session_request is not None:
            from .remote_render import get_or_create_session
            replay_session = get_or_create_session(emout_kwargs=_session_request)

        if replay_session is not None and _commands:
            img_bytes = replay_session.replay_figure(_commands, fmt=fmt, dpi=dpi).result()
            from .remote_render import display_image
            display_image(img_bytes)

        _commands = []
        _recording_session = None
        _session_request = None
