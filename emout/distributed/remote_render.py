"""Worker 側で可視化をレンダリングし PNG bytes だけを返す Dask Actor 基盤.

設計思想
--------
- 重い計算（backtrace 等）は worker で 1 回だけ実行し、結果を worker メモリに保持
- 可視化パラメータ（cmap, vmin, vmax, 射影軸など）を変えて何度でも再レンダリング
- client に転送されるのは PNG/SVG bytes（数十 KB）だけ
- ユーザーは ``result.vxvz.plot()`` のように通常どおりのインタフェースで使える
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

_key_counter = itertools.count()


def _next_key(prefix: str = "result") -> str:
    return f"{prefix}_{next(_key_counter)}"


# ---------------------------------------------------------------------------
# Worker-side Actor
# ---------------------------------------------------------------------------


class RemoteSession:
    """Dask Actor: worker プロセスに常駐し、計算結果をキャッシュしつつ再描画に応える。

    ``client.submit(RemoteSession, emout_dir, actor=True)`` で生成する。
    """

    def __init__(
        self,
        emout_dir: str | None = None,
        input_path: str | None = None,
        emout_kwargs: dict[str, Any] | None = None,
    ):
        import matplotlib
        matplotlib.use("Agg")

        import emout
        self._emout_kwargs = _normalize_emout_kwargs(
            emout_dir=emout_dir,
            input_path=input_path,
            emout_kwargs=emout_kwargs,
        )
        self._data = emout.Emout(**self._emout_kwargs)
        self._cache: dict[str, Any] = {}

    # -- computation (result stays on worker) --------------------------------

    def compute_probabilities(self, key: str, **kwargs) -> bool:
        result = self._data.backtrace.get_probabilities(**kwargs)
        self._cache[key] = result
        return True

    def compute_backtraces(self, key: str, **kwargs) -> bool:
        result = self._data.backtrace.get_backtraces_from_particles(**kwargs)
        self._cache[key] = result
        return True

    # -- rendering (PNG bytes returned to client) ----------------------------

    def render_pair(
        self, key: str, var1: str, var2: str,
        fmt: str = "png", dpi: int = 150, **plot_kwargs,
    ) -> bytes:
        import matplotlib.pyplot as plt
        from io import BytesIO

        result = self._cache[key]
        heatmap = result.pair(var1, var2)

        fig, ax = plt.subplots()
        heatmap.plot(ax=ax, **plot_kwargs)

        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def render_energy_spectrum(
        self, key: str, energy_bins=None, scale: str = "log",
        fmt: str = "png", dpi: int = 150,
    ) -> bytes:
        import matplotlib.pyplot as plt
        from io import BytesIO

        result = self._cache[key]
        fig, ax = plt.subplots()
        plt.sca(ax)
        result.plot_energy_spectrum(energy_bins=energy_bins, scale=scale)

        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def render_backtrace_pair(
        self, key: str, var1: str, var2: str,
        fmt: str = "png", dpi: int = 150, **plot_kwargs,
    ) -> bytes:
        import matplotlib.pyplot as plt
        from io import BytesIO

        result = self._cache[key]
        xy = result.pair(var1, var2)

        fig, ax = plt.subplots()
        xy.plot(ax=ax, **plot_kwargs)

        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def render_field(
        self, attr_name: str, index: tuple,
        fmt: str = "png", dpi: int = 150, **plot_kwargs,
    ) -> bytes:
        """フィールドデータ (data.phisp[-1, :, 100, :] 等) を描画して PNG bytes で返す。"""
        import matplotlib.pyplot as plt
        from io import BytesIO

        arr = getattr(self._data, attr_name)[index]

        fig, ax = plt.subplots()
        plt.sca(ax)
        arr.plot(**plot_kwargs)

        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def fetch_field(self, attr_name: str, index: tuple) -> dict:
        """フィールドのスライス済みデータ + メタデータを返す（ローカル描画用）。

        全 3D 配列ではなく、スライス済みの小さな配列（2D: 数 KB〜数 MB）だけを
        転送するので、ローカルで plt.axhline() 等を重ねられる。
        """
        arr = getattr(self._data, attr_name)[index]
        return {
            "array": np.asarray(arr),
            "name": arr.name,
            "slices": arr.slices,
            "slice_axes": arr.slice_axes,
            "axisunits": arr.axisunits,
            "valunit": arr.valunit,
        }

    def render_plot_surfaces(
        self, attr_name: str, t_index: int,
        use_si: bool = True, fmt: str = "png", dpi: int = 150,
        **plot_kwargs,
    ) -> bytes:
        """Data3d.plot_surfaces を worker 上で実行し PNG bytes を返す。"""
        import matplotlib.pyplot as plt
        from io import BytesIO

        data3d = getattr(self._data, attr_name)[t_index]
        boundaries = self._data.boundaries

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        data3d.plot_surfaces(
            surfaces=boundaries, ax=ax, use_si=use_si, **plot_kwargs,
        )

        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def render_boundaries(
        self, use_si: bool = True, fmt: str = "png", dpi: int = 150,
        **plot_kwargs,
    ) -> bytes:
        """BoundaryCollection.plot() を worker 上で実行し PNG bytes を返す。"""
        import matplotlib.pyplot as plt
        from io import BytesIO

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self._data.boundaries.plot(ax=ax, use_si=use_si, **plot_kwargs)

        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def replay_figure(self, commands: list, fmt: str = "png", dpi: int = 150) -> bytes:
        """コマンドリストを順に再生し、レンダリング結果を返す。

        ``remote_figure()`` コンテキストが収集したコマンドを worker 上の
        matplotlib で再生する。コマンド形式:

        - ``("field_plot", attr_name, recipe_index, plot_kwargs)``
        - ``("plt", method_name, args, kwargs)``
        - ``("boundary_plot", plot_kwargs)``
        - ``("backtrace_render", cache_key, var1, var2, plot_kwargs)``
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO

        for cmd in commands:
            kind = cmd[0]
            if kind == "field_plot":
                _, attr_name, recipe_index, plot_kwargs = cmd
                arr = getattr(self._data, attr_name)[recipe_index]
                arr.plot(**plot_kwargs)

            elif kind == "plt":
                _, method_name, args, kwargs = cmd
                func = getattr(plt, method_name)
                func(*args, **kwargs)

            elif kind == "boundary_plot":
                _, plot_kwargs = cmd
                ax = plt.gca()
                self._data.boundaries.plot(ax=ax, **plot_kwargs)

            elif kind == "backtrace_render":
                _, cache_key, var1, var2, plot_kwargs = cmd
                result = self._cache[cache_key]
                heatmap = result.pair(var1, var2)
                ax = plt.gca()
                heatmap.plot(ax=ax, **plot_kwargs)

            elif kind == "energy_spectrum":
                _, cache_key, spec_kwargs = cmd
                result = self._cache[cache_key]
                result.plot_energy_spectrum(**spec_kwargs)

        buf = BytesIO()
        plt.gcf().savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close("all")
        return buf.getvalue()

    def drop(self, key: str) -> None:
        """キャッシュから結果を削除してメモリを解放する。"""
        self._cache.pop(key, None)

    def keys(self) -> list[str]:
        return list(self._cache.keys())


# ---------------------------------------------------------------------------
# Client-side proxies (same interface as real result objects)
# ---------------------------------------------------------------------------


class RemoteHeatmap:
    """``HeatmapData.plot()`` と同じインタフェースの proxy。worker 側でレンダリングする。"""

    def __init__(self, session: RemoteSession, cache_key: str, var1: str, var2: str):
        self._session = session
        self._key = cache_key
        self._var1 = var1
        self._var2 = var2

    def plot(self, ax=None, fmt: str = "png", dpi: int = 150, **plot_kwargs):
        from .remote_figure import bind_session, is_recording, record_backtrace_render
        if is_recording():
            bind_session(self._session)
            record_backtrace_render(self._key, self._var1, self._var2, plot_kwargs)
            return None
        img = self._session.render_pair(
            self._key, self._var1, self._var2, fmt=fmt, dpi=dpi, **plot_kwargs,
        ).result()
        return display_image(img, ax=ax)

    def __repr__(self):
        return f"<RemoteHeatmap: {self._var1} vs {self._var2} (key={self._key!r})>"


class RemoteXYData:
    """``XYData.plot()`` / ``MultiXYData.plot()`` と同じインタフェースの proxy。"""

    def __init__(self, session: RemoteSession, cache_key: str, var1: str, var2: str):
        self._session = session
        self._key = cache_key
        self._var1 = var1
        self._var2 = var2

    def plot(self, ax=None, fmt: str = "png", dpi: int = 150, **plot_kwargs):
        from .remote_figure import bind_session, is_recording, record_backtrace_render
        if is_recording():
            bind_session(self._session)
            record_backtrace_render(self._key, self._var1, self._var2, plot_kwargs)
            return None
        img = self._session.render_backtrace_pair(
            self._key, self._var1, self._var2, fmt=fmt, dpi=dpi, **plot_kwargs,
        ).result()
        return display_image(img, ax=ax)

    def __repr__(self):
        return f"<RemoteXYData: {self._var1} vs {self._var2} (key={self._key!r})>"


class RemoteProbabilityResult:
    """``ProbabilityResult`` と同じインタフェースの proxy。

    ``result.vxvz.plot(cmap="plasma")`` のように、ローカルと全く同じ書き方で使える。
    実際の描画は worker 上で行われ、PNG bytes だけが返る。
    """

    _AXES = ["x", "y", "z", "vx", "vy", "vz"]

    def __init__(self, session: RemoteSession, cache_key: str):
        self._session = session
        self._key = cache_key

    def pair(self, var1: str, var2: str) -> RemoteHeatmap:
        return RemoteHeatmap(self._session, self._key, var1, var2)

    def plot_energy_spectrum(
        self, energy_bins=None, scale: str = "log", fmt: str = "png", dpi: int = 150,
    ):
        from .remote_figure import bind_session, is_recording, record_energy_spectrum
        if is_recording():
            bind_session(self._session)
            record_energy_spectrum(
                self._key,
                {"energy_bins": energy_bins, "scale": scale},
            )
            return None
        img = self._session.render_energy_spectrum(
            self._key, energy_bins=energy_bins, scale=scale, fmt=fmt, dpi=dpi,
        ).result()
        return display_image(img)

    def drop(self) -> None:
        """worker メモリから結果を解放する。"""
        self._session.drop(self._key)

    def __getattr__(self, name: str):
        # ProbabilityResult.__getattr__ と同じ: result.vxvz → pair("vx","vz")
        for key1 in self._AXES:
            if name.startswith(key1):
                rest = name[len(key1):]
                if rest in self._AXES and rest != key1:
                    return self.pair(key1, rest)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self):
        return f"<RemoteProbabilityResult (key={self._key!r})>"


class RemoteBacktraceResult:
    """``BacktraceResult`` / ``MultiBacktraceResult`` と同じ proxy。"""

    _AXES = ["x", "y", "z", "vx", "vy", "vz"]

    def __init__(self, session: RemoteSession, cache_key: str):
        self._session = session
        self._key = cache_key

    def pair(self, var1: str, var2: str) -> RemoteXYData:
        return RemoteXYData(self._session, self._key, var1, var2)

    def drop(self) -> None:
        self._session.drop(self._key)

    def __getattr__(self, name: str):
        for key1 in self._AXES:
            if name.startswith(key1):
                rest = name[len(key1):]
                if rest in self._AXES and rest != key1:
                    return self.pair(key1, rest)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self):
        return f"<RemoteBacktraceResult (key={self._key!r})>"


# ---------------------------------------------------------------------------
# Image display helper
# ---------------------------------------------------------------------------


def display_image(img_bytes: bytes, ax=None):
    """PNG bytes を Jupyter に表示するか、matplotlib axes に描画する。"""
    if ax is None:
        try:
            from IPython.display import display as ipydisplay, Image
            ipydisplay(Image(data=img_bytes))
            return None
        except ImportError:
            pass

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from io import BytesIO

    if ax is None:
        _, ax = plt.subplots()
    img = mpimg.imread(BytesIO(img_bytes))
    ax.imshow(img)
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

_session_cache: dict[str, RemoteSession] = {}


def _normalize_emout_kwargs(
    emout_dir=None,
    input_path=None,
    emout_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if emout_kwargs is None:
        if emout_dir is None:
            raise ValueError("emout_dir or emout_kwargs is required")
        normalized = {"directory": str(Path(emout_dir).resolve())}
        if input_path is not None:
            normalized["input_path"] = str(Path(input_path).resolve())
            normalized["output_directory"] = str(Path(emout_dir).resolve())
    else:
        normalized = dict(emout_kwargs)

    for key in ("directory", "input_path", "output_directory"):
        value = normalized.get(key)
        if value is not None:
            normalized[key] = str(Path(value).resolve())

    append_directories = normalized.get("append_directories")
    if append_directories is not None:
        normalized["append_directories"] = [
            str(Path(path).resolve()) for path in append_directories
        ]

    return normalized


def get_or_create_session(
    emout_dir=None, input_path=None, emout_kwargs: dict[str, Any] | None = None,
) -> Optional[RemoteSession]:
    """Dask client が起動していれば RemoteSession actor を返す。なければ None。

    ``~/.emout/server.json`` が存在し、かつ Dask client がまだ無い場合は
    自動的に ``connect()`` して接続する。これにより ``client = connect()`` を
    スクリプトに明示的に書かなくても、``emout server start`` しておけば
    透過的にリモート実行される。
    """
    if sys.version_info.minor < 10:
        return None

    try:
        from dask.distributed import default_client
    except ImportError:
        return None

    # 既存の client を探す
    try:
        client = default_client()
    except ValueError:
        client = None

    # client が無いが server.json があれば自動接続
    if client is None:
        state_file = Path.home() / ".emout" / "server.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                from dask.distributed import Client
                client = Client(state["address"])
            except Exception:
                return None
        else:
            return None

    if emout_dir is None and emout_kwargs is None:
        return None

    normalized_kwargs = _normalize_emout_kwargs(
        emout_dir=emout_dir,
        input_path=input_path,
        emout_kwargs=emout_kwargs,
    )
    cache_key = json.dumps(normalized_kwargs, sort_keys=True)
    if cache_key not in _session_cache:
        future = client.submit(
            RemoteSession,
            actor=True,
            emout_kwargs=normalized_kwargs,
        )
        _session_cache[cache_key] = future.result()
    return _session_cache[cache_key]


def clear_sessions() -> None:
    """全セッションをクリアする。"""
    _session_cache.clear()
