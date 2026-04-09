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
        """Run backtrace probability calculation and cache the result.

        Parameters
        ----------
        key : str
            Cache key under which the result is stored.
        **kwargs
            Keyword arguments forwarded to
            ``Emout.backtrace.get_probabilities()``.

        Returns
        -------
        bool
            ``True`` on success.
        """
        result = self._data.backtrace.get_probabilities(**kwargs)
        self._cache[key] = result
        return True

    def compute_backtraces(self, key: str, **kwargs) -> bool:
        """Run particle backtrace calculation and cache the result.

        Parameters
        ----------
        key : str
            Cache key under which the result is stored.
        **kwargs
            Keyword arguments forwarded to
            ``Emout.backtrace.get_backtraces_from_particles()``.

        Returns
        -------
        bool
            ``True`` on success.
        """
        result = self._data.backtrace.get_backtraces_from_particles(**kwargs)
        self._cache[key] = result
        return True

    # -- rendering (PNG bytes returned to client) ----------------------------

    @staticmethod
    def _render_to_bytes(draw_fn, fmt: str = "png", dpi: int = 150) -> bytes:
        """Run *draw_fn* on a fresh figure and return the image bytes.

        Parameters
        ----------
        draw_fn : callable(fig, ax)
            Function that draws onto *fig* / *ax*.
        fmt, dpi : str, int
            Output format and resolution.
        """
        import matplotlib.pyplot as plt
        from io import BytesIO

        fig, ax = plt.subplots()
        draw_fn(fig, ax)
        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def render_pair(
        self, key: str, var1: str, var2: str,
        fmt: str = "png", dpi: int = 150, **plot_kwargs,
    ) -> bytes:
        """Render a 2-D heatmap of a cached probability result and return image bytes."""
        result = self._cache[key]
        heatmap = result.pair(var1, var2)
        return self._render_to_bytes(
            lambda fig, ax: heatmap.plot(ax=ax, **plot_kwargs), fmt, dpi,
        )

    def fetch_heatmap_data(self, key: str, var1: str, var2: str) -> dict:
        """Return serialised HeatmapData arrays for local plotting.

        Parameters
        ----------
        key : str
            Cache key referencing a previously computed probability result.
        var1, var2 : str
            Axis variable names (e.g. ``"vx"``, ``"vz"``).

        Returns
        -------
        dict
            ``{"X", "Y", "Z", "xlabel", "ylabel", "title", "units"}``
            where X/Y/Z are numpy arrays and units is a tuple or None.
        """
        result = self._cache[key]
        heatmap = result.pair(var1, var2)
        units_payload = None
        if heatmap.units is not None:
            units_payload = [
                {"from_unit": u.from_unit, "to_unit": u.to_unit,
                 "name": u.name, "unit": u.unit}
                for u in heatmap.units
            ]
        return {
            "X": np.asarray(heatmap.X),
            "Y": np.asarray(heatmap.Y),
            "Z": np.asarray(heatmap.Z),
            "xlabel": heatmap.xlabel,
            "ylabel": heatmap.ylabel,
            "title": heatmap.title,
            "units": units_payload,
        }

    def fetch_xy_data(self, key: str, var1: str, var2: str) -> dict:
        """Return serialised XYData / MultiXYData arrays for local plotting.

        Parameters
        ----------
        key : str
            Cache key referencing a previously computed backtrace result.
        var1, var2 : str
            Axis variable names (e.g. ``"x"``, ``"vz"``).

        Returns
        -------
        dict
            ``{"x", "y", "xlabel", "ylabel", "title", "units", "last_indexes"}``
            where x/y are numpy arrays.  ``last_indexes`` is present only for
            :class:`MultiXYData`.
        """
        result = self._cache[key]
        xy = result.pair(var1, var2)
        units_payload = None
        if xy.units is not None:
            units_payload = [
                {"from_unit": u.from_unit, "to_unit": u.to_unit,
                 "name": u.name, "unit": u.unit}
                for u in xy.units
            ]
        payload = {
            "x": np.asarray(xy.x),
            "y": np.asarray(xy.y),
            "xlabel": xy.xlabel,
            "ylabel": xy.ylabel,
            "title": xy.title,
            "units": units_payload,
        }
        if hasattr(xy, "last_indexes"):
            payload["last_indexes"] = np.asarray(xy.last_indexes)
        return payload

    def render_energy_spectrum(
        self, key: str, energy_bins=None, scale: str = "log",
        fmt: str = "png", dpi: int = 150,
    ) -> bytes:
        """Render energy spectrum of a cached probability result."""
        import matplotlib.pyplot as plt
        result = self._cache[key]

        def _draw(fig, ax):
            plt.sca(ax)
            result.plot_energy_spectrum(energy_bins=energy_bins, scale=scale)

        return self._render_to_bytes(_draw, fmt, dpi)

    def render_backtrace_pair(
        self, key: str, var1: str, var2: str,
        fmt: str = "png", dpi: int = 150, **plot_kwargs,
    ) -> bytes:
        """Render a backtrace XY pair as a line plot."""
        result = self._cache[key]
        xy = result.pair(var1, var2)
        return self._render_to_bytes(
            lambda fig, ax: xy.plot(ax=ax, **plot_kwargs), fmt, dpi,
        )

    def render_field(
        self, attr_name: str, index: tuple,
        fmt: str = "png", dpi: int = 150, **plot_kwargs,
    ) -> bytes:
        """Render a sliced field (e.g. data.phisp[-1, :, 100, :]) as image bytes."""
        import matplotlib.pyplot as plt
        arr = getattr(self._data, attr_name)[index]

        def _draw(fig, ax):
            plt.sca(ax)
            arr.plot(**plot_kwargs)

        return self._render_to_bytes(_draw, fmt, dpi)

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
        """Render Data3d.plot_surfaces on the worker and return image bytes."""
        data3d = getattr(self._data, attr_name)[t_index]
        boundaries = self._data.boundaries

        def _draw(fig, ax):
            import matplotlib.pyplot as plt
            ax = fig.add_subplot(111, projection="3d")
            data3d.plot_surfaces(
                surfaces=boundaries, ax=ax, use_si=use_si, **plot_kwargs,
            )

        return self._render_to_bytes(_draw, fmt, dpi)

    def render_boundaries(
        self, use_si: bool = True, fmt: str = "png", dpi: int = 150,
        **plot_kwargs,
    ) -> bytes:
        """Render BoundaryCollection.plot() on the worker and return image bytes."""
        def _draw(fig, ax):
            ax = fig.add_subplot(111, projection="3d")
            self._data.boundaries.plot(ax=ax, use_si=use_si, **plot_kwargs)

        return self._render_to_bytes(_draw, fmt, dpi)

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

            elif kind == "backtrace_render_inline":
                # Pre-fetched foreign data — reconstruct and plot locally
                _, payload, plot_kwargs = cmd
                from emout.core.backtrace.probability_result import HeatmapData
                units = _rebuild_units(payload.get("units"))
                heatmap = HeatmapData(
                    X=payload["X"], Y=payload["Y"], Z=payload["Z"],
                    xlabel=payload["xlabel"], ylabel=payload["ylabel"],
                    title=payload["title"], units=units,
                )
                ax = plt.gca()
                heatmap.plot(ax=ax, **plot_kwargs)

            elif kind == "field_plot_inline":
                # Pre-fetched foreign field data
                _, payload, plot_kwargs = cmd
                from emout.core.data.data import Data1d, Data2d, Data3d, Data4d
                arr = payload["array"]
                cls_map = {1: Data1d, 2: Data2d, 3: Data3d, 4: Data4d}
                DataCls = cls_map.get(arr.ndim, Data2d)
                local_data = DataCls(
                    arr, name=payload["name"],
                    axisunits=payload["axisunits"],
                    valunit=payload["valunit"],
                )
                local_data.slices = payload["slices"]
                local_data.slice_axes = payload["slice_axes"]
                local_data._emout_dir = None
                local_data._emout_open_kwargs = None
                local_data.plot(**plot_kwargs)

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
    """``HeatmapData.plot()`` と同じインタフェースの proxy。

    ``remote_figure()`` 内ではコマンド記録、外ではデータ転送＋ローカル描画。
    """

    def __init__(self, session: RemoteSession, cache_key: str, var1: str, var2: str):
        self._session = session
        self._key = cache_key
        self._var1 = var1
        self._var2 = var2

    def fetch(self):
        """Fetch the heatmap data from the worker and return a local :class:`HeatmapData`.

        Returns
        -------
        HeatmapData
            A local copy that supports full matplotlib customisation.
        """
        from emout.core.backtrace.probability_result import HeatmapData
        from emout.utils.units import UnitTranslator

        payload = self._session.fetch_heatmap_data(
            self._key, self._var1, self._var2,
        ).result()
        units = None
        if payload["units"] is not None:
            units = [
                UnitTranslator(u["from_unit"], u["to_unit"],
                               name=u["name"], unit=u["unit"])
                for u in payload["units"]
            ]
        return HeatmapData(
            X=payload["X"], Y=payload["Y"], Z=payload["Z"],
            xlabel=payload["xlabel"], ylabel=payload["ylabel"],
            title=payload["title"], units=units,
        )

    def plot(self, ax=None, fmt: str = "png", dpi: int = 150, **plot_kwargs):
        """Plot the heatmap.

        Inside a ``remote_figure()`` context the call is recorded and
        replayed on the worker together with any ``plt.*`` calls.
        Outside, the rendering is executed on the worker and the
        resulting image is displayed.

        To get a local :class:`HeatmapData` for full matplotlib control,
        use :meth:`fetch` instead.
        """
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
    """``XYData.plot()`` / ``MultiXYData.plot()`` と同じインタフェースの proxy。

    ``remote_figure()`` 内ではコマンド記録、外ではデータ転送＋ローカル描画。
    """

    def __init__(self, session: RemoteSession, cache_key: str, var1: str, var2: str):
        self._session = session
        self._key = cache_key
        self._var1 = var1
        self._var2 = var2

    def fetch(self):
        """Fetch the XY data from the worker and return a local object.

        Returns
        -------
        XYData or MultiXYData
            A local copy that supports full matplotlib customisation.
        """
        from emout.core.backtrace.xy_data import MultiXYData, XYData
        from emout.utils.units import UnitTranslator

        payload = self._session.fetch_xy_data(
            self._key, self._var1, self._var2,
        ).result()
        units = None
        if payload["units"] is not None:
            units = [
                UnitTranslator(u["from_unit"], u["to_unit"],
                               name=u["name"], unit=u["unit"])
                for u in payload["units"]
            ]
        if "last_indexes" in payload:
            return MultiXYData(
                x=payload["x"], y=payload["y"],
                last_indexes=payload["last_indexes"],
                xlabel=payload["xlabel"], ylabel=payload["ylabel"],
                title=payload["title"], units=units,
            )
        return XYData(
            x=payload["x"], y=payload["y"],
            xlabel=payload["xlabel"], ylabel=payload["ylabel"],
            title=payload["title"], units=units,
        )

    def plot(self, ax=None, fmt: str = "png", dpi: int = 150, **plot_kwargs):
        """Plot the XY data.

        Inside a ``remote_figure()`` context the call is recorded and
        replayed on the worker together with any ``plt.*`` calls.
        Outside, the rendering is executed on the worker and the
        resulting image is displayed.

        To get a local :class:`XYData` / :class:`MultiXYData` for full
        matplotlib control, use :meth:`fetch` instead.
        """
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
