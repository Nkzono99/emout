"""Dask Actor infrastructure that renders visualisations on workers and returns only PNG bytes.

Design
------
- Heavy computations (backtrace, etc.) run once on the worker and stay in worker memory.
- Visualisation parameters (cmap, vmin, vmax, projection axis, etc.) can be changed freely for re-rendering.
- Only PNG/SVG bytes (tens of KB) are transferred to the client.
- Users interact via the same interface as local objects, e.g. ``result.vxvz.plot()``.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

_key_counter = itertools.count()
_REMOTE_REF_MARKER = "__remote_ref__"
_scope_stack: list["RemoteScope"] = []


def _next_key(prefix: str = "result") -> str:
    return f"{prefix}_{next(_key_counter)}"


def _await_remote(result):
    if hasattr(result, "result") and callable(result.result):
        return result.result()
    return result


def _register_remote_key(session, key: str) -> None:
    if _scope_stack:
        _scope_stack[-1]._register(session, key)


# ---------------------------------------------------------------------------
# Worker-side Actor
# ---------------------------------------------------------------------------


class RemoteSession:
    """Shared Dask Actor that manages multiple Emout instances on one worker.

    A single ``RemoteSession`` lazily loads :class:`Emout` instances as they
    are referenced by ``emout_kwargs``, so results from different simulations
    can coexist in the same ``remote_figure()`` block.

    Create with ``client.submit(RemoteSession, actor=True)``.
    """

    def __init__(
        self,
        emout_dir: str | None = None,
        input_path: str | None = None,
        emout_kwargs: dict[str, Any] | None = None,
    ):
        import matplotlib

        matplotlib.use("Agg")

        self._instances: dict[str, Any] = {}  # JSON key → Emout
        self._cache: dict[str, Any] = {}

        # Backward compat: eagerly load if kwargs provided
        if emout_dir is not None or emout_kwargs is not None:
            normalized = _normalize_emout_kwargs(
                emout_dir=emout_dir,
                input_path=input_path,
                emout_kwargs=emout_kwargs,
            )
            self._get_emout(normalized)

    def _get_emout(self, emout_kwargs: dict):
        """Lazy-load and cache an Emout instance by its normalized kwargs."""
        import json as _json

        cache_key = _json.dumps(emout_kwargs, sort_keys=True)
        if cache_key not in self._instances:
            import emout

            self._instances[cache_key] = emout.Emout(**emout_kwargs)
        return self._instances[cache_key]

    def _resolve(self, emout_kwargs=None):
        """Return the Emout for *emout_kwargs*, or the first loaded instance."""
        if emout_kwargs is not None:
            return self._get_emout(emout_kwargs)
        if not self._instances:
            raise RuntimeError("No Emout instances loaded in this session")
        return next(iter(self._instances.values()))

    @property
    def _data(self):
        """Backward-compat property: return the first loaded Emout."""
        return self._resolve()

    # -- computation (result stays on worker) --------------------------------

    def compute_probabilities(self, key: str, emout_kwargs=None, **kwargs) -> bool:
        """Run backtrace probability calculation and cache the result.

        Parameters
        ----------
        key : str
            Cache key under which the result is stored.
        emout_kwargs : dict, optional
            Identifies which Emout instance to use.
        **kwargs
            Keyword arguments forwarded to
            ``Emout.backtrace.get_probabilities()``.

        Returns
        -------
        bool
            ``True`` on success.
        """
        data = self._resolve(emout_kwargs)
        result = data.backtrace.get_probabilities(**kwargs)
        self._cache[key] = result
        return True

    def compute_backtraces(self, key: str, emout_kwargs=None, **kwargs) -> bool:
        """Run particle backtrace calculation and cache the result.

        Parameters
        ----------
        key : str
            Cache key under which the result is stored.
        emout_kwargs : dict, optional
            Identifies which Emout instance to use.
        **kwargs
            Keyword arguments forwarded to
            ``Emout.backtrace.get_backtraces_from_particles()``.

        Returns
        -------
        bool
            ``True`` on success.
        """
        data = self._resolve(emout_kwargs)
        result = data.backtrace.get_backtraces_from_particles(**kwargs)
        self._cache[key] = result
        return True

    def _decode_remote_value(self, value):
        if isinstance(value, dict):
            if set(value) == {_REMOTE_REF_MARKER}:
                return self._cache[value[_REMOTE_REF_MARKER]]
            return {k: self._decode_remote_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._decode_remote_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._decode_remote_value(v) for v in value)
        return value

    def cache_emout_attr(self, key: str, emout_kwargs: dict[str, Any], name: str) -> bool:
        """Cache an attribute resolved from an Emout instance."""
        self._cache[key] = getattr(self._resolve(emout_kwargs), name)
        return True

    def cache_attr(self, key: str, parent_key: str, name: str) -> bool:
        """Cache ``getattr(parent, name)``."""
        self._cache[key] = getattr(self._cache[parent_key], name)
        return True

    def cache_getitem(self, key: str, parent_key: str, index) -> bool:
        """Cache ``parent[index]``."""
        self._cache[key] = self._cache[parent_key][self._decode_remote_value(index)]
        return True

    def call_cached(self, key: str, parent_key: str, args=(), kwargs=None) -> bool:
        """Call a cached callable and store the return value."""
        kwargs = {} if kwargs is None else kwargs
        obj = self._cache[parent_key]
        resolved_args = self._decode_remote_value(args)
        resolved_kwargs = self._decode_remote_value(kwargs)
        self._cache[key] = obj(*resolved_args, **resolved_kwargs)
        return True

    def call_method(self, key: str, parent_key: str, method_name: str, args=(), kwargs=None) -> bool:
        """Call ``parent.method_name(*args, **kwargs)`` and cache the result."""
        kwargs = {} if kwargs is None else kwargs
        obj = self._cache[parent_key]
        resolved_args = self._decode_remote_value(args)
        resolved_kwargs = self._decode_remote_value(kwargs)
        method = getattr(obj, method_name)
        self._cache[key] = method(*resolved_args, **resolved_kwargs)
        return True

    def apply_function(self, key: str, parent_key: str, func, args=(), kwargs=None) -> bool:
        """Apply ``func(parent, *args, **kwargs)`` and cache the result."""
        kwargs = {} if kwargs is None else kwargs
        obj = self._cache[parent_key]
        resolved_args = self._decode_remote_value(args)
        resolved_kwargs = self._decode_remote_value(kwargs)
        self._cache[key] = func(obj, *resolved_args, **resolved_kwargs)
        return True

    def fetch_object(self, key: str):
        """Return a cached object as-is."""
        return self._cache[key]

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
        self,
        key: str,
        var1: str,
        var2: str,
        fmt: str = "png",
        dpi: int = 150,
        **plot_kwargs,
    ) -> bytes:
        """Render a 2-D heatmap of a cached probability result and return image bytes."""
        result = self._cache[key]
        heatmap = result.pair(var1, var2)
        return self._render_to_bytes(
            lambda fig, ax: heatmap.plot(ax=ax, **plot_kwargs),
            fmt,
            dpi,
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
                {"from_unit": u.from_unit, "to_unit": u.to_unit, "name": u.name, "unit": u.unit} for u in heatmap.units
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
                {"from_unit": u.from_unit, "to_unit": u.to_unit, "name": u.name, "unit": u.unit} for u in xy.units
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
        self,
        key: str,
        energy_bins=None,
        scale: str = "log",
        fmt: str = "png",
        dpi: int = 150,
    ) -> bytes:
        """Render energy spectrum of a cached probability result."""
        import matplotlib.pyplot as plt

        result = self._cache[key]

        def _draw(fig, ax):
            plt.sca(ax)
            result.plot_energy_spectrum(energy_bins=energy_bins, scale=scale)

        return self._render_to_bytes(_draw, fmt, dpi)

    def render_backtrace_pair(
        self,
        key: str,
        var1: str,
        var2: str,
        fmt: str = "png",
        dpi: int = 150,
        **plot_kwargs,
    ) -> bytes:
        """Render a backtrace XY pair as a line plot."""
        result = self._cache[key]
        xy = result.pair(var1, var2)
        return self._render_to_bytes(
            lambda fig, ax: xy.plot(ax=ax, **plot_kwargs),
            fmt,
            dpi,
        )

    def render_field(
        self,
        attr_name: str,
        index: tuple,
        fmt: str = "png",
        dpi: int = 150,
        emout_kwargs=None,
        **plot_kwargs,
    ) -> bytes:
        """Render a sliced field (e.g. data.phisp[-1, :, 100, :]) as image bytes."""
        import matplotlib.pyplot as plt

        data = self._resolve(emout_kwargs)
        arr = getattr(data, attr_name)[index]

        def _draw(fig, ax):
            plt.sca(ax)
            arr.plot(**plot_kwargs)

        return self._render_to_bytes(_draw, fmt, dpi)

    def render_cached_plot(
        self,
        key: str,
        fmt: str = "png",
        dpi: int = 150,
        **plot_kwargs,
    ) -> bytes:
        """Render ``cached_object.plot(...)`` on the worker and return image bytes."""
        import matplotlib.pyplot as plt

        obj = self._cache[key]

        def _draw(fig, ax):
            plt.sca(ax)
            obj.plot(**plot_kwargs)

        return self._render_to_bytes(_draw, fmt, dpi)

    def fetch_field(self, attr_name: str, index: tuple, emout_kwargs=None) -> dict:
        """Return sliced field data and metadata for local plotting.

        Only the sliced (2-D, a few KB to a few MB) array is transferred
        instead of the full 3-D volume, enabling local overlay operations
        such as ``plt.axhline()``.
        """
        data = self._resolve(emout_kwargs)
        arr = getattr(data, attr_name)[index]
        return {
            "array": np.asarray(arr),
            "name": arr.name,
            "slices": arr.slices,
            "slice_axes": arr.slice_axes,
            "axisunits": arr.axisunits,
            "valunit": arr.valunit,
        }

    def render_plot_surfaces(
        self,
        attr_name: str,
        t_index: int,
        use_si: bool = True,
        fmt: str = "png",
        dpi: int = 150,
        emout_kwargs=None,
        **plot_kwargs,
    ) -> bytes:
        """Render Data3d.plot_surfaces on the worker and return image bytes."""
        data = self._resolve(emout_kwargs)
        data3d = getattr(data, attr_name)[t_index]
        boundaries = data.boundaries

        def _draw(fig, ax):
            ax = fig.add_subplot(111, projection="3d")
            data3d._plot_surfaces_local(
                surfaces=boundaries,
                ax=ax,
                use_si=use_si,
                **plot_kwargs,
            )

        return self._render_to_bytes(_draw, fmt, dpi)

    def render_boundaries(
        self,
        use_si: bool = True,
        fmt: str = "png",
        dpi: int = 150,
        emout_kwargs=None,
        **plot_kwargs,
    ) -> bytes:
        """Render BoundaryCollection.plot() on the worker and return image bytes."""
        data = self._resolve(emout_kwargs)

        def _draw(fig, ax):
            ax = fig.add_subplot(111, projection="3d")
            data.boundaries.plot(ax=ax, use_si=use_si, **plot_kwargs)

        return self._render_to_bytes(_draw, fmt, dpi)

    def replay_figure(self, commands: list, fmt: str = "png", dpi: int = 150) -> bytes:
        """Replay a command list and return the rendered image bytes.

        Command formats::

            ("field_plot", attr_name, recipe_index, plot_kwargs, emout_kwargs)
            ("plot_surfaces", attr_name, recipe_index, surfaces, plot_kwargs, emout_kwargs)
            ("cached_plot", key, plot_kwargs)
            ("figure_call", figure_id, method_name, args, kwargs, target)
            ("axes_call", ax_id, method_name, args, kwargs)
            ("axis_call", ax_id, axis_name, method_name, args, kwargs)
            ("spine_call", ax_id, spine_name, method_name, args, kwargs)
            ("colorbar_call", colorbar_id, method_name, args, kwargs)
            ("proxy_setattr", kind, proxy_id, name, value)
            ("plt", method_name, args, kwargs)
            ("boundary_plot", plot_kwargs, emout_kwargs)
            ("backtrace_render", cache_key, var1, var2, plot_kwargs)
            ("energy_spectrum", cache_key, spec_kwargs)
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO

        figures: dict[str, Any] = {}
        axes: dict[str, Any] = {}
        colorbars: dict[str, Any] = {}
        marker = "__remote_plot_proxy__"

        def _resolve_proxy(payload):
            kind = payload["kind"]
            proxy_id = payload["id"]
            if kind == "figure":
                return figures[proxy_id]
            if kind == "axes":
                return axes[proxy_id]
            if kind == "colorbar":
                return colorbars[proxy_id]
            if kind == "axis":
                return getattr(axes[proxy_id], payload["axis_name"])
            if kind == "spine":
                return axes[proxy_id].spines[payload["spine_name"]]
            raise KeyError(f"Unknown remote plot proxy kind: {kind}")

        def _decode(value):
            if isinstance(value, dict):
                if marker in value:
                    return _resolve_proxy(value[marker])
                return {k: _decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_decode(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_decode(v) for v in value)
            return value

        def _bind_single(target, result):
            payload = target[marker]
            kind = payload["kind"]
            proxy_id = payload["id"]
            if kind == "figure":
                figures[proxy_id] = result
            elif kind == "axes":
                axes[proxy_id] = result
            elif kind == "colorbar":
                colorbars[proxy_id] = result

        def _bind_target(target, result):
            if target is None:
                return
            if isinstance(target, dict) and marker in target:
                _bind_single(target, result)
                return
            if not isinstance(target, dict):
                return

            if "figure" in target:
                fig_obj = result[0] if isinstance(result, tuple) else getattr(result, "figure", result)
                _bind_single(target["figure"], fig_obj)
            if "axes" in target:
                ax_obj = result[1] if isinstance(result, tuple) else result
                _bind_single(target["axes"], ax_obj)
            if "axes_grid" in target:
                axes_result = result[1] if isinstance(result, tuple) else result
                target_grid = target["axes_grid"]
                nrows = len(target_grid)
                ncols = len(target_grid[0]) if nrows else 0
                axes_arr = np.array(axes_result, dtype=object).reshape(nrows, ncols)
                for row_index, row in enumerate(target_grid):
                    for col_index, proxy_target in enumerate(row):
                        _bind_single(proxy_target, axes_arr[row_index, col_index])
            if "colorbar" in target:
                _bind_single(target["colorbar"], result)
            if "cax" in target:
                _bind_single(target["cax"], result.ax)

        for cmd in commands:
            kind = cmd[0]
            if kind == "field_plot":
                _, attr_name, recipe_index, plot_kwargs, emout_kwargs = cmd
                plot_kwargs = _decode(plot_kwargs)
                emout_kwargs = _decode(emout_kwargs)
                data = self._resolve(emout_kwargs)
                arr = getattr(data, attr_name)[recipe_index]
                arr.plot(**plot_kwargs)

            elif kind == "plot_surfaces":
                _, attr_name, recipe_index, surfaces, plot_kwargs, emout_kwargs = cmd
                surfaces = _decode(surfaces)
                plot_kwargs = _decode(plot_kwargs)
                emout_kwargs = _decode(emout_kwargs)
                data = self._resolve(emout_kwargs)
                arr = getattr(data, attr_name)[recipe_index]
                arr._plot_surfaces_local(surfaces, **plot_kwargs)

            elif kind == "cached_plot":
                _, key, plot_kwargs = cmd
                plot_kwargs = _decode(plot_kwargs)
                obj = self._cache[key]
                obj.plot(**plot_kwargs)

            elif kind == "figure_call":
                _, figure_id, method_name, args, kwargs, target = cmd
                args = _decode(args)
                kwargs = _decode(kwargs)
                if figure_id is None:
                    func = getattr(plt, method_name)
                    result = func(*args, **kwargs)
                else:
                    fig = figures[figure_id]
                    result = getattr(fig, method_name)(*args, **kwargs)
                _bind_target(target, result)

            elif kind == "axes_call":
                _, ax_id, method_name, args, kwargs = cmd
                args = _decode(args)
                kwargs = _decode(kwargs)
                getattr(axes[ax_id], method_name)(*args, **kwargs)

            elif kind == "axis_call":
                _, ax_id, axis_name, method_name, args, kwargs = cmd
                args = _decode(args)
                kwargs = _decode(kwargs)
                axis = getattr(axes[ax_id], axis_name)
                getattr(axis, method_name)(*args, **kwargs)

            elif kind == "spine_call":
                _, ax_id, spine_name, method_name, args, kwargs = cmd
                args = _decode(args)
                kwargs = _decode(kwargs)
                spine = axes[ax_id].spines[spine_name]
                getattr(spine, method_name)(*args, **kwargs)

            elif kind == "colorbar_call":
                _, colorbar_id, method_name, args, kwargs = cmd
                args = _decode(args)
                kwargs = _decode(kwargs)
                getattr(colorbars[colorbar_id], method_name)(*args, **kwargs)

            elif kind == "proxy_setattr":
                _, proxy_kind, proxy_id, name, value = cmd
                target = _resolve_proxy({"kind": proxy_kind, "id": proxy_id})
                setattr(target, name, _decode(value))

            elif kind == "plt":
                _, method_name, args, kwargs = cmd
                args = _decode(args)
                kwargs = _decode(kwargs)
                func = getattr(plt, method_name)
                func(*args, **kwargs)

            elif kind == "boundary_plot":
                _, plot_kwargs, emout_kwargs = cmd
                plot_kwargs = _decode(plot_kwargs)
                emout_kwargs = _decode(emout_kwargs)
                data = self._resolve(emout_kwargs)
                if "ax" not in plot_kwargs:
                    plot_kwargs = {**plot_kwargs, "ax": plt.gca()}
                data.boundaries.plot(**plot_kwargs)

            elif kind == "backtrace_render":
                _, cache_key, var1, var2, plot_kwargs = cmd
                plot_kwargs = _decode(plot_kwargs)
                result = self._cache[cache_key]
                heatmap = result.pair(var1, var2)
                if "ax" not in plot_kwargs:
                    plot_kwargs = {**plot_kwargs, "ax": plt.gca()}
                heatmap.plot(**plot_kwargs)

            elif kind == "energy_spectrum":
                _, cache_key, spec_kwargs = cmd
                spec_kwargs = _decode(spec_kwargs)
                result = self._cache[cache_key]
                result.plot_energy_spectrum(**spec_kwargs)

        buf = BytesIO()
        plt.gcf().savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close("all")
        return buf.getvalue()

    def drop(self, key: str) -> None:
        """Remove a cached result and free the associated memory."""
        self._cache.pop(key, None)

    def keys(self) -> list[str]:
        return list(self._cache.keys())


# ---------------------------------------------------------------------------
# Remote object scope / generic proxies
# ---------------------------------------------------------------------------


class RemoteScope:
    """Context manager that auto-releases remote cache entries on exit."""

    def __init__(self):
        self._entries: list[tuple[Any, str]] = []

    def _register(self, session, key: str) -> None:
        self._entries.append((session, key))

    def __enter__(self) -> "RemoteScope":
        _scope_stack.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if _scope_stack and _scope_stack[-1] is self:
            _scope_stack.pop()
        elif self in _scope_stack:
            _scope_stack.remove(self)

        seen: set[tuple[int, str]] = set()
        for session, key in reversed(self._entries):
            ident = (id(session), key)
            if ident in seen:
                continue
            seen.add(ident)
            try:
                _await_remote(session.drop(key))
            except Exception as err:  # pragma: no cover - defensive cleanup path
                warnings.warn(
                    f"Failed to drop remote object {key!r}: {err}",
                    ResourceWarning,
                    stacklevel=2,
                )


@contextlib.contextmanager
def remote_scope():
    """Auto-drop remote objects created inside the ``with`` block."""
    scope = RemoteScope()
    with scope:
        yield scope


class RemoteEmout:
    """Client-side proxy that resolves Emout attributes on the worker."""

    def __init__(self, session: RemoteSession, emout_kwargs: dict[str, Any]):
        self._session = session
        self._emout_kwargs = dict(emout_kwargs)

    def __getattr__(self, name: str):
        key = _next_key("ref")
        _await_remote(self._session.cache_emout_attr(key, self._emout_kwargs, name))
        return RemoteRef(self._session, key)

    def __repr__(self):
        directory = self._emout_kwargs.get("output_directory", self._emout_kwargs.get("directory"))
        return f"<RemoteEmout directory={directory!r}>"


class RemoteRef:
    """Client-side proxy for an object cached inside :class:`RemoteSession`."""

    def __init__(self, session: RemoteSession, key: str):
        self._session = session
        self._key = key
        _register_remote_key(session, key)

    def _encode_remote_arg(self, value):
        if isinstance(value, RemoteRef):
            if value._session is not self._session:
                raise ValueError("RemoteRef arguments must belong to the same session")
            return {_REMOTE_REF_MARKER: value._key}
        if isinstance(value, list):
            return [self._encode_remote_arg(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._encode_remote_arg(v) for v in value)
        if isinstance(value, dict):
            return {k: self._encode_remote_arg(v) for k, v in value.items()}
        return value

    def _spawn(self, prefix: str) -> str:
        return _next_key(prefix)

    def fetch(self):
        """Fetch the cached object to the local process."""
        return _await_remote(self._session.fetch_object(self._key))

    def drop(self) -> None:
        """Release the cached object from worker memory."""
        _await_remote(self._session.drop(self._key))

    def call(self, method_name: str, *args, **kwargs) -> "RemoteRef":
        """Call a method on the cached object and return a new remote proxy."""
        key = self._spawn("ref")
        _await_remote(
            self._session.call_method(
                key,
                self._key,
                method_name,
                args=self._encode_remote_arg(args),
                kwargs=self._encode_remote_arg(kwargs),
            )
        )
        return RemoteRef(self._session, key)

    def apply(self, func, *args, **kwargs) -> "RemoteRef":
        """Apply ``func(self, *args, **kwargs)`` on the worker."""
        key = self._spawn("ref")
        _await_remote(
            self._session.apply_function(
                key,
                self._key,
                func,
                args=self._encode_remote_arg(args),
                kwargs=self._encode_remote_arg(kwargs),
            )
        )
        return RemoteRef(self._session, key)

    def plot(self, ax=None, fmt: str = "png", dpi: int = 150, **plot_kwargs):
        """Plot the cached object remotely when it exposes ``plot()``."""
        from .remote_figure import bind_session, is_recording, record_cached_plot

        if is_recording():
            bind_session(self._session)
            record_cached_plot(self._key, plot_kwargs)
            return None
        img = _await_remote(self._session.render_cached_plot(self._key, fmt=fmt, dpi=dpi, **plot_kwargs))
        return display_image(img, ax=ax)

    def __getitem__(self, index) -> "RemoteRef":
        key = self._spawn("ref")
        _await_remote(self._session.cache_getitem(key, self._key, self._encode_remote_arg(index)))
        return RemoteRef(self._session, key)

    def __getattr__(self, name: str) -> "RemoteRef":
        if name.startswith("__"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        key = self._spawn("ref")
        _await_remote(self._session.cache_attr(key, self._key, name))
        return RemoteRef(self._session, key)

    def __call__(self, *args, **kwargs) -> "RemoteRef":
        key = self._spawn("ref")
        _await_remote(
            self._session.call_cached(
                key,
                self._key,
                args=self._encode_remote_arg(args),
                kwargs=self._encode_remote_arg(kwargs),
            )
        )
        return RemoteRef(self._session, key)

    def __repr__(self):
        return f"<RemoteRef key={self._key!r}>"


# ---------------------------------------------------------------------------
# Client-side proxies (same interface as real result objects)
# ---------------------------------------------------------------------------


class RemoteHeatmap:
    """Proxy with the same interface as ``HeatmapData.plot()``.

    Inside ``remote_figure()``, calls are recorded as commands; outside,
    data is transferred and drawn locally.
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
            self._key,
            self._var1,
            self._var2,
        ).result()
        units = None
        if payload["units"] is not None:
            units = [
                UnitTranslator(u["from_unit"], u["to_unit"], name=u["name"], unit=u["unit"]) for u in payload["units"]
            ]
        return HeatmapData(
            X=payload["X"],
            Y=payload["Y"],
            Z=payload["Z"],
            xlabel=payload["xlabel"],
            ylabel=payload["ylabel"],
            title=payload["title"],
            units=units,
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
            self._key,
            self._var1,
            self._var2,
            fmt=fmt,
            dpi=dpi,
            **plot_kwargs,
        ).result()
        return display_image(img, ax=ax)

    def __repr__(self):
        return f"<RemoteHeatmap: {self._var1} vs {self._var2} (key={self._key!r})>"


class RemoteXYData:
    """Proxy with the same interface as ``XYData.plot()`` / ``MultiXYData.plot()``.

    Inside ``remote_figure()``, calls are recorded as commands; outside,
    data is transferred and drawn locally.
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
            self._key,
            self._var1,
            self._var2,
        ).result()
        units = None
        if payload["units"] is not None:
            units = [
                UnitTranslator(u["from_unit"], u["to_unit"], name=u["name"], unit=u["unit"]) for u in payload["units"]
            ]
        if "last_indexes" in payload:
            return MultiXYData(
                x=payload["x"],
                y=payload["y"],
                last_indexes=payload["last_indexes"],
                xlabel=payload["xlabel"],
                ylabel=payload["ylabel"],
                title=payload["title"],
                units=units,
            )
        return XYData(
            x=payload["x"],
            y=payload["y"],
            xlabel=payload["xlabel"],
            ylabel=payload["ylabel"],
            title=payload["title"],
            units=units,
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
            self._key,
            self._var1,
            self._var2,
            fmt=fmt,
            dpi=dpi,
            **plot_kwargs,
        ).result()
        return display_image(img, ax=ax)

    def __repr__(self):
        return f"<RemoteXYData: {self._var1} vs {self._var2} (key={self._key!r})>"


class RemoteProbabilityResult:
    """Proxy with the same interface as ``ProbabilityResult``.

    Use it exactly like the local version, e.g.
    ``result.vxvz.plot(cmap="plasma")``.  The actual rendering runs on
    the worker and only PNG bytes are returned.
    """

    _AXES = ["x", "y", "z", "vx", "vy", "vz"]

    def __init__(self, session: RemoteSession, cache_key: str):
        self._session = session
        self._key = cache_key
        _register_remote_key(session, cache_key)

    def pair(self, var1: str, var2: str) -> RemoteHeatmap:
        return RemoteHeatmap(self._session, self._key, var1, var2)

    def plot_energy_spectrum(
        self,
        energy_bins=None,
        scale: str = "log",
        fmt: str = "png",
        dpi: int = 150,
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
            self._key,
            energy_bins=energy_bins,
            scale=scale,
            fmt=fmt,
            dpi=dpi,
        ).result()
        return display_image(img)

    def drop(self) -> None:
        """Release the result from worker memory."""
        self._session.drop(self._key)

    def __getattr__(self, name: str):
        # Same as ProbabilityResult.__getattr__: result.vxvz -> pair("vx","vz")
        for key1 in self._AXES:
            if name.startswith(key1):
                rest = name[len(key1) :]
                if rest in self._AXES and rest != key1:
                    return self.pair(key1, rest)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self):
        return f"<RemoteProbabilityResult (key={self._key!r})>"


class RemoteBacktraceResult:
    """Proxy with the same interface as ``BacktraceResult`` / ``MultiBacktraceResult``."""

    _AXES = ["x", "y", "z", "vx", "vy", "vz"]

    def __init__(self, session: RemoteSession, cache_key: str):
        self._session = session
        self._key = cache_key
        _register_remote_key(session, cache_key)

    def pair(self, var1: str, var2: str) -> RemoteXYData:
        return RemoteXYData(self._session, self._key, var1, var2)

    def drop(self) -> None:
        self._session.drop(self._key)

    def __getattr__(self, name: str):
        for key1 in self._AXES:
            if name.startswith(key1):
                rest = name[len(key1) :]
                if rest in self._AXES and rest != key1:
                    return self.pair(key1, rest)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __repr__(self):
        return f"<RemoteBacktraceResult (key={self._key!r})>"


# ---------------------------------------------------------------------------
# Image display helper
# ---------------------------------------------------------------------------


def display_image(img_bytes: bytes, ax=None):
    """Display PNG bytes in Jupyter or draw onto a matplotlib axes."""
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

_shared_session: Optional[RemoteSession] = None


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
        normalized["append_directories"] = [str(Path(path).resolve()) for path in append_directories]

    return normalized


def get_or_create_session(
    emout_dir=None,
    input_path=None,
    emout_kwargs: dict[str, Any] | None = None,
) -> Optional[RemoteSession]:
    """Return the shared RemoteSession actor, creating it if needed.

    A single shared session manages all Emout instances lazily.
    ``emout_kwargs`` is accepted for API compatibility but is no longer
    used as a cache key — every call returns the same session.

    Returns ``None`` when no Dask client is available.
    """
    global _shared_session

    if sys.version_info < (3, 10):
        return None

    try:
        from dask.distributed import default_client
    except ImportError:
        return None

    try:
        client = default_client()
    except ValueError:
        client = None

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

    if _shared_session is None:
        future = client.submit(RemoteSession, actor=True)
        _shared_session = future.result()

    return _shared_session


def clear_sessions() -> None:
    """Clear the shared session."""
    global _shared_session
    _shared_session = None
