"""Integrated trace workflow result containers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from emout.plot._pyvista_helpers import _offseted, _require_pyvista


class TraceResult:
    """Integrated result returned by the high-level ``data.trace`` API.

    The object always has the same shape regardless of which payloads were
    requested. Probability and trajectory payloads that were not requested are
    stored as ``None``.
    """

    def __init__(
        self,
        direction: str,
        probabilities: Any = None,
        backward_traces: Any = None,
        forward_traces: Any = None,
        phases: Optional[np.ndarray] = None,
        dims: Optional[tuple[int, int, int, int, int, int]] = None,
        particles: Any = None,
        unit: Any = None,
    ):
        self.direction = direction
        self.probabilities = probabilities
        self.backward_traces = backward_traces
        self.forward_traces = forward_traces
        self.phases = phases
        self.dims = dims
        self.particles = particles
        self.unit = unit

    def __repr__(self) -> str:
        payloads = []
        if self.probabilities is not None:
            payloads.append("probabilities")
        if self.backward_traces is not None:
            payloads.append("backward_traces")
        if self.forward_traces is not None:
            payloads.append("forward_traces")
        payload_text = ", ".join(payloads) if payloads else "empty"
        return f"<TraceResult: direction={self.direction!r}, payloads={payload_text}>"

    @property
    def traces(self):
        """Return the unambiguous trajectory payload.

        ``both()`` results contain two trajectory payloads, so callers must use
        ``backward_traces`` or ``forward_traces`` explicitly.
        """
        has_backward = self.backward_traces is not None
        has_forward = self.forward_traces is not None
        if has_backward and has_forward:
            raise ValueError("TraceResult has both backward and forward traces; choose one explicitly")
        if has_backward:
            return self.backward_traces
        if has_forward:
            return self.forward_traces
        return None

    @property
    def alpha(self):
        """Return probability-derived alpha values, or ``None`` if unavailable."""
        if self.probabilities is None:
            return None
        values = np.asarray(self.probabilities.probabilities, dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(values, 0.0, 1.0)

    def pair(self, var1: str, var2: str):
        """Project the probability result onto a two-axis plane."""
        if self.probabilities is None:
            raise ValueError("probabilities are not available")
        return self.probabilities.pair(var1, var2)

    def plot_probabilities(self, var1: str = "vx", var2: str = "vz", **plot_kwargs):
        """Plot the probability projection for ``var1`` and ``var2``."""
        return self.pair(var1, var2).plot(**plot_kwargs)

    def plot(
        self,
        var1: str = "vx",
        var2: str = "vz",
        kind: str = "auto",
        direction: Optional[str] = None,
        **plot_kwargs,
    ):
        """Plot probabilities when available, otherwise plot trajectories."""
        if kind == "auto":
            kind = "probability" if self.probabilities is not None else "trace"
        if kind in {"probability", "probabilities"}:
            return self.plot_probabilities(var1, var2, **plot_kwargs)
        if kind in {"trace", "traces"}:
            return self.plot_traces(var1, var2, direction=direction, **plot_kwargs)
        raise ValueError("kind must be 'auto', 'probability', or 'trace'")

    def plot_traces(
        self,
        var1: str = "x",
        var2: str = "z",
        direction: Optional[str] = None,
        alpha: Any = "auto",
        **plot_kwargs,
    ):
        """Plot trajectory pairs with optional probability-derived alpha.

        When both backward and forward payloads are available, omitting
        ``direction`` overlays both payloads on the same axes.
        """
        trace_payloads = self._select_trace_payloads(direction)
        if not trace_payloads:
            raise ValueError("traces are not available")

        resolved_alpha = self._resolve_alpha(alpha)
        if resolved_alpha is not None:
            plot_kwargs = {**plot_kwargs, "alpha": resolved_alpha}

        ax = plot_kwargs.get("ax")
        for traces in trace_payloads:
            current_kwargs = dict(plot_kwargs)
            if ax is not None:
                current_kwargs["ax"] = ax
            ax = traces.pair(var1, var2).plot(**current_kwargs)
        return ax

    def plot3d(
        self,
        direction: Optional[str] = None,
        plotter: Any = None,
        use_si: bool = True,
        offsets=None,
        show: bool = False,
        alpha: Any = "auto",
        color: str = "white",
        line_width: float = 2.0,
        tube_radius: Optional[float] = None,
        *,
        min_alpha: float = 0.0,
        alpha_scale: str = "linear",
        clip_bounds=None,
        backward_color=None,
        forward_color=None,
        show_bounds=None,
        bounds_kwargs=None,
        camera=None,
        camera_position=None,
        **mesh_kwargs,
    ):
        """Draw trajectories on a PyVista plotter and return it.

        Passing an existing ``plotter`` overlays the trajectories on the
        caller's current 3-D field or boundary view. When both backward and
        forward payloads are available, omitting ``direction`` draws both.
        ``direction='both'`` is an explicit alias. ``clip_bounds`` is an
        (xmin, xmax, ymin, ymax, zmin, zmax) box in displayed coordinates
        after SI conversion and offsets. Segments are clipped at the box
        faces without connecting disjoint pieces. ``alpha_scale`` accepts
        'linear', 'normalize' (divide by the maximum), or 'log' (normalize
        the logarithms of positive values). ``min_alpha`` sets a floor.
        """
        trace_payloads = self._select_trace_payloads(direction)
        if not trace_payloads:
            raise ValueError("traces are not available")

        pv = _require_pyvista()
        if plotter is None:
            plotter = pv.Plotter()

        alphas = self._scaled_alpha(alpha, min_alpha=min_alpha, alpha_scale=alpha_scale)
        if clip_bounds is not None:
            clip_bounds = np.asarray(clip_bounds, dtype=float)
            if clip_bounds.shape != (6,) or np.isnan(clip_bounds).any():
                raise ValueError("clip_bounds must contain six non-NaN bounds in (x, y, z) order")
            if np.any(clip_bounds[::2] > clip_bounds[1::2]):
                raise ValueError("each clip_bounds minimum must be <= its maximum")
        if offsets is None:
            offsets = (None, None, None)

        for traces in trace_payloads:
            positions = np.asarray(traces.positions_list, dtype=float)
            last_indexes = np.asarray(traces.last_indexes, dtype=int)
            unit = getattr(traces, "unit", None) or self.unit

            for index in range(positions.shape[0]):
                end = int(last_indexes[index])
                points = np.array(positions[index, :end, :], dtype=float, copy=True)
                if len(points) < 2:
                    continue
                if unit is not None and use_si:
                    points = unit.length.reverse(points)
                for axis in range(3):
                    points[:, axis] = _offseted(points[:, axis], offsets[axis])

                opacity = 1.0
                if alphas is not None:
                    if hasattr(alphas, "__len__"):
                        opacity = float(alphas[index])
                    else:
                        opacity = float(alphas)

                add_mesh_kwargs = {
                    "color": (
                        backward_color
                        if traces is self.backward_traces and backward_color is not None
                        else forward_color
                        if traces is self.forward_traces and forward_color is not None
                        else color
                    ),
                    "line_width": line_width,
                    "opacity": opacity,
                }
                add_mesh_kwargs.update(mesh_kwargs)
                pieces = [points] if clip_bounds is None else _clip_polyline(points, clip_bounds)
                for piece in pieces:
                    line = pv.PolyData(piece)
                    line.lines = np.concatenate(([len(piece)], np.arange(len(piece), dtype=int)))
                    mesh = line.tube(radius=tube_radius) if tube_radius is not None else line
                    plotter.add_mesh(mesh, **add_mesh_kwargs)

        from emout.plot._pyvista_helpers import _configure_plotter

        unit = self.unit
        if unit is None and trace_payloads:
            unit = getattr(trace_payloads[0], "unit", None)
        suffix = f" [{unit.length.unit}]" if use_si and unit is not None else ""
        _configure_plotter(
            plotter,
            {axis: f"{axis}{suffix}" for axis in ("x", "y", "z")},
            show_bounds=show_bounds,
            bounds_kwargs=bounds_kwargs,
            camera=camera,
            camera_position=camera_position,
        )
        plotter.add_axes()
        if show:
            plotter.show()
        return plotter

    def _select_traces(self, direction: Optional[str]):
        if direction is None:
            return self.traces
        if direction == "backward":
            return self.backward_traces
        if direction == "forward":
            return self.forward_traces
        raise ValueError("direction must be 'backward', 'forward', or None")

    def _select_trace_payloads(self, direction: Optional[str]):
        if direction in (None, "both"):
            return tuple(traces for traces in (self.backward_traces, self.forward_traces) if traces is not None)
        traces = self._select_traces(direction)
        return () if traces is None else (traces,)

    def _resolve_alpha(self, alpha: Any):
        if isinstance(alpha, str):
            if alpha == "auto":
                return self.alpha
            if alpha == "probability":
                values = self.alpha
                if values is None:
                    raise ValueError("probabilities are not available for probability alpha")
                return values
        return alpha

    def _scaled_alpha(self, alpha, *, min_alpha, alpha_scale):
        if not np.isfinite(min_alpha) or not 0 <= min_alpha <= 1:
            raise ValueError("min_alpha must be between 0 and 1")
        if alpha_scale not in ("linear", "normalize", "log"):
            raise ValueError("alpha_scale must be 'linear', 'normalize', or 'log'")
        values = self._resolve_alpha(alpha)
        if values is None or (min_alpha == 0 and alpha_scale == "linear"):
            return values
        values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
        values = np.maximum(values, 0.0)
        positive = values > 0
        if alpha_scale == "normalize" and values.size and np.max(values) > 0:
            values = values / np.max(values)
        elif alpha_scale == "log" and np.any(positive):
            logs = np.log(values[positive])
            scaled = np.ones_like(logs) if np.ptp(logs) == 0 else (logs - logs.min()) / np.ptp(logs)
            values = np.zeros_like(values)
            values[positive] = scaled
        values = np.clip(values, min_alpha, 1.0)
        return float(values) if values.ndim == 0 else values


def _clip_polyline(points, bounds):
    """Clip line segments to a box, preserving disconnected path pieces."""
    pieces, current = [], []
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        lower, upper = 0.0, 1.0
        for axis in range(3):
            lo, hi = bounds[2 * axis : 2 * axis + 2]
            if delta[axis] == 0:
                if start[axis] < lo or start[axis] > hi:
                    upper = -1.0
                    break
            else:
                crossings = sorted(((lo - start[axis]) / delta[axis], (hi - start[axis]) / delta[axis]))
                lower, upper = max(lower, crossings[0]), min(upper, crossings[1])
        if lower >= upper:
            if current:
                pieces.append(np.asarray(current))
                current = []
            continue
        first, last = start + lower * delta, start + upper * delta
        if current and not np.array_equal(current[-1], first):
            pieces.append(np.asarray(current))
            current = []
        if not current:
            current.append(first)
        current.append(last)
    if current:
        pieces.append(np.asarray(current))
    return pieces
