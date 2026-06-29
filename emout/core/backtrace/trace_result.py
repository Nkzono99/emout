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
        """Plot trajectory pairs with optional probability-derived alpha."""
        traces = self._select_traces(direction)
        if traces is None:
            raise ValueError("traces are not available")

        resolved_alpha = self._resolve_alpha(alpha)
        if resolved_alpha is not None:
            plot_kwargs = {**plot_kwargs, "alpha": resolved_alpha}

        return traces.pair(var1, var2).plot(**plot_kwargs)

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
        **mesh_kwargs,
    ):
        """Draw trajectories on a PyVista plotter and return it.

        Passing an existing ``plotter`` overlays the trajectories on the
        caller's current 3-D field or boundary view.
        """
        traces = self._select_traces(direction)
        if traces is None:
            raise ValueError("traces are not available")

        pv = _require_pyvista()
        if plotter is None:
            plotter = pv.Plotter()

        positions = np.asarray(traces.positions_list, dtype=float)
        last_indexes = np.asarray(traces.last_indexes, dtype=int)
        alphas = self._resolve_alpha(alpha)
        if offsets is None:
            offsets = (None, None, None)

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

            line = pv.PolyData(points)
            line.lines = np.concatenate(([len(points)], np.arange(len(points), dtype=int)))
            mesh = line.tube(radius=tube_radius) if tube_radius is not None else line

            opacity = 1.0
            if alphas is not None:
                if hasattr(alphas, "__len__"):
                    opacity = float(alphas[index])
                else:
                    opacity = float(alphas)

            add_mesh_kwargs = {
                "color": color,
                "line_width": line_width,
                "opacity": opacity,
            }
            add_mesh_kwargs.update(mesh_kwargs)
            plotter.add_mesh(mesh, **add_mesh_kwargs)

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
