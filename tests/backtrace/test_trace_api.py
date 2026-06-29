"""Tests for the high-level trace workflow API."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.backtrace.test_backtrace_extended import _make_multi_backtrace_result


def _make_wrapper(dt=0.2):
    from emout.core.backtrace.trace_wrapper import TraceWrapper

    inp = SimpleNamespace(dt=dt)
    return TraceWrapper(directory="/fake/dir", inp=inp, unit=None)


def test_backward_returns_integrated_probability_result(monkeypatch):
    from emout.core.backtrace.trace_result import TraceResult

    wrapper = _make_wrapper()
    probability = SimpleNamespace(
        probabilities=np.array([0.25, 1.25, np.nan]),
        particles=["p0", "p1", "p2"],
    )
    calls = []

    def fake_get_probabilities(*args, **kwargs):
        calls.append((args, kwargs))
        return probability

    monkeypatch.setattr(wrapper.backtrace, "get_probabilities", fake_get_probabilities)

    result = wrapper.backward(1, 2, 3, 4, 5, 6)

    assert isinstance(result, TraceResult)
    assert result.direction == "backward"
    assert result.probabilities is probability
    assert result.backward_traces is None
    assert result.forward_traces is None
    np.testing.assert_allclose(result.alpha, np.array([0.25, 1.0, 0.0]))
    assert calls[0][1]["dt"] == pytest.approx(0.2)


def test_forward_trace_only_builds_particles_without_probabilities(monkeypatch):
    wrapper = _make_wrapper(dt=0.5)
    traces = _make_multi_backtrace_result(n_traj=2, n_steps=4)
    probability_calls = []
    trace_calls = []

    class FakePhaseGrid:
        def __init__(self, x, y, z, vx, vy, vz):
            self.axes = (x, y, z, vx, vy, vz)

        def create_grid(self):
            return np.zeros((2, 6))

        def create_particles(self):
            return ["p0", "p1"]

    def fake_get_probabilities(*args, **kwargs):
        probability_calls.append((args, kwargs))
        raise AssertionError("probability calculation should not run")

    def fake_get_backtraces_from_particles(particles, **kwargs):
        trace_calls.append((particles, kwargs))
        return traces

    monkeypatch.setattr(wrapper, "_phase_grid_cls", lambda: FakePhaseGrid)
    monkeypatch.setattr(wrapper.backtrace, "get_probabilities", fake_get_probabilities)
    monkeypatch.setattr(wrapper.backtrace, "get_backtraces_from_particles", fake_get_backtraces_from_particles)

    result = wrapper.forward(
        [1.0, 2.0],
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        get_trace=True,
        get_probabilities=False,
        max_step=12,
    )

    assert probability_calls == []
    assert result.probabilities is None
    assert result.forward_traces is traces
    assert result.traces is traces
    assert result.alpha is None
    assert trace_calls == [
        (
            ["p0", "p1"],
            {
                "ispec": 0,
                "istep": -1,
                "dt": pytest.approx(-0.5),
                "max_step": 12,
                "output_interval": 1,
                "use_adaptive_dt": False,
                "n_threads": 4,
            },
        )
    ]


def test_both_reuses_one_probability_grid_for_both_directions(monkeypatch):
    wrapper = _make_wrapper(dt=0.25)
    probability = SimpleNamespace(probabilities=np.array([0.2, 0.8]), particles=["p0", "p1"])
    backward = _make_multi_backtrace_result(n_traj=2, n_steps=3)
    forward = _make_multi_backtrace_result(n_traj=2, n_steps=5)
    probability_calls = []
    trace_calls = []

    def fake_get_probabilities(*args, **kwargs):
        probability_calls.append((args, kwargs))
        return probability

    def fake_get_backtraces_from_particles(particles, **kwargs):
        trace_calls.append((particles, kwargs))
        return backward if kwargs["dt"] > 0 else forward

    monkeypatch.setattr(wrapper.backtrace, "get_probabilities", fake_get_probabilities)
    monkeypatch.setattr(wrapper.backtrace, "get_backtraces_from_particles", fake_get_backtraces_from_particles)

    result = wrapper.both(1, 2, 3, 4, 5, 6, get_trace=True)

    assert len(probability_calls) == 1
    assert result.probabilities is probability
    assert result.backward_traces is backward
    assert result.forward_traces is forward
    assert [call[1]["dt"] for call in trace_calls] == [pytest.approx(0.25), pytest.approx(-0.25)]
    with pytest.raises(ValueError, match="both backward and forward"):
        _ = result.traces


def test_trace_result_plot_traces_uses_probability_alpha():
    from emout.core.backtrace.trace_result import TraceResult

    captured = {}

    class DummyXY:
        def plot(self, **kwargs):
            captured.update(kwargs)
            return "axes"

    class DummyTraces:
        def pair(self, var1, var2):
            captured["pair"] = (var1, var2)
            return DummyXY()

    probability = SimpleNamespace(probabilities=np.array([0.3, 2.0, np.nan]))
    result = TraceResult(
        direction="backward",
        probabilities=probability,
        backward_traces=DummyTraces(),
    )

    assert result.plot_traces("x", "z") == "axes"
    assert captured["pair"] == ("x", "z")
    np.testing.assert_allclose(captured["alpha"], np.array([0.3, 1.0, 0.0]))


def test_trace_result_rejects_probability_alpha_without_probabilities():
    from emout.core.backtrace.trace_result import TraceResult

    result = TraceResult(direction="forward", forward_traces=SimpleNamespace())

    with pytest.raises(ValueError, match="probabilities are not available"):
        result.plot_traces("x", "z", alpha="probability")


def test_trace_result_accepts_explicit_alpha_array():
    from emout.core.backtrace.trace_result import TraceResult

    captured = {}
    alpha = np.array([0.1, 0.4])

    class DummyXY:
        def plot(self, **kwargs):
            captured.update(kwargs)
            return "axes"

    class DummyTraces:
        def pair(self, var1, var2):
            return DummyXY()

    result = TraceResult(direction="forward", forward_traces=DummyTraces())

    assert result.plot_traces("x", "z", alpha=alpha) == "axes"
    np.testing.assert_array_equal(captured["alpha"], alpha)


def test_trace_result_plot3d_adds_trajectory_meshes(monkeypatch):
    from emout.core.backtrace import trace_result as trace_result_module
    from emout.core.backtrace.trace_result import TraceResult

    traces = _make_multi_backtrace_result(n_traj=2, n_steps=4)
    result = TraceResult(direction="forward", forward_traces=traces)

    class FakePolyData:
        def __init__(self, points):
            self.points = points
            self.lines = None

        def tube(self, radius):
            self.tube_radius = radius
            return self

    class FakePlotter:
        def __init__(self):
            self.meshes = []
            self.shown = False

        def add_mesh(self, mesh, **kwargs):
            self.meshes.append((mesh, kwargs))

        def add_axes(self):
            self.axes_added = True

        def show(self):
            self.shown = True

    fake_pv = SimpleNamespace(Plotter=FakePlotter, PolyData=FakePolyData)
    monkeypatch.setattr(trace_result_module, "_require_pyvista", lambda: fake_pv)

    plotter = result.plot3d(tube_radius=0.1, show=True)

    assert plotter.shown is True
    assert len(plotter.meshes) == 2
    assert plotter.meshes[0][0].points.shape[1] == 3
    assert plotter.meshes[0][1]["opacity"] == pytest.approx(1.0)
