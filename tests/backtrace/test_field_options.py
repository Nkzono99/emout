"""Field opt-outs must reach each solver and remote workflow consistently."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from emout.core.backtrace.solver_wrapper import BacktraceWrapper
from emout.core.backtrace.trace_wrapper import TraceWrapper


FIELD_OPTIONS = [(True, True), (False, True), (True, False), (False, False)]
BACKTRACE_METHODS = [
    "get_backtrace",
    "get_backtraces",
    "get_backtraces_from_particles",
    "get_probabilities",
    "get_probabilities_from_array",
    "get_probabilities_from_particles",
]


@pytest.fixture
def solver_backend(monkeypatch):
    """Provide solver API doubles without installing the optional dependency."""
    calls = []

    class Particle:
        def __init__(self, pos, vel):
            self.pos = np.asarray(pos)
            self.vel = np.asarray(vel)

    class PhaseGrid:
        def __init__(self, *axes):
            self.phases = np.asarray([axes], dtype=float)

        def create_grid(self):
            return self.phases

        def create_particles(self):
            return [Particle(phase[:3], phase[3:]) for phase in self.phases]

    def make_backend(name, parallel):
        def backend(*args, **kwargs):
            calls.append((name, parallel, kwargs))
            if name == "get_backtrace":
                return np.zeros(2), 1.0, np.zeros((2, 3)), np.zeros((2, 3))
            particles = kwargs["particles"]
            n = len(particles)
            if name == "get_probabilities":
                return np.ones(n), particles
            return np.zeros((n, 2)), np.ones(n), np.zeros((n, 2, 3)), np.zeros((n, 2, 3)), np.full(n, 2)

        return backend

    package = ModuleType("vdsolverf")
    core = ModuleType("vdsolverf.core")
    core.Particle = Particle
    core.PhaseGrid = PhaseGrid
    emses = ModuleType("vdsolverf.emses")
    mpi = ModuleType("vdsolverf.emses.mpi")
    for name in ("get_backtrace", "get_backtraces", "get_probabilities"):
        setattr(emses, name, make_backend(name, "thread"))
        setattr(mpi, name, make_backend(name, "mpi"))
        setattr(mpi, f"srun_{name}", make_backend(name, "srun"))
    for module in (package, core, emses, mpi):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr("emout.core.backtrace.solver_wrapper.run_backend", lambda func, *a, **kw: func(*a, **kw))
    return SimpleNamespace(calls=calls, Particle=Particle, make_backend=make_backend)


def _call_backtrace(wrapper, method, solver, **kwargs):
    position = np.array([1.0, 2.0, 3.0])
    velocity = np.array([4.0, 5.0, 6.0])
    if method == "get_backtrace":
        return wrapper.get_backtrace(position, velocity, **kwargs)
    if method == "get_probabilities":
        return wrapper.get_probabilities(*position, *velocity, remote=False, **kwargs)
    if method.endswith("_from_particles"):
        return getattr(wrapper, method)([solver.Particle(position, velocity)], **kwargs)
    return getattr(wrapper, method)(position[None, :], velocity[None, :], **kwargs)


@pytest.mark.parametrize("method", BACKTRACE_METHODS)
@pytest.mark.parametrize("parallel", ["thread", "mpi", "srun"])
@pytest.mark.parametrize("electric,magnetic", FIELD_OPTIONS)
def test_backtrace_field_options_reach_selected_backend(solver_backend, method, parallel, electric, magnetic):
    wrapper = BacktraceWrapper("/fake/output", SimpleNamespace(dt=0.1), None)

    result = _call_backtrace(
        wrapper,
        method,
        solver_backend,
        parallel=parallel,
        use_electric_field=electric,
        use_magnetic_field=magnetic,
    )

    assert result is not None
    assert len(solver_backend.calls) == 1
    name, selected_parallel, kwargs = solver_backend.calls[0]
    expected_name = (
        "get_backtrace"
        if method == "get_backtrace"
        else ("get_probabilities" if "probabilities" in method else "get_backtraces")
    )
    assert (name, selected_parallel) == (expected_name, parallel)
    assert kwargs.get("use_electric_field", True) is electric
    assert kwargs.get("use_magnetic_field", True) is magnetic
    assert "parallel" not in kwargs


@pytest.mark.parametrize("method", BACKTRACE_METHODS)
@pytest.mark.parametrize("explicit_defaults", [False, True])
def test_default_fields_remain_compatible_with_older_backends(solver_backend, method, explicit_defaults):
    wrapper = BacktraceWrapper("/fake/output", SimpleNamespace(dt=0.1), None)
    name = (
        "get_backtrace"
        if method == "get_backtrace"
        else ("get_probabilities" if "probabilities" in method else "get_backtraces")
    )
    backend = solver_backend.make_backend(name, "custom")

    # An older backend has no field keywords or arbitrary keyword arguments.
    def old_backend(
        directory,
        ispec,
        istep,
        dt,
        max_step,
        use_adaptive_dt,
        particle=None,
        particles=None,
        output_interval=1,
        n_threads=1,
    ):
        return backend(particle=particle, particles=particles)

    kwargs = {"parallel": old_backend}
    if explicit_defaults:
        kwargs.update(use_electric_field=True, use_magnetic_field=True)
    assert _call_backtrace(wrapper, method, solver_backend, **kwargs) is not None


@pytest.mark.parametrize("method", BACKTRACE_METHODS)
def test_explicit_zero_dt_reaches_backend(solver_backend, method):
    wrapper = BacktraceWrapper("/fake/output", SimpleNamespace(dt=0.1), None)
    _call_backtrace(wrapper, method, solver_backend, dt=0.0)
    assert solver_backend.calls[0][2]["dt"] == 0.0


@pytest.mark.parametrize("direction", ["backward", "forward", "both"])
@pytest.mark.parametrize("get_trace,get_probabilities", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("electric,magnetic", FIELD_OPTIONS)
def test_trace_uses_same_fields_for_every_requested_payload(
    solver_backend, direction, get_trace, get_probabilities, electric, magnetic
):
    wrapper = TraceWrapper("/fake/output", SimpleNamespace(dt=0.1), None)

    result = getattr(wrapper, direction)(
        1,
        2,
        3,
        4,
        5,
        6,
        get_trace=get_trace,
        get_probabilities=get_probabilities,
        use_electric_field=electric,
        use_magnetic_field=magnetic,
        remote=False,
    )

    probability_calls = [call for call in solver_backend.calls if call[0] == "get_probabilities"]
    trace_calls = [call for call in solver_backend.calls if call[0] == "get_backtraces"]
    assert len(probability_calls) == int(get_probabilities)
    expected_dt = {"backward": [0.1], "forward": [-0.1], "both": [0.1, -0.1]}[direction] if get_trace else []
    assert [call[2]["dt"] for call in trace_calls] == expected_dt
    assert (result.probabilities is not None) is get_probabilities
    for _, _, kwargs in solver_backend.calls:
        assert kwargs.get("use_electric_field", True) is electric
        assert kwargs.get("use_magnetic_field", True) is magnetic


@pytest.mark.parametrize("method", ["backward", "forward", "both", "get_probabilities"])
@pytest.mark.parametrize("electric,magnetic", FIELD_OPTIONS)
def test_remote_field_options_reach_worker_and_solver(monkeypatch, solver_backend, method, electric, magnetic):
    from emout.distributed import remote_render

    worker_trace = TraceWrapper("/fake/output", SimpleNamespace(dt=0.1), None)
    worker = remote_render.RemoteSession.__new__(remote_render.RemoteSession)
    worker._cache = {}
    monkeypatch.setattr(
        worker, "_resolve", lambda emout_kwargs: SimpleNamespace(trace=worker_trace, backtrace=worker_trace.backtrace)
    )
    submitted = []

    class Session:
        def compute_trace(self, key, **kwargs):
            submitted.append(kwargs)
            return SimpleNamespace(result=lambda: worker.compute_trace(key, **kwargs))

        def compute_probabilities(self, key, **kwargs):
            submitted.append(kwargs)
            return SimpleNamespace(result=lambda: worker.compute_probabilities(key, **kwargs))

    session = Session()
    monkeypatch.setattr(remote_render, "get_or_create_session", lambda **kwargs: session)
    monkeypatch.setattr(remote_render, "_next_key", lambda prefix: "fields")
    wrapper = TraceWrapper(
        "/fake/output", SimpleNamespace(dt=0.1), None, remote_open_kwargs={"directory": "/fake/output"}
    )
    if method == "get_probabilities":
        result = wrapper.backtrace.get_probabilities(
            1, 2, 3, 4, 5, 6, use_electric_field=electric, use_magnetic_field=magnetic
        )
        assert isinstance(result, remote_render.RemoteProbabilityResult)
    else:
        result = getattr(wrapper, method)(
            1, 2, 3, 4, 5, 6, get_trace=True, use_electric_field=electric, use_magnetic_field=magnetic
        )
        assert isinstance(result, remote_render.RemoteTraceResult)

    assert submitted[0]["remote"] is False
    assert submitted[0].get("use_electric_field", True) is electric
    assert submitted[0].get("use_magnetic_field", True) is magnetic
    assert "fields" in worker._cache
    assert solver_backend.calls
    for _, _, kwargs in solver_backend.calls:
        assert kwargs.get("use_electric_field", True) is electric
        assert kwargs.get("use_magnetic_field", True) is magnetic
