"""High-level trace workflow facade built on top of backtrace solvers."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

from .solver_wrapper import BacktraceWrapper
from .trace_result import TraceResult


class TraceWrapper:
    """High-level trace API exposed as ``data.trace``."""

    def __init__(
        self,
        directory: Any,
        inp: Any,
        unit: Any,
        remote_open_kwargs: Any = None,
    ):
        self.directory = directory
        self.inp = inp
        self.unit = unit
        self.remote_open_kwargs = remote_open_kwargs
        self.backtrace = BacktraceWrapper(directory, inp, unit, remote_open_kwargs=remote_open_kwargs)

    def backward(
        self,
        x: Union[Tuple[float, float, int], Sequence[float], float],
        y: Union[Tuple[float, float, int], Sequence[float], float],
        z: Union[Tuple[float, float, int], Sequence[float], float],
        vx: Union[Tuple[float, float, int], Sequence[float], float],
        vy: Union[Tuple[float, float, int], Sequence[float], float],
        vz: Union[Tuple[float, float, int], Sequence[float], float],
        ispec: int = 0,
        istep: int = -1,
        dt: Optional[float] = None,
        probability_dt: Optional[float] = None,
        max_step: int = 10000,
        output_interval: int = 1,
        use_adaptive_dt: bool = False,
        n_threads: int = 4,
        get_trace: bool = False,
        get_probabilities: bool = True,
        remote: bool = True,
        **kwargs,
    ) -> TraceResult:
        """Compute a backward trace workflow over a phase-space grid."""
        remote_result = self._maybe_remote(
            "backward",
            dict(
                x=x,
                y=y,
                z=z,
                vx=vx,
                vy=vy,
                vz=vz,
                ispec=ispec,
                istep=istep,
                dt=dt,
                probability_dt=probability_dt,
                max_step=max_step,
                output_interval=output_interval,
                use_adaptive_dt=use_adaptive_dt,
                n_threads=n_threads,
                get_trace=get_trace,
                get_probabilities=get_probabilities,
                remote=False,
                **kwargs,
            ),
            remote=remote,
        )
        if remote_result is not None:
            return remote_result

        trace_dt = self._backward_dt(dt)
        prob_dt = self._backward_dt(probability_dt)
        return self._run_one_direction(
            "backward",
            x,
            y,
            z,
            vx,
            vy,
            vz,
            trace_dt=trace_dt,
            probability_dt=prob_dt,
            ispec=ispec,
            istep=istep,
            max_step=max_step,
            output_interval=output_interval,
            use_adaptive_dt=use_adaptive_dt,
            n_threads=n_threads,
            get_trace=get_trace,
            get_probabilities=get_probabilities,
            kwargs=kwargs,
        )

    def forward(
        self,
        x: Union[Tuple[float, float, int], Sequence[float], float],
        y: Union[Tuple[float, float, int], Sequence[float], float],
        z: Union[Tuple[float, float, int], Sequence[float], float],
        vx: Union[Tuple[float, float, int], Sequence[float], float],
        vy: Union[Tuple[float, float, int], Sequence[float], float],
        vz: Union[Tuple[float, float, int], Sequence[float], float],
        ispec: int = 0,
        istep: int = -1,
        dt: Optional[float] = None,
        probability_dt: Optional[float] = None,
        max_step: int = 10000,
        output_interval: int = 1,
        use_adaptive_dt: bool = False,
        n_threads: int = 4,
        get_trace: bool = False,
        get_probabilities: bool = True,
        remote: bool = True,
        **kwargs,
    ) -> TraceResult:
        """Compute a forward trace workflow over a phase-space grid."""
        remote_result = self._maybe_remote(
            "forward",
            dict(
                x=x,
                y=y,
                z=z,
                vx=vx,
                vy=vy,
                vz=vz,
                ispec=ispec,
                istep=istep,
                dt=dt,
                probability_dt=probability_dt,
                max_step=max_step,
                output_interval=output_interval,
                use_adaptive_dt=use_adaptive_dt,
                n_threads=n_threads,
                get_trace=get_trace,
                get_probabilities=get_probabilities,
                remote=False,
                **kwargs,
            ),
            remote=remote,
        )
        if remote_result is not None:
            return remote_result

        trace_dt = self._forward_dt(dt)
        prob_dt = self._backward_dt(probability_dt)
        return self._run_one_direction(
            "forward",
            x,
            y,
            z,
            vx,
            vy,
            vz,
            trace_dt=trace_dt,
            probability_dt=prob_dt,
            ispec=ispec,
            istep=istep,
            max_step=max_step,
            output_interval=output_interval,
            use_adaptive_dt=use_adaptive_dt,
            n_threads=n_threads,
            get_trace=get_trace,
            get_probabilities=get_probabilities,
            kwargs=kwargs,
        )

    def both(
        self,
        x: Union[Tuple[float, float, int], Sequence[float], float],
        y: Union[Tuple[float, float, int], Sequence[float], float],
        z: Union[Tuple[float, float, int], Sequence[float], float],
        vx: Union[Tuple[float, float, int], Sequence[float], float],
        vy: Union[Tuple[float, float, int], Sequence[float], float],
        vz: Union[Tuple[float, float, int], Sequence[float], float],
        ispec: int = 0,
        istep: int = -1,
        dt: Optional[float] = None,
        probability_dt: Optional[float] = None,
        max_step: int = 10000,
        output_interval: int = 1,
        use_adaptive_dt: bool = False,
        n_threads: int = 4,
        get_trace: bool = False,
        get_probabilities: bool = True,
        remote: bool = True,
        **kwargs,
    ) -> TraceResult:
        """Compute backward and forward trajectories from one phase-space grid."""
        remote_result = self._maybe_remote(
            "both",
            dict(
                x=x,
                y=y,
                z=z,
                vx=vx,
                vy=vy,
                vz=vz,
                ispec=ispec,
                istep=istep,
                dt=dt,
                probability_dt=probability_dt,
                max_step=max_step,
                output_interval=output_interval,
                use_adaptive_dt=use_adaptive_dt,
                n_threads=n_threads,
                get_trace=get_trace,
                get_probabilities=get_probabilities,
                remote=False,
                **kwargs,
            ),
            remote=remote,
        )
        if remote_result is not None:
            return remote_result

        self._validate_requested_payloads(get_trace, get_probabilities)
        probability, phases, dims, particles = self._prepare_probability_or_particles(
            x,
            y,
            z,
            vx,
            vy,
            vz,
            ispec=ispec,
            istep=istep,
            probability_dt=self._backward_dt(probability_dt),
            max_step=max_step,
            use_adaptive_dt=use_adaptive_dt,
            n_threads=n_threads,
            get_probabilities=get_probabilities,
            kwargs=kwargs,
        )

        backward_traces = None
        forward_traces = None
        if get_trace:
            backward_traces = self._get_traces_from_particles(
                particles,
                ispec=ispec,
                istep=istep,
                dt=self._backward_dt(dt),
                max_step=max_step,
                output_interval=output_interval,
                use_adaptive_dt=use_adaptive_dt,
                n_threads=n_threads,
                kwargs=kwargs,
            )
            forward_traces = self._get_traces_from_particles(
                particles,
                ispec=ispec,
                istep=istep,
                dt=self._forward_dt(dt),
                max_step=max_step,
                output_interval=output_interval,
                use_adaptive_dt=use_adaptive_dt,
                n_threads=n_threads,
                kwargs=kwargs,
            )

        return TraceResult(
            direction="both",
            probabilities=probability,
            backward_traces=backward_traces,
            forward_traces=forward_traces,
            phases=phases,
            dims=dims,
            particles=particles,
            unit=self.unit,
        )

    def _run_one_direction(
        self,
        direction: str,
        x,
        y,
        z,
        vx,
        vy,
        vz,
        trace_dt: float,
        probability_dt: float,
        ispec: int,
        istep: int,
        max_step: int,
        output_interval: int,
        use_adaptive_dt: bool,
        n_threads: int,
        get_trace: bool,
        get_probabilities: bool,
        kwargs: dict,
    ) -> TraceResult:
        self._validate_requested_payloads(get_trace, get_probabilities)
        probability, phases, dims, particles = self._prepare_probability_or_particles(
            x,
            y,
            z,
            vx,
            vy,
            vz,
            ispec=ispec,
            istep=istep,
            probability_dt=probability_dt,
            max_step=max_step,
            use_adaptive_dt=use_adaptive_dt,
            n_threads=n_threads,
            get_probabilities=get_probabilities,
            kwargs=kwargs,
        )

        traces = None
        if get_trace:
            traces = self._get_traces_from_particles(
                particles,
                ispec=ispec,
                istep=istep,
                dt=trace_dt,
                max_step=max_step,
                output_interval=output_interval,
                use_adaptive_dt=use_adaptive_dt,
                n_threads=n_threads,
                kwargs=kwargs,
            )

        return TraceResult(
            direction=direction,
            probabilities=probability,
            backward_traces=traces if direction == "backward" else None,
            forward_traces=traces if direction == "forward" else None,
            phases=phases,
            dims=dims,
            particles=particles,
            unit=self.unit,
        )

    def _prepare_probability_or_particles(
        self,
        x,
        y,
        z,
        vx,
        vy,
        vz,
        ispec: int,
        istep: int,
        probability_dt: float,
        max_step: int,
        use_adaptive_dt: bool,
        n_threads: int,
        get_probabilities: bool,
        kwargs: dict,
    ):
        if get_probabilities:
            probability = self.backtrace.get_probabilities(
                x,
                y,
                z,
                vx,
                vy,
                vz,
                ispec=ispec,
                istep=istep,
                dt=probability_dt,
                max_step=max_step,
                use_adaptive_dt=use_adaptive_dt,
                n_threads=n_threads,
                **kwargs,
            )
            return (
                probability,
                getattr(probability, "phases", None),
                getattr(probability, "dims", None),
                probability.particles,
            )

        phase_grid = self._phase_grid_cls()(x, y, z, vx, vy, vz)
        phases = phase_grid.create_grid()
        particles = phase_grid.create_particles()
        dims = tuple(self._axis_size(axis) for axis in (x, y, z, vx, vy, vz))
        return None, phases, dims, particles

    def _get_traces_from_particles(
        self,
        particles,
        ispec: int,
        istep: int,
        dt: float,
        max_step: int,
        output_interval: int,
        use_adaptive_dt: bool,
        n_threads: int,
        kwargs: dict,
    ):
        return self.backtrace.get_backtraces_from_particles(
            particles,
            ispec=ispec,
            istep=istep,
            dt=dt,
            max_step=max_step,
            output_interval=output_interval,
            use_adaptive_dt=use_adaptive_dt,
            n_threads=n_threads,
            **kwargs,
        )

    def _maybe_remote(self, method: str, kwargs: dict, remote: bool):
        if not remote or self.remote_open_kwargs is None:
            return None
        from emout.distributed.remote_render import (
            RemoteTraceResult,
            _next_key,
            get_or_create_session,
        )

        session = get_or_create_session(
            emout_kwargs=self.remote_open_kwargs,
            emout_dir=self.directory,
        )
        if session is None:
            return None
        key = _next_key("trace")
        session.compute_trace(
            key,
            emout_kwargs=self.remote_open_kwargs,
            method=method,
            **kwargs,
        ).result()
        return RemoteTraceResult(session, key)

    def _phase_grid_cls(self):
        from vdsolverf.core import PhaseGrid

        return PhaseGrid

    def _backward_dt(self, dt: Optional[float]) -> float:
        if dt is None:
            return abs(float(self.inp.dt))
        return float(dt)

    def _forward_dt(self, dt: Optional[float]) -> float:
        if dt is None:
            return -abs(float(self.inp.dt))
        return float(dt)

    def _validate_requested_payloads(self, get_trace: bool, get_probabilities: bool) -> None:
        if not get_trace and not get_probabilities:
            raise ValueError("At least one of get_trace or get_probabilities must be True")

    def _axis_size(self, axis) -> int:
        if isinstance(axis, tuple) and len(axis) == 3 and isinstance(axis[2], int):
            return axis[2]
        if hasattr(axis, "__len__") and not isinstance(axis, str):
            return len(axis)
        return 1
