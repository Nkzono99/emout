"""High-level wrapper for particle backtrace solvers.

:class:`BacktraceWrapper` configures field interpolation, integrator
parameters, and optional Dask-based remote execution, then delegates
to the underlying ODE solver.
"""

from typing import Any, List, Sequence, Tuple, Union

import numpy as np

from .backtrace_result import BacktraceResult
from .multi_backtrace_result import MultiBacktraceResult
from .probability_result import ProbabilityResult

from emout.distributed.utils import run_backend


class BacktraceWrapper:
    """High-level wrapper for particle backtrace solvers.

    Configures field interpolation, integrator parameters, and optional
    Dask-based remote execution, then delegates to the underlying ODE
    solver.
    """

    def __init__(
        self,
        directory: Any,
        inp: Any,
        unit: Any,
        remote_open_kwargs: Any = None,
    ):
        """
        Parameters
        ----------
        directory : Path
            Simulation output directory (same semantics as ``Emout.directory``)
        inp : InpFile
            Parsed input-parameter object holding ``dt`` and related fields
        """
        self.directory = directory
        self.inp = inp
        self.unit = unit
        self.remote_open_kwargs = remote_open_kwargs

    def get_backtrace(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        ispec: int = 0,
        istep: int = -1,
        dt: Union[float, None] = None,
        max_step: int = 30000,
        output_interval: int = 1,
        use_adaptive_dt: bool = False,
        **kwargs,
    ) -> Tuple[Any, Any, Any, Any]:
        """Run a single-particle backtrace and return the trajectory.

        Parameters
        ----------
        position : np.ndarray
            Initial particle position
        velocity : np.ndarray
            Initial particle velocity
        ispec : int, optional
            Particle species index (0-based)
        istep : int, optional
            Starting time-step index
        dt : float or None, optional
            Time step size
        max_step : int, optional
            Maximum number of backtrace steps
        output_interval : int, optional
            Output interval in steps
        use_adaptive_dt : bool, optional
            If True, use adaptive time stepping during backtrace
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying function.

        Returns
        -------
        BacktraceResult
            Trajectory data containing times, probability, positions, and velocities.
        """
        from vdsolverf.core import Particle
        from vdsolverf.emses import get_backtrace as _backend

        particle = Particle(position, velocity)

        ts, probability, positions, velocities = run_backend(
            _backend,
            directory=self.directory,
            ispec=ispec,
            istep=istep,
            particle=particle,
            dt=dt or self.inp.dt,
            max_step=max_step,
            output_interval=output_interval,
            use_adaptive_dt=use_adaptive_dt,
            **kwargs,
        )
        return BacktraceResult(ts, probability, positions, velocities, unit=self.unit)

    def get_backtraces(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        ispec: int = 0,
        istep: int = -1,
        dt: Union[float, None] = None,
        max_step: int = 10000,
        output_interval: int = 1,
        use_adaptive_dt: bool = False,
        n_threads: int = 4,
        **kwargs,
    ) -> Any:
        """Run backtraces for multiple particles and return aggregated results.

        Parameters
        ----------
        positions : np.ndarray
            Particle position array, shape ``(N, 3)``
        velocities : np.ndarray
            Particle velocity array, shape ``(N, 3)``
        ispec : int, optional
            Particle species index (0-based)
        istep : int, optional
            Starting time-step index
        dt : float or None, optional
            Time step size
        max_step : int, optional
            Maximum number of backtrace steps
        output_interval : int, optional
            Output interval in steps
        use_adaptive_dt : bool, optional
            If True, use adaptive time stepping during backtrace
        n_threads : int, optional
            Number of parallel threads
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying function.

        Returns
        -------
        MultiBacktraceResult
            Aggregated trajectory data for all particles.
        """
        from vdsolverf.core import Particle
        from vdsolverf.emses import get_backtraces as _backend

        if positions.shape != velocities.shape:
            raise ValueError("positions and velocities must have the same shape")

        particles: List[Any] = [Particle(pos_vec, vel_vec) for pos_vec, vel_vec in zip(positions, velocities)]

        ts_list, probabilities, positions_list, velocities_list, last_indexes = run_backend(
            _backend,
            self.directory,
            ispec=ispec,
            istep=istep,
            particles=particles,
            dt=dt or self.inp.dt,
            max_step=max_step,
            output_interval=output_interval,
            use_adaptive_dt=use_adaptive_dt,
            n_threads=n_threads,
            **kwargs,
        )

        return MultiBacktraceResult(
            ts_list,
            probabilities,
            positions_list,
            velocities_list,
            last_indexes,
            unit=self.unit,
        )

    def get_backtraces_from_particles(
        self,
        particles: Sequence[Any],
        ispec: int = 0,
        istep: int = -1,
        dt: Union[float, None] = None,
        max_step: int = 10000,
        output_interval: int = 1,
        use_adaptive_dt: bool = False,
        n_threads: int = 4,
        **kwargs,
    ) -> Any:
        """Run backtraces from pre-built Particle objects.

        Parameters
        ----------
        particles : Sequence[Any]
            Collection of particles to backtrace
        ispec : int, optional
            Particle species index (0-based)
        istep : int, optional
            Starting time-step index
        dt : float or None, optional
            Time step size
        max_step : int, optional
            Maximum number of backtrace steps
        output_interval : int, optional
            Output interval in steps
        use_adaptive_dt : bool, optional
            If True, use adaptive time stepping during backtrace
        n_threads : int, optional
            Number of parallel threads
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying function.

        Returns
        -------
        MultiBacktraceResult
            Aggregated trajectory data for all particles.
        """
        from vdsolverf.emses import get_backtraces as _backend

        ts_list, probabilities, positions_list, velocities_list, last_indexes = run_backend(
            _backend,
            self.directory,
            ispec=ispec,
            istep=istep,
            particles=particles,
            dt=dt or self.inp.dt,
            max_step=max_step,
            output_interval=output_interval,
            use_adaptive_dt=use_adaptive_dt,
            n_threads=n_threads,
            **kwargs,
        )

        return MultiBacktraceResult(
            ts_list,
            probabilities,
            positions_list,
            velocities_list,
            last_indexes,
            unit=self.unit,
        )

    def get_probabilities(
        self,
        x: Union[Tuple[float, float, int], Sequence[float]],
        y: Union[Tuple[float, float, int], Sequence[float]],
        z: Union[Tuple[float, float, int], Sequence[float]],
        vx: Union[Tuple[float, float, int], Sequence[float]],
        vy: Union[Tuple[float, float, int], Sequence[float]],
        vz: Union[Tuple[float, float, int], Sequence[float]],
        ispec: int = 0,
        istep: int = -1,
        dt: Union[float, None] = None,
        max_step: int = 10000,
        use_adaptive_dt: bool = False,
        n_threads: int = 4,
        remote: bool = True,
        **kwargs,
    ) -> "ProbabilityResult":
        """Compute arrival probabilities over a 6-D phase-space grid.

        Parameters
        ----------
        x : tuple of (float, float, int) or sequence of float
            X coordinates or grid specification ``(start, stop, n)``
        y : tuple of (float, float, int) or sequence of float
            Y coordinates or grid specification
        z : tuple of (float, float, int) or sequence of float
            Z coordinates or grid specification
        vx : tuple of (float, float, int) or sequence of float
            Velocity x-component values or grid specification
        vy : tuple of (float, float, int) or sequence of float
            Velocity y-component values or grid specification
        vz : tuple of (float, float, int) or sequence of float
            Velocity z-component values or grid specification
        ispec : int, optional
            Particle species index (0-based)
        istep : int, optional
            Starting time-step index
        dt : float or None, optional
            Time step size
        max_step : int, optional
            Maximum number of backtrace steps
        use_adaptive_dt : bool, optional
            If True, use adaptive time stepping during backtrace
        n_threads : int, optional
            Number of parallel threads
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying function.

        Returns
        -------
        ProbabilityResult
            Phase-space probability distribution.
        """
        # If a Dask Actor is running, compute + cache on the worker and
        # return a proxy (avoids transferring large numpy arrays locally)
        if remote:
            from emout.distributed.remote_render import (
                get_or_create_session,
                RemoteProbabilityResult,
                _next_key,
            )

            session = get_or_create_session(
                emout_kwargs=self.remote_open_kwargs,
                emout_dir=self.directory,
            )
            if session is not None:
                key = _next_key("prob")
                session.compute_probabilities(
                    key,
                    emout_kwargs=self.remote_open_kwargs,
                    x=x,
                    y=y,
                    z=z,
                    vx=vx,
                    vy=vy,
                    vz=vz,
                    ispec=ispec,
                    istep=istep,
                    dt=dt,
                    max_step=max_step,
                    use_adaptive_dt=use_adaptive_dt,
                    n_threads=n_threads,
                    remote=False,  # no recursion on the worker side
                    **kwargs,
                ).result()
                return RemoteProbabilityResult(session, key)

        from vdsolverf.core import PhaseGrid
        from vdsolverf.emses import get_probabilities as _backend

        phase_grid = PhaseGrid(x, y, z, vx, vy, vz)
        phases = phase_grid.create_grid()  # shape = (N_points, 6)
        particles = phase_grid.create_particles()

        prob_flat, ret_particles = run_backend(
            _backend,
            directory=self.directory,
            ispec=ispec,
            istep=istep,
            particles=particles,
            dt=dt or self.inp.dt,
            max_step=max_step,
            use_adaptive_dt=use_adaptive_dt,
            n_threads=n_threads,
            **kwargs,
        )

        def _size(var):
            """Return the number of grid points for a single axis variable.

            Parameters
            ----------
            var : tuple or sequence or scalar
                Axis specification (start, stop, n), explicit values, or scalar

            Returns
            -------
            int
                Number of grid points along this axis.
            """
            if isinstance(var, tuple) and len(var) == 3 and isinstance(var[2], int):
                return var[2]
            if hasattr(var, "__len__") and not isinstance(var, str):
                return len(var)
            return 1

        nx = _size(x)
        ny = _size(y)
        nz = _size(z)
        nvx = _size(vx)
        nvy = _size(vy)
        nvz = _size(vz)
        dims = (nx, ny, nz, nvx, nvy, nvz)

        return ProbabilityResult(phases, prob_flat, dims, ret_particles, particles, ispec, self.inp, self.unit)

    def get_probabilities_from_array(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        ispec: int = 0,
        istep: int = -1,
        dt: Union[float, None] = None,
        max_step: int = 10000,
        use_adaptive_dt: bool = False,
        n_threads: int = 4,
        **kwargs,
    ) -> Any:
        """Compute arrival probabilities from position/velocity arrays.

        Parameters
        ----------
        positions : np.ndarray
            Particle position array, shape ``(N, 3)``
        velocities : np.ndarray
            Particle velocity array, shape ``(N, 3)``
        ispec : int, optional
            Particle species index (0-based)
        istep : int, optional
            Starting time-step index
        dt : float or None, optional
            Time step size
        max_step : int, optional
            Maximum number of backtrace steps
        use_adaptive_dt : bool, optional
            If True, use adaptive time stepping during backtrace
        n_threads : int, optional
            Number of parallel threads
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying function.

        Returns
        -------
        Any
            Raw probability array returned by the backend.
        """
        from vdsolverf.core import Particle
        from vdsolverf.emses import get_probabilities as _backend

        if positions.shape != velocities.shape:
            raise ValueError("positions and velocities must have the same shape")

        particles = [Particle(p, v) for p, v in zip(positions, velocities)]

        return run_backend(
            _backend,
            self.directory,
            ispec=ispec,
            istep=istep,
            particles=particles,
            dt=dt or self.inp.dt,
            max_step=max_step,
            use_adaptive_dt=use_adaptive_dt,
            n_threads=n_threads,
            **kwargs,
        )

    def get_probabilities_from_particles(
        self,
        particles: Sequence[Any],
        ispec: int = 0,
        istep: int = -1,
        dt: Union[float, None] = None,
        max_step: int = 10000,
        use_adaptive_dt: bool = False,
        n_threads: int = 4,
        **kwargs,
    ) -> Any:
        """Compute arrival probabilities from pre-built Particle objects.

        Parameters
        ----------
        particles : Sequence[Any]
            Collection of particles for probability computation
        ispec : int, optional
            Particle species index (0-based)
        istep : int, optional
            Starting time-step index
        dt : float or None, optional
            Time step size
        max_step : int, optional
            Maximum number of backtrace steps
        use_adaptive_dt : bool, optional
            If True, use adaptive time stepping during backtrace
        n_threads : int, optional
            Number of parallel threads
        **kwargs : dict
            Additional keyword arguments forwarded to the underlying function.

        Returns
        -------
        Any
            Raw probability array returned by the backend.
        """
        from vdsolverf.emses import get_probabilities as _backend

        return run_backend(
            _backend,
            self.directory,
            ispec=ispec,
            istep=istep,
            particles=particles,
            dt=dt or self.inp.dt,
            max_step=max_step,
            use_adaptive_dt=use_adaptive_dt,
            n_threads=n_threads,
            **kwargs,
        )
