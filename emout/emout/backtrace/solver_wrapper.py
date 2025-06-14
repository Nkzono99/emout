from typing import Any, List, Sequence, Tuple, Union

import numpy as np

from .backtrace_result import BacktraceResult
from .multi_backtrace_result import MultiBacktraceResult
from .probability_result import ProbabilityResult

from emout.distributed.utils import run_backend


class BacktraceWrapper:
    def __init__(self, directory: Any, inp: Any, unit: Any):
        """
        Parameters
        ----------
        directory : Path
            Emout.directory と同様のディレクトリパス
        inp : InpFile
            Emout.inp と同様の InpFile オブジェクト。dt などの情報を保持
        """
        self.directory = directory
        self.inp = inp
        self.unit = unit

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
        from vdsolverf.core import Particle
        from vdsolverf.emses import get_backtraces as _backend

        if positions.shape != velocities.shape:
            raise ValueError("positions と velocities の shape が違います")

        particles: List[Any] = [
            Particle(pos_vec, vel_vec)
            for pos_vec, vel_vec in zip(positions, velocities)
        ]

        ts_list, probabilities, positions_list, velocities_list, last_indexes = (
            run_backend(
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
        from vdsolverf.emses import get_backtraces as _backend

        ts_list, probabilities, positions_list, velocities_list, last_indexes = (
            run_backend(
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
        **kwargs,
    ) -> ProbabilityResult:
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

        return ProbabilityResult(
            phases, prob_flat, dims, ret_particles, particles, self.unit
        )

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
        from vdsolverf.core import Particle
        from vdsolverf.emses import get_probabilities as _backend

        if positions.shape != velocities.shape:
            raise ValueError("positions と velocities の shape が違います")

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
