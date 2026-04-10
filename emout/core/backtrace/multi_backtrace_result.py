"""Multi-particle backtrace result container.

:class:`MultiBacktraceResult` aggregates multiple :class:`BacktraceResult`
instances and supports sampling, iteration, and statistical queries.
"""

from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np

from .xy_data import MultiXYData


class MultiBacktraceResult:
    """Container for multiple particle backtrace results.

    Attributes (all NumPy arrays):

    - ``ts_list``         -- shape ``(N_traj, N_steps)``
    - ``probabilities``   -- shape ``(N_traj,)``
    - ``positions_list``  -- shape ``(N_traj, N_steps, 3)``
    - ``velocities_list`` -- shape ``(N_traj, N_steps, 3)``
    - ``last_indexes``    -- shape ``(N_traj,)``

    Usage::

        ts_list, probs, pos_list, vel_list = result  # tuple unpacking

        result.pair("x", "y")  # MultiXYData: x vs y per trajectory
        result.pair("t", "x")  # t vs x per trajectory
        result.tvx             # shorthand for pair("t", "vx")

        result.sample(10)      # randomly sample 10 trajectories
        result.sample([0,2,5]) # pick specific trajectory indices

        result.yvz.plot()
    """

    _VALID_KEYS = {"t", "x", "y", "z", "vx", "vy", "vz"}

    def __init__(
        self,
        ts_list: np.ndarray,
        probabilities: np.ndarray,
        positions_list: np.ndarray,
        velocities_list: np.ndarray,
        last_indexes: np.ndarray,
        unit=None,
    ):
        """
        Parameters
        ----------
        ts_list : numpy.ndarray, shape = (N_traj, N_steps)
        probabilities : numpy.ndarray, shape = (N_traj,)
        positions_list : numpy.ndarray, shape = (N_traj, N_steps, 3)
        velocities_list : numpy.ndarray, shape = (N_traj, N_steps, 3)
        last_indexes : numpy.ndarray, shape = (N_traj,)
        """
        if ts_list.ndim != 2:
            raise ValueError("ts_list must be a 2-D array of shape (N_traj, N_steps)")
        N, T = ts_list.shape

        if probabilities.shape != (N,):
            raise ValueError("probabilities must have shape (N_traj,)")
        if positions_list.ndim != 3 or positions_list.shape != (N, T, 3):
            raise ValueError("positions_list must have shape (N_traj, N_steps, 3)")
        if velocities_list.ndim != 3 or velocities_list.shape != (N, T, 3):
            raise ValueError("velocities_list must have shape (N_traj, N_steps, 3)")

        self.ts_list = ts_list
        self.probabilities = probabilities
        self.positions_list = positions_list
        self.velocities_list = velocities_list
        self.last_indexes = last_indexes

        self.unit = unit

    def __iter__(self) -> Iterator[Any]:
        """Support tuple unpacking::

        ts_list, probabilities, positions_list, velocities_list, last_indexes = result
        """
        yield self.ts_list
        yield self.probabilities
        yield self.positions_list
        yield self.velocities_list
        yield self.last_indexes

    def __repr__(self) -> str:
        """Return a string representation.

        Returns
        -------
        str
            Human-readable summary.
        """
        N, T = self.ts_list.shape
        return (
            f"<MultiBacktraceResult: n_traj={N}, n_steps={T}, keys={list(MultiBacktraceResult._VALID_KEYS)}> "
            f"(use result.pair(var1,var2) or result.sample(...) to inspect)"
        )

    def sample(
        self,
        indices: Union[int, Sequence[int], range, slice],
        random_state: Optional[int] = None,
    ) -> "MultiBacktraceResult":
        """Sample a subset of trajectories and return a new result.

        Parameters
        ----------
        indices : int
            Positive integer *k*: randomly sample *k* trajectories.
        indices : Sequence[int], range, or slice
            Explicit index selection.
        random_state : int, optional
            Random seed for reproducible sampling.
        """
        N, T = self.ts_list.shape

        if isinstance(indices, int):
            k = indices
            if not (0 <= k <= N):
                raise ValueError("sample(int): k must satisfy 0 <= k <= N_traj")
            rng = np.random.RandomState(random_state)
            chosen = rng.choice(N, size=k, replace=False)

        elif isinstance(indices, slice):
            chosen = list(range(N))[indices]

        elif isinstance(indices, range):
            chosen = list(indices)

        elif hasattr(indices, "__iter__"):
            chosen = list(indices)

        else:
            raise TypeError("sample() argument must be int, slice, range, or Sequence[int]")

        if any((i < 0 or i >= N) for i in chosen):
            raise IndexError("sample(): index out of range")

        ts_sub = self.ts_list[chosen, :]
        prob_sub = self.probabilities[chosen]
        pos_sub = self.positions_list[chosen, :, :]
        vel_sub = self.velocities_list[chosen, :, :]
        last_indexes_sub = self.last_indexes[chosen]

        return MultiBacktraceResult(
            ts_sub,
            prob_sub,
            pos_sub,
            vel_sub,
            last_indexes_sub,
            unit=self.unit,
        )

    def pair(self, var1: str, var2: str) -> MultiXYData:
        """Extract two variables and return a MultiXYData.

        Parameters *var1* and *var2* must each be one of
        ``'t'``, ``'x'``, ``'y'``, ``'z'``, ``'vx'``, ``'vy'``, ``'vz'``.

        Examples::

            result.pair("x", "y")  # x vs y per trajectory
            result.pair("t", "x")  # t vs x per trajectory
            result.tvy             # shorthand for pair("t", "vy")
        """
        if var1 not in MultiBacktraceResult._VALID_KEYS or var2 not in MultiBacktraceResult._VALID_KEYS:
            raise KeyError(f"Allowed keys = {MultiBacktraceResult._VALID_KEYS}, but got '{var1}', '{var2}'")

        def _get_array_list(key: str) -> np.ndarray:
            """Return the data array and unit for the given key.

            Parameters
            ----------
            key : str
                Variable key (e.g. ``"t"``, ``"x"``, ``"vx"``).

            Returns
            -------
            tuple of (np.ndarray, UnitTranslator or None)
                Data array and associated unit translator.
            """
            if key == "t":
                u = self.unit.t if self.unit else None

                return self.ts_list, u

            elif key in {"x", "y", "z"}:
                idx = {"x": 0, "y": 1, "z": 2}[key]

                u = self.unit.length if self.unit else None

                return self.positions_list[:, :, idx], u

            elif key in {"vx", "vy", "vz"}:
                idx = {"vx": 0, "vy": 1, "vz": 2}[key]

                u = self.unit.v if self.unit else None

                return self.velocities_list[:, :, idx], u

            else:
                raise KeyError(f"Unexpected key: {key}")

        arr1, u1 = _get_array_list(var1)  # shape = (N_traj, N_steps)
        arr2, u2 = _get_array_list(var2)

        xlabel = f"{var1} [m]" if self.unit else var1
        ylabel = f"{var2} [m]" if self.unit else var2
        title = f"{var1} vs {var2} (multiple trajectories)"

        return MultiXYData(
            arr1,
            arr2,
            self.last_indexes,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            units=(u1, u2) if u1 else None,
        )

    def __getattr__(self, name: str) -> Any:
        """Interpret attribute access as a pair name.

        Examples::

            result.xvy  # -> pair("x", "vy")
            result.tz   # -> pair("t", "z")
            result.tvx  # -> pair("t", "vx")
        """

        for key1 in MultiBacktraceResult._VALID_KEYS:
            if name.startswith(key1):
                rest = name[len(key1) :]
                if rest in MultiBacktraceResult._VALID_KEYS:
                    return self.pair(key1, rest)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
