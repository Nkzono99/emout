"""Single-particle backtrace result container.

:class:`BacktraceResult` stores the trajectory and field values along a
single particle backtrace and provides attribute-based column access.
"""

from typing import Any, Iterator

import numpy as np

from .xy_data import XYData


class BacktraceResult:
    """Container for a single-particle backtrace result.

    Attributes (all NumPy arrays):

    - ``ts``         -- shape ``(N_steps,)``
    - ``probability`` -- shape ``(N_steps,)``
    - ``positions``  -- shape ``(N_steps, 3)``  (x, y, z)
    - ``velocities`` -- shape ``(N_steps, 3)``  (vx, vy, vz)

    Usage::

        ts, prob, pos, vel = result      # tuple unpacking

        result.pair("x", "y").plot()     # x vs y
        result.pair("t", "x").plot()     # t vs x
        result.tx.plot()                 # shorthand for pair("t", "x")
        result.yvz.plot()               # shorthand for pair("y", "vz")
    """

    # Supported variable keys
    _VALID_KEYS = {"t", "x", "y", "z", "vx", "vy", "vz"}

    def __init__(
        self,
        ts: np.ndarray,
        probability: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        unit=None,
    ):
        """
        Parameters
        ----------
        ts : numpy.ndarray, shape = (N_steps,)
        probability : numpy.ndarray, shape = (N_steps,)
        positions : numpy.ndarray, shape = (N_steps, 3)
        velocities : numpy.ndarray, shape = (N_steps, 3)
        """
        N = ts.shape[0]
        if positions.ndim != 2 or positions.shape != (N, 3):
            raise ValueError("positions must be an array of shape (N_steps, 3)")
        if velocities.ndim != 2 or velocities.shape != (N, 3):
            raise ValueError("velocities must be an array of shape (N_steps, 3)")

        self.ts = ts
        self.probability = probability
        self.positions = positions
        self.velocities = velocities

        self.unit = unit

    def __iter__(self) -> Iterator[Any]:
        """Support tuple unpacking: ``ts, prob, pos, vel = result``."""
        yield self.ts
        yield self.probability
        yield self.positions
        yield self.velocities

    def __repr__(self) -> str:
        """Return a string representation.

        Returns
        -------
        str
            Human-readable summary.
        """
        N = len(self.ts)
        return (
            f"<BacktraceResult: n_steps={N}, keys={list(BacktraceResult._VALID_KEYS)}> "
            f"(use result.pair(var1,var2) to get XYData)"
        )

    def pair(self, var1: str, var2: str) -> XYData:
        """Extract two variables and return an XYData.

        Parameters *var1* and *var2* must each be one of
        ``'t'``, ``'x'``, ``'y'``, ``'z'``, ``'vx'``, ``'vy'``, ``'vz'``.

        Examples::

            result.pair("x", "y")   # x vs y
            result.pair("t", "x")   # t vs x
            result.pair("z", "vy")  # z vs vy
        """
        if var1 not in BacktraceResult._VALID_KEYS or var2 not in BacktraceResult._VALID_KEYS:
            raise KeyError(f"Allowed keys = {BacktraceResult._VALID_KEYS}, but got '{var1}', '{var2}'")

        def _get_array(key: str) -> np.ndarray:
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

                return self.ts, u

            elif key in {"x", "y", "z"}:
                idx = {"x": 0, "y": 1, "z": 2}[key]

                u = self.unit.length if self.unit else None

                return self.positions[:, idx], u

            elif key in {"vx", "vy", "vz"}:
                idx = {"vx": 0, "vy": 1, "vz": 2}[key]

                u = self.unit.v if self.unit else None

                return self.velocities[:, idx], u

            else:
                raise KeyError(f"Unexpected key: {key}")

        arr1, u1 = _get_array(var1)
        arr2, u2 = _get_array(var2)
        xlabel = var1
        ylabel = var2
        title = f"{var1} vs {var2}"

        return XYData(
            arr1,
            arr2,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            units=(u1, u2) if u1 else None,
        )

    def __getattr__(self, name: str) -> Any:
        """Interpret attribute access as a pair name.

        Matches the longest prefix in ``_VALID_KEYS`` and delegates to
        :meth:`pair`.

        Examples::

            result.tx   # -> pair("t", "x")
            result.tvx  # -> pair("t", "vx")
            result.xvy  # -> pair("x", "vy")
            result.yz   # -> pair("y", "z")
        """
        for key1 in BacktraceResult._VALID_KEYS:
            if name.startswith(key1):
                rest = name[len(key1) :]
                if rest in BacktraceResult._VALID_KEYS:
                    return self.pair(key1, rest)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
