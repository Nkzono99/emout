"""Single-timestep particle data container.

:class:`ParticleData` holds per-particle arrays (position, velocity, etc.)
loaded from ``p{species}{comp}*_{part}.h5`` files and provides a
:meth:`plot` shortcut for scatter / histogram visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ParticleData:
    """Single-timestep container for one particle-data component.

    Wraps a 1-D float array of per-particle values (e.g. x-positions or
    vx-velocities) with optional unit metadata and a :meth:`plot` helper.
    """
    values: np.ndarray
    valunit: Optional[Any] = None
    name: str = "value"

    def __post_init__(self):
        """Validate and normalise after initialisation."""
        self.values = np.asarray(self.values, dtype=float)
        self.values[self.values == -9999] = np.nan
        if self.values.ndim != 1:
            raise ValueError("ParticleData must be a 1D float array.")

    @property
    def val_si(self) -> "ParticleData":
        """Return a new ParticleData converted to SI units."""
        if self.valunit is None:
            raise ValueError("valunit is not set.")

        new_values = self.valunit.reverse(self.values)
        return ParticleData(new_values, valunit=self.valunit, name=self.name)

    # --- pandas bridge -----------------

    def to_series(self, index=None) -> pd.Series:
        """Convert to a :class:`pandas.Series`."""

        return pd.Series(self.values, index=index, name=self.name)

    def plot(self, *args, **kwargs):
        """Plot the particle data via :meth:`pandas.Series.plot`.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :meth:`pandas.Series.plot`
            (e.g. *kind*, *x*, *y*).
        **kwargs : dict
            Keyword arguments forwarded to :meth:`pandas.Series.plot`
            (e.g. *ax*, *figsize*, *title*, *xlabel*, *ylabel*, *grid*,
            *legend*, *color*, *style*, *xlim*, *ylim*).

        Returns
        -------
        matplotlib.axes.Axes or matplotlib.artist.Artist
            Plot object returned by pandas (depends on *kind*).
        """
        return self.to_series().plot(*args, **kwargs)

    def __len__(self):
        """Return the number of particles.

        Returns
        -------
        int
            Number of elements.
        """
        return len(self.values)

    def __repr__(self):
        """Return a string representation.

        Returns
        -------
        str
            Human-readable summary.
        """
        return f"ParticleData(name={self.name}, unit={self.valunit}, values={self.values})"
