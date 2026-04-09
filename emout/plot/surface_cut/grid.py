"""Uniform cell-centred grid definition for surface-cut operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class UniformCellCenteredGrid:
    """Uniform, cell-centered grid for data shaped (nz, ny, nx).

    Coordinates (cell centers):
      x(i) = x0 + (i + 0.5) * dx
      y(j) = y0 + (j + 0.5) * dy
      z(k) = z0 + (k + 0.5) * dz
    """

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    def x_centers(self) -> np.ndarray:
        """Return x-direction cell center coordinates.

        Returns
        -------
        np.ndarray
            Array of x center coordinates with length `nx`.
        """
        return self.x0 + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dx

    def y_centers(self) -> np.ndarray:
        """Return y-direction cell center coordinates.

        Returns
        -------
        np.ndarray
            Array of y center coordinates with length `ny`.
        """
        return self.y0 + (np.arange(self.ny, dtype=np.float64) + 0.5) * self.dy

    def z_centers(self) -> np.ndarray:
        """Return z-direction cell center coordinates.

        Returns
        -------
        np.ndarray
            Array of z center coordinates with length `nz`.
        """
        return self.z0 + (np.arange(self.nz, dtype=np.float64) + 0.5) * self.dz

    def extent_edges(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Return the grid edge extents (not cell centers).

        Returns
        -------
        tuple(tuple(float, float), tuple(float, float), tuple(float, float))
            `(x_min, x_max)`, `(y_min, y_max)`, `(z_min, z_max)`.
        """
        return (
            (self.x0, self.x0 + self.nx * self.dx),
            (self.y0, self.y0 + self.ny * self.dy),
            (self.z0, self.z0 + self.nz * self.dz),
        )
