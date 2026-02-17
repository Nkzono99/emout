from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .grid import UniformCellCenteredGrid


class Field3D:
    """Holds (nz,ny,nx) cell-centered data and a SciPy interpolator.

    Note: RegularGridInterpolator expects axes in the same order as data indexing.
          Here we define axes as (z, y, x) to match data[k,j,i].
    """

    def __init__(self, grid: UniformCellCenteredGrid, data_zyx: np.ndarray):
        if data_zyx.shape != (grid.nz, grid.ny, grid.nx):
            raise ValueError(
                f"data shape must be (nz,ny,nx)=({grid.nz},{grid.ny},{grid.nx}), got {data_zyx.shape}"
            )
        self.grid = grid
        self.data = np.asarray(data_zyx, dtype=np.float64)

        zc = grid.z_centers()
        yc = grid.y_centers()
        xc = grid.x_centers()

        self._interp = RegularGridInterpolator(
            (zc, yc, xc),
            self.data,
            bounds_error=False,
            fill_value=np.nan,
        )

    def sample(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Sample field at world coords (x,y,z). Returns array of broadcasted shape."""
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        pts = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=-1)  # (z,y,x)
        out = self._interp(pts).reshape(np.broadcast(x, y, z).shape)
        return out
