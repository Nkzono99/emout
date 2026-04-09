"""3-D scalar field wrapper for surface-cut interpolation."""

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
        """インスタンスを初期化する。

        Parameters
        ----------
        grid : UniformCellCenteredGrid
            対象格子定義。`data_zyx` と同じ `(z, y, x)` 形状を持つ必要があります。
        data_zyx : np.ndarray
            セル中心値。形状は `(grid.nz, grid.ny, grid.nx)`。
        """
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
        """指定座標で 3D フィールドを補間サンプリングする。

        Parameters
        ----------
        x : np.ndarray
            x 座標配列。
        y : np.ndarray
            y 座標配列。
        z : np.ndarray
            z 座標配列。

        Returns
        -------
        np.ndarray
            `x`, `y`, `z` をブロードキャストした形状の補間値配列。
        """
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        pts = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=-1)  # (z,y,x)
        out = self._interp(pts).reshape(np.broadcast(x, y, z).shape)
        return out
