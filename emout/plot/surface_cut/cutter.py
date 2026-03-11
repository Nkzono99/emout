from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from .field import Field3D
from .sdf import Surface3D


KeepSide = Literal["inside", "outside"]


class SurfaceCutter:
    """Clip or sample Field3D using a Surface3D."""

    def __init__(self, field: Field3D):
        """インスタンスを初期化する。
        
        Parameters
        ----------
        field : Field3D
            切り出し・補間対象となる 3D スカラー場です。
        """
        self.field = field
        self.grid = field.grid

    def clip_volume(
        self,
        surface: Surface3D,
        keep: KeepSide = "inside",
        fill_value: float = np.nan,
        *,
        chunk_k: int = 1,
    ) -> np.ndarray:
        """Clip the whole (nz,ny,nx) volume by surface.

        keep="inside": keep sdf<=0
        keep="outside": keep sdf>0

        chunk_k: process in z-chunks to control peak memory.
        """
        g = self.grid
        out = self.field.data.copy()

        xs = g.x_centers()
        ys = g.y_centers()
        zs = g.z_centers()
        X2, Y2 = np.meshgrid(xs, ys, indexing="xy")  # (ny,nx)

        if chunk_k <= 0:
            raise ValueError("chunk_k must be >= 1")

        for k0 in range(0, g.nz, chunk_k):
            k1 = min(g.nz, k0 + chunk_k)
            Zk = zs[k0:k1][:, None, None]  # (kchunk,1,1)
            sdf = surface.sdf(X2[None, :, :], Y2[None, :, :], Zk)

            if keep == "inside":
                mask_keep = sdf <= 0.0
            elif keep == "outside":
                mask_keep = sdf > 0.0
            else:
                raise ValueError("keep must be 'inside' or 'outside'")

            block = out[k0:k1, :, :]
            block[~mask_keep] = fill_value
            out[k0:k1, :, :] = block

        return out

    def sample_on_plane(
        self,
        plane: Literal["x", "y", "z"],
        value: float,
        ulim: Tuple[float, float],
        vlim: Tuple[float, float],
        nu: int = 400,
        nv: int = 300,
        *,
        fill_value: float = np.nan,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate field on a plane.

        plane="z": z=value, (u,v)=(x,y)
        plane="x": x=value, (u,v)=(y,z)
        plane="y": y=value, (u,v)=(x,z)

        Returns X,Y,Z,F arrays.
        """
        u = np.linspace(ulim[0], ulim[1], nu)
        v = np.linspace(vlim[0], vlim[1], nv)
        U, V = np.meshgrid(u, v, indexing="xy")

        if plane == "z":
            X, Y, Z = U, V, np.full_like(U, value)
        elif plane == "x":
            X, Y, Z = np.full_like(U, value), U, V
        elif plane == "y":
            X, Y, Z = U, np.full_like(U, value), V
        else:
            raise ValueError("plane must be 'x','y','z'")

        F = self.field.sample(X, Y, Z)
        F = np.where(np.isfinite(F), F, fill_value)
        return X, Y, Z, F

    def mask_slice(
        self,
        surface: Surface3D,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        F: np.ndarray,
        keep: KeepSide = "inside",
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """Apply surface mask to an already-sampled slice (X,Y,Z,F)."""
        sdf = surface.sdf(X, Y, Z)
        if keep == "inside":
            mk = sdf <= 0.0
        elif keep == "outside":
            mk = sdf > 0.0
        else:
            raise ValueError("keep must be 'inside' or 'outside'")
        out = np.array(F, copy=True)
        out[~mk] = fill_value
        return out
