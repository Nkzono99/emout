from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

try:
    from skimage.measure import marching_cubes
except ImportError as e:
    raise ImportError(
        "scikit-image is required for marching cubes. Install: pip install scikit-image"
    ) from e


Mode = Literal["cmap", "cont", "cmap+cont"]


@dataclass(frozen=True)
class Bounds3D:
    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]

    def expanded(self, frac: float = 0.05) -> "Bounds3D":
        def ex(a, b):
            w = b - a
            return (a - frac * w, b + frac * w)

        return Bounds3D(ex(*self.x), ex(*self.y), ex(*self.z))


def _as_list(surfaces) -> list:
    if isinstance(surfaces, (list, tuple)):
        return list(surfaces)
    return [surfaces]


def _make_norm(
    vmin=None, vmax=None, *, robust_data: Optional[np.ndarray] = None
) -> Normalize:
    if vmin is None or vmax is None:
        if robust_data is None:
            vmin0, vmax0 = -1.0, 1.0
        else:
            d = np.asarray(robust_data, dtype=float)
            d = d[np.isfinite(d)]
            if d.size == 0:
                vmin0, vmax0 = -1.0, 1.0
            else:
                vmin0, vmax0 = np.percentile(d, [1, 99])
        vmin = vmin0 if vmin is None else vmin
        vmax = vmax0 if vmax is None else vmax
    return Normalize(vmin=vmin, vmax=vmax)


def _sample_sdf_on_grid(surface, bounds: Bounds3D, grid_shape: Tuple[int, int, int]):
    """Sample surface.sdf on a uniform grid in world coordinates.

    Returns
    -------
    xs, ys, zs : 1D arrays
    sdf_zyx : ndarray (nz,ny,nx)
    """
    nx, ny, nz = grid_shape  # (x,y,z)
    xs = np.linspace(bounds.x[0], bounds.x[1], nx)
    ys = np.linspace(bounds.y[0], bounds.y[1], ny)
    zs = np.linspace(bounds.z[0], bounds.z[1], nz)

    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    sdf = surface.sdf(X, Y, Z).astype(np.float64)
    return xs, ys, zs, sdf


def _mesh_from_sdf(xs, ys, zs, sdf_zyx, level: float = 0.0):
    """Marching cubes on sdf (z,y,x). Returns V (x,y,z) and F."""
    dx = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ys.size > 1 else 1.0
    dz = float(zs[1] - zs[0]) if zs.size > 1 else 1.0

    verts_zyx, faces, _, _ = marching_cubes(sdf_zyx, level=level, spacing=(dz, dy, dx))

    verts_zyx[:, 0] += zs[0]
    verts_zyx[:, 1] += ys[0]
    verts_zyx[:, 2] += xs[0]

    V = np.column_stack([verts_zyx[:, 2], verts_zyx[:, 1], verts_zyx[:, 0]]).astype(
        np.float64
    )
    F = faces.astype(np.int64)
    return V, F


def _face_values_from_vertex_values(F: np.ndarray, vval: np.ndarray) -> np.ndarray:
    return np.nanmean(vval[F], axis=1)


def _poly_collection(
    V: np.ndarray,
    F: np.ndarray,
    face_values: np.ndarray,
    *,
    cmap,
    norm: Normalize,
    alpha: float = 1.0,
) -> Poly3DCollection:
    tris = V[F]
    fc = cmap(norm(face_values))
    fc[:, 3] = np.where(np.isfinite(face_values), alpha, 0.0)
    poly = Poly3DCollection(tris, facecolors=fc, edgecolors="none")
    poly.set_alpha(None)
    return poly


def _contour_segments_on_mesh(
    V: np.ndarray, F: np.ndarray, vval: np.ndarray, levels: Sequence[float]
) -> list[np.ndarray]:
    """Generate contour line segments for scalar values on a triangular mesh."""
    segs: list[np.ndarray] = []

    p = V[F]
    s = vval[F]

    valid_face = np.all(np.isfinite(s), axis=1)
    p = p[valid_face]
    s = s[valid_face]

    p0, p1, p2 = p[:, 0, :], p[:, 1, :], p[:, 2, :]
    s0, s1, s2 = s[:, 0], s[:, 1], s[:, 2]

    edges = [(p0, s0, p1, s1), (p1, s1, p2, s2), (p2, s2, p0, s0)]

    for L in levels:
        inter_pts = []
        for pa, va, pb, vb in edges:
            den = vb - va
            cross = (den != 0.0) & ((va - L) * (vb - L) < 0.0)
            t = np.zeros_like(va, dtype=np.float64)
            t[cross] = (L - va[cross]) / den[cross]
            ip = pa + t[:, None] * (pb - pa)
            inter_pts.append((cross, ip))

        for fi in range(p0.shape[0]):
            pts = []
            for cross, ip in inter_pts:
                if cross[fi]:
                    pts.append(ip[fi])
            if len(pts) == 2:
                segs.append(np.stack([pts[0], pts[1]], axis=0))

    return segs


def plot_surfaces(
    ax,
    *,
    field,
    surfaces,
    bounds: Optional[Bounds3D] = None,
    grid_shape: Tuple[int, int, int] = (140, 140, 100),
    mode: Mode = "cmap+cont",
    cmap_name: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 1.0,
    contour_levels: Union[int, Sequence[float]] = 10,
    contour_color: str = "k",
    contour_lw: float = 0.8,
    sdf_level: float = 0.0,
):
    """Render implicit surfaces (including boolean composites) with colormap and contours."""

    g = field.grid
    if bounds is None:
        x_edges, y_edges, z_edges = g.extent_edges()
        bounds = Bounds3D(x_edges, y_edges, z_edges).expanded(0.05)

    cmap = cm.get_cmap(cmap_name)
    norm = _make_norm(vmin, vmax, robust_data=field.data)

    if isinstance(contour_levels, int):
        levels = np.linspace(norm.vmin, norm.vmax, contour_levels)
    else:
        levels = np.asarray(list(contour_levels), dtype=float)

    for srf in _as_list(surfaces):
        xs, ys, zs, sdf = _sample_sdf_on_grid(srf, bounds, grid_shape)
        V, F = _mesh_from_sdf(xs, ys, zs, sdf, level=sdf_level)

        vval = field.sample(V[:, 0], V[:, 1], V[:, 2])

        if mode in ("cmap", "cmap+cont"):
            face_val = _face_values_from_vertex_values(F, vval)
            poly = _poly_collection(V, F, face_val, cmap=cmap, norm=norm, alpha=alpha)
            ax.add_collection3d(poly)

        if mode in ("cont", "cmap+cont"):
            segs = _contour_segments_on_mesh(V, F, vval, levels)
            if segs:
                lc = Line3DCollection(segs, colors=contour_color, linewidths=contour_lw)
                ax.add_collection3d(lc)

    ax.set_xlim(*bounds.x)
    ax.set_ylim(*bounds.y)
    ax.set_zlim(*bounds.z)

    return cmap, norm


def add_colorbar(fig, ax, *, cmap, norm, label=r"$\phi$ (V)", fraction=0.03, pad=0.08):
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    cbar = fig.colorbar(m, ax=ax, fraction=fraction, pad=pad)
    cbar.set_label(label)
    return cbar
