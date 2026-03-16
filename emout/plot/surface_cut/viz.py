from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

try:
    from skimage.measure import marching_cubes
except ImportError as e:
    raise ImportError(
        "scikit-image is required for marching cubes. Install: pip install scikit-image"
    ) from e


Mode = Literal["cmap", "cont", "cmap+cont"]


@dataclass(frozen=True)
class RenderItem:
    """Rendering specification for a surface.

    style:
      - 'field': color by sampled field values (colormap)
      - 'solid': constant face color (e.g., gray cut faces)

    mask:
      Optional solid/region used to *keep* only triangles whose centroid is inside (sdf<=0).
      This is useful to show cut-faces only where they are exposed by a cut-away.
    """

    surface: object
    style: Literal["field", "solid"] = "field"
    solid_color: Union[str, Tuple[float, float, float], Tuple[float, float, float, float]] = "0.6"
    alpha: Optional[float] = None
    mask: Optional[object] = None
    draw_contours: bool = True
    edge_color: Optional[str] = None
    edge_lw: float = 0.4


@dataclass(frozen=True)
class Bounds3D:
    """Bounds3D クラス。
    """
    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]

    def expanded(self, frac: float = 0.05) -> "Bounds3D":
        """境界を指定割合だけ拡張した Bounds を返す。
        
        Parameters
        ----------
        frac : float, optional
            区間内補間係数です。
        Returns
        -------
        "Bounds3D"
            処理結果です。
        """
        def ex(a, b):
            """1 次元区間を指定割合で拡張する。
            
            Parameters
            ----------
            a : object
                始点側の値です。
            b : object
                終点側の値です。
            Returns
            -------
            object
                処理結果です。
            """
            w = b - a
            return (a - frac * w, b + frac * w)

        return Bounds3D(ex(*self.x), ex(*self.y), ex(*self.z))


def _as_list(surfaces) -> list[RenderItem]:
    """Normalize inputs into a list[RenderItem]."""
    if isinstance(surfaces, (list, tuple)):
        items = list(surfaces)
    else:
        items = [surfaces]

    out: list[RenderItem] = []
    for it in items:
        if isinstance(it, RenderItem):
            out.append(it)
        else:
            out.append(RenderItem(surface=it))
    return out


def _make_norm(
    vmin=None, vmax=None, *, robust_data: Optional[np.ndarray] = None
) -> Normalize:
    """カラーマップ用の正規化オブジェクトを作成する。
    
    Parameters
    ----------
    vmin : object, optional
        表示範囲の最小値。
    vmax : object, optional
        表示範囲の最大値。
    robust_data : Optional[np.ndarray], optional
        外れ値除去に使う参照データです。
    Returns
    -------
    Normalize
        処理結果です。
    """
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
    """Marching cubes on sdf (z,y,x).

    Returns
    -------
    V : (nV,3) float64
        Vertices in world (x,y,z).
    F : (nF,3) int64
        Triangle indices.
    N : (nV,3) float64
        Vertex normals in world (x,y,z). Normal direction follows scikit-image
        convention: points toward increasing sdf values (for a true SDF, this is
        "outward").
    """
    dx = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ys.size > 1 else 1.0
    dz = float(zs[1] - zs[0]) if zs.size > 1 else 1.0

    verts_zyx, faces, normals_zyx, _ = marching_cubes(
        sdf_zyx, level=level, spacing=(dz, dy, dx)
    )

    verts_zyx[:, 0] += zs[0]
    verts_zyx[:, 1] += ys[0]
    verts_zyx[:, 2] += xs[0]

    V = np.column_stack([verts_zyx[:, 2], verts_zyx[:, 1], verts_zyx[:, 0]]).astype(np.float64)

    # normals_zyx are in (z,y,x) order -> convert to (x,y,z)
    N = np.column_stack([normals_zyx[:, 2], normals_zyx[:, 1], normals_zyx[:, 0]]).astype(np.float64)
    nrm = np.linalg.norm(N, axis=1)
    good = nrm > 0
    N[good] /= nrm[good][:, None]
    F = faces.astype(np.int64)
    return V, F, N


def _face_values_from_vertex_values(F: np.ndarray, vval: np.ndarray) -> np.ndarray:
    """頂点値から各面の代表値を計算する。
    
    Parameters
    ----------
    F : np.ndarray
        三角形面インデックス配列です。
    vval : np.ndarray
        頂点ごとのスカラー値です（`V` と同じ頂点順）。
    Returns
    -------
    np.ndarray
        各三角形面の平均スカラー値です。
    """
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
    """面値で着色したポリゴンコレクションを生成する。
    
    Parameters
    ----------
    V : np.ndarray
        頂点座標配列です。
    F : np.ndarray
        三角形面インデックス配列です。
    face_values : np.ndarray
        面ごとのスカラー値です。`F` の行順と対応します。
    cmap : object
        カラーマップ。
    norm : Normalize
        色正規化設定です。
    alpha : float, optional
        透過率です。
    Returns
    -------
    Poly3DCollection
        カラーマップを適用した三角形ポリゴンコレクションです。
    """
    tris = V[F]
    fc = cmap(norm(face_values))
    fc[:, 3] = np.where(np.isfinite(face_values), alpha, 0.0)
    poly = Poly3DCollection(tris, facecolors=fc, edgecolors="none")
    poly.set_alpha(None)
    return poly


def _solid_poly_collection(
    V: np.ndarray,
    F: np.ndarray,
    *,
    color,
    alpha: float = 1.0,
    edge_color: Optional[str] = None,
    edge_lw: float = 0.4,
) -> Poly3DCollection:
    """単色のポリゴンコレクションを生成する。
    
    Parameters
    ----------
    V : np.ndarray
        頂点座標配列です。
    F : np.ndarray
        三角形面インデックス配列です。
    color : object
        面色です。Matplotlib の色指定形式（名前、RGB(A)、HEX など）を受け付けます。
    alpha : float, optional
        透過率です。
    edge_color : Optional[str], optional
        輪郭線の色です。`None` の場合は輪郭線を描画しません。
    edge_lw : float, optional
        輪郭線の線幅です。
    Returns
    -------
    Poly3DCollection
        描画に使う三角形ポリゴンコレクションです。
    """
    tris = V[F]
    rgba = list(to_rgba(color))
    rgba[3] = float(alpha)
    poly = Poly3DCollection(
        tris,
        facecolors=[tuple(rgba)],
        edgecolors=("none" if edge_color is None else edge_color),
        linewidths=edge_lw,
    )
    return poly


def _filter_faces_by_mask(V: np.ndarray, F: np.ndarray, mask) -> np.ndarray:
    """Return faces to keep based on mask.sdf(centroid)<=0."""
    tris = V[F]
    c = tris.mean(axis=1)  # (nF,3)
    m = mask.sdf(c[:, 0], c[:, 1], c[:, 2])
    keep = np.asarray(m <= 0.0)
    return F[keep]




def _view_dir_from_axes(ax) -> np.ndarray:
    """Approximate view direction (from data toward the camera) in data coordinates.

    Uses mplot3d's azim/elev convention. This is only used for back-face culling
    of contour segments; it is evaluated at draw time (static view).
    """

    az = np.deg2rad(getattr(ax, "azim", -60.0) or -60.0)
    el = np.deg2rad(getattr(ax, "elev", 30.0) or 30.0)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], dtype=np.float64)

def _contour_segments_on_mesh(
    V: np.ndarray,
    F: np.ndarray,
    vval: np.ndarray,
    levels: Sequence[float],
    *,
    face_normals: Optional[np.ndarray] = None,
    offset: float = 0.0,
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

    # Optional tiny offset to pull the contour above the surface.
    # This reduces "z-fighting" and (more importantly in mplot3d)
    # helps the painter's algorithm draw the contour on top.
    fn = None
    if face_normals is not None and offset != 0.0:
        fn = np.asarray(face_normals, dtype=np.float64)
        if fn.shape[0] == F.shape[0]:
            fn = fn[valid_face]
        else:
            fn = None

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
                seg = np.stack([pts[0], pts[1]], axis=0)
                if fn is not None:
                    seg = seg + float(offset) * fn[fi][None, :]
                segs.append(seg)

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
    contour_on_top: bool = True,
    contour_offset: float = 0.0,
    contour_side: str = "front"
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

    for i, srf in enumerate(_as_list(surfaces)):
        surface = srf.surface
        xs, ys, zs, sdf = _sample_sdf_on_grid(surface, bounds, grid_shape)
        V, F, N = _mesh_from_sdf(xs, ys, zs, sdf, level=sdf_level)

        if srf.mask is not None:
            F = _filter_faces_by_mask(V, F, srf.mask)
            if F.size == 0:
                continue

        item_alpha = alpha if srf.alpha is None else float(srf.alpha)

        if srf.style == "solid":
            poly = _solid_poly_collection(
                V,
                F,
                color=srf.solid_color,
                alpha=item_alpha,
                edge_color=srf.edge_color,
                edge_lw=srf.edge_lw,
            )
            # poly.set_sort_zpos(bounds.z[0] - 1.0e9 + i*1e3)
            poly.set_sort_zpos(bounds.z[0] - 1.0e9 + i*1e3)
            ax.add_collection3d(poly)
            continue

        vval = field.sample(V[:, 0], V[:, 1], V[:, 2])

        if mode in ("cmap", "cmap+cont"):
            face_val = _face_values_from_vertex_values(F, vval)
            poly = _poly_collection(V, F, face_val, cmap=cmap, norm=norm, alpha=item_alpha)
            if contour_on_top and hasattr(poly, "set_sort_zpos"):
                # Push filled polygons slightly "behind" so contours can win in painter sort.
                poly.set_sort_zpos(bounds.z[0] - 1.0e9 + i*1e3)
            ax.add_collection3d(poly)

        if srf.draw_contours and mode in ("cont", "cmap+cont"):
            # Face normals from vertex normals.
            fn = N[F].mean(axis=1)
            nrm = np.linalg.norm(fn, axis=1)
            good = nrm > 0
            fn[good] /= nrm[good][:, None]

            # Optional back-face culling for contours (avoid seeing lines from the back side).
            if contour_side not in ("both", "front", "back"):
                raise ValueError('contour_side must be "both", "front", or "back"')
            if contour_side != "both" and len(F) > 0:
                vdir = _view_dir_from_axes(ax)
                d = (fn[:, 0] * vdir[0] + fn[:, 1] * vdir[1] + fn[:, 2] * vdir[2])
                keep = d > 0.0 if contour_side == "front" else d < 0.0
                # If orientation is flipped, keep may become almost empty; auto-flip in that case.
                if keep.mean() < 0.05:
                    keep = ~keep
                F = F[keep]
                fn = fn[keep]

            segs = _contour_segments_on_mesh(
                V,
                F,
                vval,
                levels,
                face_normals=fn,
                offset=contour_offset,
            )
            if segs:
                lc = Line3DCollection(segs, colors=contour_color, linewidths=contour_lw)
                if contour_on_top and hasattr(lc, "set_sort_zpos"):
                    lc.set_sort_zpos(bounds.z[1] + 1.0e9 + i*1e3)
                try:
                    lc.set_zorder(10)
                except Exception:
                    pass
                ax.add_collection3d(lc)

    ax.set_xlim(*bounds.x)
    ax.set_ylim(*bounds.y)
    ax.set_zlim(*bounds.z)

    return cmap, norm


def add_colorbar(fig, ax, *, cmap, norm, label=r"$\phi$ (V)", cax=None, fraction=0.03, pad=0.08):
    """描画に対応するカラーバーを追加する。
    
    Parameters
    ----------
    fig : object
        描画先の Figure。
    ax : object
        描画先の Axes。
    cmap : object
        カラーマップ。
    norm : object
        色正規化設定です。
    label : str, optional
        カラーバーのラベル文字列です。
    fraction : float, optional
        カラーバーの幅比率です（`Figure.colorbar` の `fraction`）。
    pad : float, optional
        プロット本体とカラーバーの間隔です（`Figure.colorbar` の `pad`）。
    Returns
    -------
    object
        追加したカラーバーオブジェクトです。
    """
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    cbar = fig.colorbar(m, ax=ax, fraction=fraction, pad=pad, cax=cax)
    cbar.set_label(label)
    return cbar
