from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


Mode = Literal["cmap", "cont", "cmap+cont"]


@dataclass(frozen=True)
class RenderItem:
    """Rendering specification for a mesh surface.

    style:
      - 'field': color by sampled field values (colormap)
      - 'solid': constant face color (e.g., gray cut faces)
    """

    surface: object
    style: Literal["field", "solid"] = "field"
    solid_color: Union[str, Tuple[float, float, float], Tuple[float, float, float, float]] = "0.6"
    alpha: Optional[float] = None
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


def _surface_mesh(surface) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ``(V, F)`` arrays from a :class:`MeshSurface3D`.

    Accepts anything with a callable ``mesh()`` method that returns either a
    ``(V, F)`` pair directly or another mesh-like object whose own ``mesh()``
    yields the pair (e.g. a :class:`Boundary` from
    :mod:`emout.emout.boundaries`, whose ``mesh()`` returns a
    ``MeshSurface3D``). One level of unwrapping is enough because
    :class:`CompositeMeshSurface` already returns ``(V, F)`` directly.
    """
    mesh_fn = getattr(surface, "mesh", None)
    if not callable(mesh_fn):
        raise TypeError(
            f"Expected a MeshSurface3D-like object with a .mesh() method, "
            f"got {type(surface).__name__}"
        )
    result = mesh_fn()
    # Unwrap one level so Boundary objects (mesh() → MeshSurface3D) work too.
    inner = getattr(result, "mesh", None)
    if callable(inner):
        result = inner()
    V, F = result
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    return V, F


def _bounds_from_mesh_surfaces(surfaces) -> Optional[Bounds3D]:
    """Infer plot bounds from explicit mesh surfaces."""
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for item in _as_list(surfaces):
        try:
            V, _ = _surface_mesh(item.surface)
        except TypeError:
            return None
        if V.size == 0:
            continue
        mins.append(np.nanmin(V, axis=0))
        maxs.append(np.nanmax(V, axis=0))

    if not mins:
        return None

    xyz_min = np.min(np.vstack(mins), axis=0)
    xyz_max = np.max(np.vstack(maxs), axis=0)
    return Bounds3D(
        (float(xyz_min[0]), float(xyz_max[0])),
        (float(xyz_min[1]), float(xyz_max[1])),
        (float(xyz_min[2]), float(xyz_max[2])),
    ).expanded(0.05)


def _face_normals_from_mesh(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Compute normalized face normals from triangle geometry."""
    tris = V[F]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm = np.linalg.norm(fn, axis=1)
    good = nrm > 0.0
    fn[good] /= nrm[good][:, None]
    return fn


def _clip_faces_to_bounds(V: np.ndarray, F: np.ndarray, bounds: Bounds3D) -> np.ndarray:
    """Return the subset of ``F`` whose centroids fall inside ``bounds``.

    This is a fast centroid-based clip rather than a true polygon-versus-AABB
    intersection: triangles straddling the bounding box are kept or dropped
    wholesale based on which side their centroid lies on. The result has
    slightly jagged edges right at the bounds but is visually acceptable for
    overlay plots and avoids any new vertex generation. Vertices are not
    compacted — unused entries simply stop being referenced.
    """
    if F.size == 0:
        return F
    tris = V[F]                         # (nF, 3, 3)
    centroids = tris.mean(axis=1)       # (nF, 3)
    cx, cy, cz = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    keep = (
        (cx >= bounds.x[0]) & (cx <= bounds.x[1])
        & (cy >= bounds.y[0]) & (cy <= bounds.y[1])
        & (cz >= bounds.z[0]) & (cz <= bounds.z[1])
    )
    return F[keep]


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
    field=None,
    surfaces,
    bounds: Optional[Bounds3D] = None,
    mode: Mode = "cmap+cont",
    cmap_name: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 1.0,
    contour_levels: Union[int, Sequence[float]] = 10,
    contour_color: str = "k",
    contour_lw: float = 0.8,
    contour_on_top: bool = True,
    contour_offset: float = 0.0,
    contour_side: str = "front",
    clip_to_bounds: bool = True,
):
    """Render explicit triangle mesh surfaces with optional colormap + contours.

    When ``clip_to_bounds`` is ``True`` (the default) every mesh is filtered
    against the resolved ``bounds`` (either the user-supplied box or the
    field/mesh-derived one) by dropping triangles whose centroid lies
    outside. Pass ``clip_to_bounds=False`` to disable the clip and render
    every mesh in full — useful when you intentionally want surfaces
    extending beyond the plot extent.

    Side effect: this function sets ``ax.computed_zorder = False`` so it
    can pin the merged polygon collection behind the contour line
    collections via explicit ``set_zorder``. mplot3d's automatic
    inter-collection depth sort relies on a single per-collection z value
    that goes through the camera projection, which is camera-dependent and
    causes contours to vanish behind the polygon they were extracted from
    at certain viewpoints. Disabling computed_zorder makes the contour
    layering camera-independent. Per-triangle depth sort *within* the
    merged polygon collection still runs (it lives in
    ``Poly3DCollection.do_3d_projection``) so interleaving meshes still
    render in the right order.

    ``contour_offset`` defaults to ``0.0`` because shifting contour
    segments along the source triangle's face normal **breaks line
    continuity on curved surfaces**: adjacent triangles share an edge
    but have different face normals, so the shared edge crossing gets
    pushed in two different directions and the contour fractures into
    per-triangle dashes. The polygon-vs-contour layering is already
    handled by the explicit ``set_zorder`` mechanism above, so an
    offset is not needed for visibility. Pass a small positive
    ``contour_offset`` only when working with strictly flat surfaces
    where every triangle shares the same normal.
    """

    items = _as_list(surfaces)

    g = None if field is None else field.grid
    if bounds is None:
        if g is not None:
            x_edges, y_edges, z_edges = g.extent_edges()
            bounds = Bounds3D(x_edges, y_edges, z_edges).expanded(0.05)
        else:
            bounds = _bounds_from_mesh_surfaces(items)
            if bounds is None:
                raise ValueError(
                    "bounds is required when field is None and the surfaces "
                    "do not expose vertex extents"
                )

    cmap = plt.get_cmap(cmap_name)
    norm = _make_norm(vmin, vmax, robust_data=None if field is None else field.data)

    if isinstance(contour_levels, int):
        levels = np.linspace(norm.vmin, norm.vmax, contour_levels)
    else:
        levels = np.asarray(list(contour_levels), dtype=float)

    # NOTE on rendering order:
    #   matplotlib's mplot3d uses a painter's algorithm. There are two
    #   levels of sorting:
    #
    #     (a) Across collections: each Poly3DCollection gets a single
    #         "depth" value (set_sort_zpos override, or by default the
    #         average camera-space z of its vertices). All collections
    #         are drawn back-to-front by that single value when
    #         ax.computed_zorder is True (the default).
    #
    #     (b) Within a single Poly3DCollection: every triangle is sorted
    #         independently by its own camera-space average z. This is
    #         the only level of sort that gives *correct* depth ordering
    #         when surfaces interpenetrate or interleave in 3D.
    #
    #   We merge every face from every input mesh into ONE
    #   Poly3DCollection (with per-face colors / edges / linewidths) so
    #   the per-triangle sort kicks in and the user gets the visually
    #   correct order from any viewpoint.
    #
    #   For the contour Line3DCollections vs. the merged polygon, we
    #   cannot rely on (a) because mpl's auto sort goes through the
    #   camera projection: at some camera angles the contours sort as
    #   "background" relative to the merged polygon's average z and
    #   vanish behind it. Instead we disable computed_zorder and pin
    #   the layering with explicit set_zorder values. The internal
    #   per-triangle sort inside the merged Poly3DCollection still runs
    #   (it lives inside do_3d_projection, which is called either way).
    ax.computed_zorder = False

    poly_V_chunks: list[np.ndarray] = []
    poly_F_chunks: list[np.ndarray] = []
    poly_fc_chunks: list[np.ndarray] = []
    poly_ec_chunks: list[np.ndarray] = []
    poly_lw_chunks: list[np.ndarray] = []
    v_offset = 0

    for srf in items:
        surface = srf.surface
        V, F = _surface_mesh(surface)
        if V.size == 0 or F.size == 0:
            continue

        if clip_to_bounds:
            F = _clip_faces_to_bounds(V, F, bounds)
            if F.size == 0:
                continue

        item_alpha = alpha if srf.alpha is None else float(srf.alpha)

        # Decide whether faces from this surface get drawn at all, and
        # compute the per-face color array (one rgba per triangle).
        draw_faces = False
        facecolors: Optional[np.ndarray] = None
        vval: Optional[np.ndarray] = None  # field samples for contour extraction

        if srf.style == "solid":
            draw_faces = True
            rgba = np.asarray(to_rgba(srf.solid_color), dtype=np.float64).copy()
            rgba[3] = item_alpha
            facecolors = np.tile(rgba, (F.shape[0], 1))
        elif srf.style == "field":
            if field is None:
                raise ValueError("field is required when style='field' or contours are requested")
            vval = field.sample(V[:, 0], V[:, 1], V[:, 2])
            if mode in ("cmap", "cmap+cont"):
                draw_faces = True
                face_val = _face_values_from_vertex_values(F, vval)
                facecolors = np.asarray(cmap(norm(face_val)), dtype=np.float64).copy()
                facecolors[:, 3] = np.where(np.isfinite(face_val), item_alpha, 0.0)
        else:
            raise ValueError(f"Unknown surface style {srf.style!r}")

        if draw_faces and facecolors is not None:
            nF = F.shape[0]
            if srf.edge_color is None:
                edgecolors = np.zeros((nF, 4), dtype=np.float64)
            else:
                edge_rgba = np.asarray(to_rgba(srf.edge_color), dtype=np.float64).copy()
                edgecolors = np.tile(edge_rgba, (nF, 1))
            linewidths = np.full(nF, float(srf.edge_lw), dtype=np.float64)

            poly_V_chunks.append(V)
            poly_F_chunks.append(F + v_offset)
            poly_fc_chunks.append(facecolors)
            poly_ec_chunks.append(edgecolors)
            poly_lw_chunks.append(linewidths)
            v_offset += V.shape[0]

        # Contour lines remain per-item — they live in a separate
        # Line3DCollection that gets pushed forward via sort_zpos.
        if srf.style == "field" and srf.draw_contours and mode in ("cont", "cmap+cont"):
            if vval is None:
                # Should be unreachable (vval is set in the field branch above),
                # but be defensive in case mode/style invariants change.
                vval = field.sample(V[:, 0], V[:, 1], V[:, 2])
            fn = _face_normals_from_mesh(V, F)

            # Optional back-face culling for contours (avoid seeing lines from the back side).
            if contour_side not in ("both", "front", "back"):
                raise ValueError('contour_side must be "both", "front", or "back"')
            F_contour = F
            if contour_side != "both" and len(F_contour) > 0:
                vdir = _view_dir_from_axes(ax)
                d = (fn[:, 0] * vdir[0] + fn[:, 1] * vdir[1] + fn[:, 2] * vdir[2])
                keep = d > 0.0 if contour_side == "front" else d < 0.0
                # If orientation is flipped, keep may become almost empty; auto-flip in that case.
                if keep.mean() < 0.05:
                    keep = ~keep
                F_contour = F_contour[keep]
                fn = fn[keep]

            segs = _contour_segments_on_mesh(
                V,
                F_contour,
                vval,
                levels,
                face_normals=fn,
                offset=contour_offset,
            )
            if segs:
                lc = Line3DCollection(
                    segs,
                    colors=contour_color,
                    linewidths=contour_lw,
                    # Round caps/joins matter a lot for thick contour
                    # lines on curved surfaces. mpl's defaults (butt /
                    # miter) leave a small triangular gap at every
                    # segment boundary, and the gaps look like the
                    # contour curve has been chopped into discrete
                    # tick marks. Round caps fill in those gaps so
                    # thick contours render as one continuous curve.
                    capstyle="round",
                    joinstyle="round",
                )
                # With ax.computed_zorder=False, set_zorder is honoured at
                # draw time. Contours sit above the merged polygon (which
                # gets zorder=1 below) when contour_on_top is True.
                lc.set_zorder(2 if contour_on_top else 1)
                ax.add_collection3d(lc)

    # ---- merged polygon collection ----
    if poly_V_chunks:
        V_all = np.vstack(poly_V_chunks).astype(np.float64)
        F_all = np.vstack(poly_F_chunks).astype(np.int64)
        fc_all = np.vstack(poly_fc_chunks)
        ec_all = np.vstack(poly_ec_chunks)
        lw_all = np.concatenate(poly_lw_chunks)

        tris = V_all[F_all]
        # If every requested edge is fully transparent, switch to "none" so
        # mpl skips the edge-drawing path entirely (slight speed win and a
        # cleaner output).
        if not np.any(ec_all[:, 3] > 0.0):
            edgecolors_arg = "none"
        else:
            edgecolors_arg = ec_all
        merged_poly = Poly3DCollection(
            tris,
            facecolors=fc_all,
            edgecolors=edgecolors_arg,
            linewidths=lw_all,
            # Disable antialiasing on the polygon faces. mpl's default
            # AA produces semi-transparent fringes around every triangle
            # which the user sees as the whole face being "a bit
            # transparent" even when the per-face alpha is 1.0. Crisp
            # edges are also a better fit for the painter's-algorithm
            # depth sort, since AA depends on draw order.
            antialiaseds=False,
        )
        # set_alpha(None) tells mpl NOT to multiply every face by a
        # global alpha — we want the per-face rgba in fc_all to be the
        # final value, including the explicit 1.0 cases.
        merged_poly.set_alpha(None)
        # Pin polygons below contour lines via explicit zorder
        # (computed_zorder is disabled at the top of this function).
        merged_poly.set_zorder(1)
        ax.add_collection3d(merged_poly)

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
