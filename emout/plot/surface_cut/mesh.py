from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np


AxisSpec = Union[str, Tuple[float, float, float], np.ndarray]


def _as_3vec(v, *, name: str) -> np.ndarray:
    """Coerce ``v`` into a length-3 NumPy float vector or raise ``ValueError``."""
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError(f"{name} must be a 3-vector, got shape {a.shape}")
    return a


def _axis_to_unit(axis: AxisSpec) -> np.ndarray:
    """Normalise an axis specifier into a unit 3-vector.

    Accepts ``"x"`` / ``"y"`` / ``"z"`` (case-insensitive) or any 3-element
    array-like. Raises ``ValueError`` for unrecognised strings or zero
    vectors.
    """
    if isinstance(axis, str):
        s = axis.lower().strip()
        if s == "x":
            a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif s == "y":
            a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        elif s == "z":
            a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            raise ValueError("axis must be one of 'x','y','z' or a 3-vector.")
    else:
        a = _as_3vec(axis, name="axis")

    n = np.linalg.norm(a)
    if n == 0.0 or not np.isfinite(n):
        raise ValueError("axis must be a finite non-zero vector.")
    return a / n


BoxFaceName = Literal["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
CylinderPartName = Literal["side", "top", "bottom"]
HollowCylinderPartName = Literal["outer", "inner", "top", "bottom"]

_TWO_PI = 2.0 * np.pi
_FULL_ANGLE_EPS = 1e-9


class MeshSurface3D(ABC):
    """Explicit triangle mesh surface for `plot_surfaces`.

    Each subclass knows how to materialise a structured ``(V, F)`` triangle
    mesh for a specific shape (box, cylinder, sphere, …). Use ``+`` to
    compose multiple shapes into a :class:`CompositeMeshSurface` and
    :meth:`render` to wrap one in a :class:`RenderItem` ready for
    :func:`emout.plot.surface_cut.plot_surfaces`.
    """

    @abstractmethod
    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(V, F)` mesh arrays.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            `V` is `(n_vertices, 3)` in `(x, y, z)` order and `F` is
            `(n_faces, 3)` triangle indices.
        """
        raise NotImplementedError

    def __add__(self, other: "MeshSurface3D") -> "MeshSurface3D":
        if not isinstance(other, MeshSurface3D):
            return NotImplemented
        return CompositeMeshSurface([self, other])

    def __radd__(self, other):
        if other == 0:  # enables sum([...]) to start from 0
            return self
        return NotImplemented

    def render(self, **style_kwargs):
        """Wrap this mesh surface in a ``RenderItem`` ready for ``plot_surfaces``.

        Parameters
        ----------
        **style_kwargs
            Forwarded to :class:`emout.plot.surface_cut.viz.RenderItem`
            (e.g. ``style="solid"``, ``solid_color="0.7"``, ``alpha=0.5``).

        Returns
        -------
        RenderItem
            A render item wrapping ``self``.
        """
        from .viz import RenderItem

        return RenderItem(surface=self, **style_kwargs)


def _normalize_count(value: int, *, name: str, minimum: int) -> int:
    ivalue = int(value)
    if ivalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return ivalue


def _normalize_resolution(
    resolution: Union[int, Tuple[int, int]],
    *,
    name: str = "resolution",
    minimum: int = 2,
) -> Tuple[int, int]:
    if isinstance(resolution, int):
        nu = nv = int(resolution)
    else:
        nu, nv = resolution
        nu = int(nu)
        nv = int(nv)
    if nu < minimum or nv < minimum:
        raise ValueError(f"{name} entries must be >= {minimum}")
    return nu, nv


def _normalize_selection(
    value: Optional[Union[str, Sequence[str]]],
    *,
    allowed: Sequence[str],
    name: str,
) -> Tuple[str, ...]:
    allowed_lut = {item.lower(): item for item in allowed}
    if value is None:
        return tuple(allowed)

    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)

    selected: list[str] = []
    for item in values:
        key = str(item).lower()
        if key == "all":
            return tuple(allowed)
        if key not in allowed_lut:
            allowed_str = ", ".join(allowed)
            raise ValueError(f"{name} entries must be one of: {allowed_str}, all")
        normalized = allowed_lut[key]
        if normalized not in selected:
            selected.append(normalized)

    if not selected:
        raise ValueError(f"{name} must not be empty")
    return tuple(selected)


def _combine_meshes(meshes: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if not meshes:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for V, F in meshes:
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int64)
        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError("mesh vertices must have shape (n_vertices, 3)")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError("mesh faces must have shape (n_faces, 3)")
        if V.size == 0 or F.size == 0:
            continue
        vertices.append(V)
        faces.append(F + offset)
        offset += V.shape[0]

    if not vertices:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)
    return np.vstack(vertices), np.vstack(faces)


def _grid_faces(nv: int, nu: int, *, wrap_u: bool = False) -> np.ndarray:
    if nv < 2:
        raise ValueError("nv must be >= 2")
    if wrap_u:
        if nu < 3:
            raise ValueError("nu must be >= 3 when wrap_u=True")
        ncell_u = nu
    else:
        if nu < 2:
            raise ValueError("nu must be >= 2")
        ncell_u = nu - 1

    faces = np.empty(((nv - 1) * ncell_u * 2, 3), dtype=np.int64)
    idx = 0
    for j in range(nv - 1):
        row0 = j * nu
        row1 = (j + 1) * nu
        for i in range(ncell_u):
            i1 = (i + 1) % nu if wrap_u else (i + 1)
            v00 = row0 + i
            v01 = row0 + i1
            v10 = row1 + i
            v11 = row1 + i1
            faces[idx] = (v00, v01, v11)
            faces[idx + 1] = (v00, v11, v10)
            idx += 2
    return faces


def _orient_faces_to_normal(
    V: np.ndarray,
    F: np.ndarray,
    *,
    expected_normal: Tuple[float, float, float],
) -> np.ndarray:
    if F.size == 0:
        return F
    normal = np.asarray(expected_normal, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm == 0.0:
        raise ValueError("expected_normal must be non-zero")
    normal /= norm

    tris = V[F[: min(32, len(F))]]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]).mean(axis=0)
    if np.dot(fn, normal) < 0.0:
        return F[:, [0, 2, 1]]
    return F


def _plane_mesh(
    points: np.ndarray,
    *,
    expected_normal: Tuple[float, float, float],
    wrap_u: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("points must have shape (nv, nu, 3)")
    nv, nu = points.shape[:2]
    V = points.reshape(-1, 3).astype(np.float64)
    F = _grid_faces(nv, nu, wrap_u=wrap_u)
    F = _orient_faces_to_normal(V, F, expected_normal=expected_normal)
    return V, F


def _center_to_3vec(
    center: Union[Tuple[float, float], Tuple[float, float, float], np.ndarray],
) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64).reshape(-1)
    if c.size == 2:
        c = np.array([c[0], c[1], 0.0], dtype=np.float64)
    if c.size != 3:
        raise ValueError("center must be a 2-tuple or 3-tuple")
    return c


def _orthonormal_frame(axis: AxisSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = _axis_to_unit(axis)

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, a)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    e1 = ref - np.dot(ref, a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    e2 /= np.linalg.norm(e2)
    return a, e1, e2


def _axial_range(
    *,
    length: Optional[float],
    tmin: Optional[float],
    tmax: Optional[float],
) -> Tuple[float, float]:
    if length is not None:
        if tmin is not None or tmax is not None:
            raise ValueError("Specify either length or tmin/tmax, not both")
        half = 0.5 * float(length)
        if half <= 0.0:
            raise ValueError("length must be > 0")
        return -half, half

    if tmin is None or tmax is None:
        raise ValueError("Finite mesh surfaces require length or both tmin/tmax")

    tmin = float(tmin)
    tmax = float(tmax)
    if not tmin < tmax:
        raise ValueError("Require tmin < tmax")
    return tmin, tmax


def _resolve_theta_range(
    theta_range: Optional[Tuple[float, float]],
) -> Tuple[float, float, bool]:
    """Return `(theta_min, theta_max, is_full)` for an optional angular range.

    ``theta_range=None`` means the full `[0, 2π)` circle. Otherwise the two
    values give the closed interval `[theta_min, theta_max]` in radians, and
    the third return value indicates whether the interval covers the full
    circle (in which case the angular dimension wraps and uses `endpoint=False`
    sampling).
    """
    if theta_range is None:
        return 0.0, _TWO_PI, True

    tmin, tmax = theta_range
    tmin = float(tmin)
    tmax = float(tmax)
    if not tmin < tmax:
        raise ValueError("theta_range must satisfy theta_min < theta_max")
    span = tmax - tmin
    if span > _TWO_PI + _FULL_ANGLE_EPS:
        raise ValueError("theta_range must span no more than 2π")
    is_full = abs(span - _TWO_PI) <= _FULL_ANGLE_EPS
    return tmin, tmax, is_full


def _sample_theta(
    ntheta: int,
    theta_min: float,
    theta_max: float,
    is_full: bool,
) -> np.ndarray:
    """Sample `ntheta` angular positions for mesh generation.

    When the range wraps (full circle) we exclude the endpoint so the grid
    joins cleanly. Otherwise we include both endpoints so the open slice has
    clean edges at `theta_min` and `theta_max`.
    """
    if is_full:
        return np.linspace(theta_min, theta_min + _TWO_PI, ntheta, endpoint=False, dtype=np.float64)
    return np.linspace(theta_min, theta_max, ntheta, endpoint=True, dtype=np.float64)


def _disc_mesh(
    base: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    *,
    radius: float,
    ntheta: int,
    nradial: int,
    expected_normal: Tuple[float, float, float],
    theta_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    theta_min, theta_max, is_full = _resolve_theta_range(theta_range)
    theta = _sample_theta(ntheta, theta_min, theta_max, is_full)
    n = theta.size
    ncell_u = n if is_full else n - 1
    if ncell_u < 1:
        raise ValueError("ntheta is too small for the requested theta_range")
    radii = np.linspace(0.0, radius, nradial, dtype=np.float64)

    vertices = [base[None, :]]
    ring_offsets: list[int] = []
    offset = 1
    ct = np.cos(theta)
    st = np.sin(theta)

    for r in radii[1:]:
        ring = base[None, :] + r * (ct[:, None] * e1[None, :] + st[:, None] * e2[None, :])
        vertices.append(ring)
        ring_offsets.append(offset)
        offset += n

    V = np.vstack(vertices).astype(np.float64)

    faces: list[Tuple[int, int, int]] = []
    first = ring_offsets[0]
    for i in range(ncell_u):
        i1 = (i + 1) % n if is_full else (i + 1)
        faces.append((0, first + i1, first + i))

    for layer in range(len(ring_offsets) - 1):
        off0 = ring_offsets[layer]
        off1 = ring_offsets[layer + 1]
        for i in range(ncell_u):
            i1 = (i + 1) % n if is_full else (i + 1)
            faces.append((off0 + i, off0 + i1, off1 + i1))
            faces.append((off0 + i, off1 + i1, off1 + i))

    F = np.asarray(faces, dtype=np.int64)
    F = _orient_faces_to_normal(V, F, expected_normal=expected_normal)
    return V, F


def _annulus_mesh(
    base: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    *,
    inner_radius: float,
    outer_radius: float,
    ntheta: int,
    nradial: int,
    expected_normal: Tuple[float, float, float],
    theta_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mesh a flat annulus between two concentric circles in the (e1, e2) plane."""
    theta_min, theta_max, is_full = _resolve_theta_range(theta_range)
    theta = _sample_theta(ntheta, theta_min, theta_max, is_full)
    radii = np.linspace(inner_radius, outer_radius, nradial, dtype=np.float64)
    R, T = np.meshgrid(radii, theta, indexing="ij")
    points = (
        base[None, None, :]
        + R[..., None] * np.cos(T)[..., None] * e1[None, None, :]
        + R[..., None] * np.sin(T)[..., None] * e2[None, None, :]
    )
    return _plane_mesh(points, expected_normal=expected_normal, wrap_u=is_full)


def _rect_with_hole_mesh(
    base: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    *,
    half_u: float,
    half_v: float,
    inner_radius: float,
    ntheta: int,
    nradial: int,
    expected_normal: Tuple[float, float, float],
    theta_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mesh a rectangle with a centered circular hole.

    The rectangle spans ``[-half_u, half_u] x [-half_v, half_v]`` in the
    ``(e1, e2)`` plane located at ``base``. The circular hole of
    ``inner_radius`` is centered on ``base`` and must fit entirely inside
    the rectangle.

    For each of the ``ntheta`` angular samples, a ray from the hole center
    is traced outward until it hits the rectangle boundary. ``nradial`` points
    are then placed linearly between the hole edge and that outer
    intersection, producing a structured ``(nradial, ntheta)`` grid that
    wraps in the angular direction when `theta_range` covers the full circle.
    """
    if inner_radius >= min(half_u, half_v):
        raise ValueError(
            "inner_radius must be < min(half_u, half_v) so the hole fits in the rectangle"
        )

    theta_min, theta_max, is_full = _resolve_theta_range(theta_range)
    theta = _sample_theta(ntheta, theta_min, theta_max, is_full)
    ct = np.cos(theta)
    st = np.sin(theta)

    eps = 1e-12
    tu = half_u / np.maximum(np.abs(ct), eps)
    tv = half_v / np.maximum(np.abs(st), eps)
    t_rect = np.minimum(tu, tv)

    inner_u = inner_radius * ct
    inner_v = inner_radius * st
    outer_u = t_rect * ct
    outer_v = t_rect * st

    alphas = np.linspace(0.0, 1.0, nradial, dtype=np.float64)
    u_grid = (1.0 - alphas)[:, None] * inner_u[None, :] + alphas[:, None] * outer_u[None, :]
    v_grid = (1.0 - alphas)[:, None] * inner_v[None, :] + alphas[:, None] * outer_v[None, :]

    points = (
        base[None, None, :]
        + u_grid[..., None] * e1[None, None, :]
        + v_grid[..., None] * e2[None, None, :]
    )
    return _plane_mesh(points, expected_normal=expected_normal, wrap_u=is_full)


class BoxMeshSurface(MeshSurface3D):
    """Explicit box faces selectable by name."""

    _allowed_faces = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
        *,
        faces: Optional[Union[BoxFaceName, Sequence[BoxFaceName]]] = None,
        resolution: Union[int, Tuple[int, int]] = (2, 2),
    ):
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.zmin = float(zmin)
        self.zmax = float(zmax)
        if not (self.xmin <= self.xmax and self.ymin <= self.ymax and self.zmin <= self.zmax):
            raise ValueError("Require xmin<=xmax, ymin<=ymax, zmin<=zmax")

        self.faces = _normalize_selection(faces, allowed=self._allowed_faces, name="faces")
        self.resolution = _normalize_resolution(resolution)

    def _face_mesh(self, face: str) -> Tuple[np.ndarray, np.ndarray]:
        nu, nv = self.resolution
        xs = np.linspace(self.xmin, self.xmax, nu, dtype=np.float64)
        ys = np.linspace(self.ymin, self.ymax, nu, dtype=np.float64)
        zs = np.linspace(self.zmin, self.zmax, nv, dtype=np.float64)

        if face == "xmin":
            Y, Z = np.meshgrid(
                np.linspace(self.ymin, self.ymax, nu, dtype=np.float64),
                np.linspace(self.zmin, self.zmax, nv, dtype=np.float64),
                indexing="xy",
            )
            X = np.full_like(Y, self.xmin)
            pts = np.stack([X, Y, Z], axis=-1)
            return _plane_mesh(pts, expected_normal=(-1.0, 0.0, 0.0))
        if face == "xmax":
            Y, Z = np.meshgrid(
                np.linspace(self.ymin, self.ymax, nu, dtype=np.float64),
                np.linspace(self.zmin, self.zmax, nv, dtype=np.float64),
                indexing="xy",
            )
            X = np.full_like(Y, self.xmax)
            pts = np.stack([X, Y, Z], axis=-1)
            return _plane_mesh(pts, expected_normal=(1.0, 0.0, 0.0))
        if face == "ymin":
            X, Z = np.meshgrid(
                np.linspace(self.xmin, self.xmax, nu, dtype=np.float64),
                np.linspace(self.zmin, self.zmax, nv, dtype=np.float64),
                indexing="xy",
            )
            Y = np.full_like(X, self.ymin)
            pts = np.stack([X, Y, Z], axis=-1)
            return _plane_mesh(pts, expected_normal=(0.0, -1.0, 0.0))
        if face == "ymax":
            X, Z = np.meshgrid(
                np.linspace(self.xmin, self.xmax, nu, dtype=np.float64),
                np.linspace(self.zmin, self.zmax, nv, dtype=np.float64),
                indexing="xy",
            )
            Y = np.full_like(X, self.ymax)
            pts = np.stack([X, Y, Z], axis=-1)
            return _plane_mesh(pts, expected_normal=(0.0, 1.0, 0.0))
        if face == "zmin":
            X, Y = np.meshgrid(
                np.linspace(self.xmin, self.xmax, nu, dtype=np.float64),
                np.linspace(self.ymin, self.ymax, nv, dtype=np.float64),
                indexing="xy",
            )
            Z = np.full_like(X, self.zmin)
            pts = np.stack([X, Y, Z], axis=-1)
            return _plane_mesh(pts, expected_normal=(0.0, 0.0, -1.0))
        if face == "zmax":
            X, Y = np.meshgrid(
                np.linspace(self.xmin, self.xmax, nu, dtype=np.float64),
                np.linspace(self.ymin, self.ymax, nv, dtype=np.float64),
                indexing="xy",
            )
            Z = np.full_like(X, self.zmax)
            pts = np.stack([X, Y, Z], axis=-1)
            return _plane_mesh(pts, expected_normal=(0.0, 0.0, 1.0))
        raise ValueError(f"Unsupported face: {face}")

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        return _combine_meshes([self._face_mesh(face) for face in self.faces])


class CylinderMeshSurface(MeshSurface3D):
    """Finite cylinder surface with selectable side/top/bottom parts.

    Optionally a ``theta_range`` may be given to take an angular subrange of
    the cylinder (e.g. a half section). A full ``2π`` range (the default)
    yields a closed mesh.
    """

    _allowed_parts = ("side", "top", "bottom")

    def __init__(
        self,
        center: Union[Tuple[float, float], Tuple[float, float, float], np.ndarray],
        axis: AxisSpec,
        radius: float,
        *,
        length: Optional[float] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        parts: Optional[Union[CylinderPartName, Sequence[CylinderPartName]]] = None,
        ntheta: int = 64,
        naxial: int = 2,
        nradial: int = 8,
        theta_range: Optional[Tuple[float, float]] = None,
    ):
        self.center = _center_to_3vec(center)
        self.axis, self.e1, self.e2 = _orthonormal_frame(axis)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")

        self.tmin, self.tmax = _axial_range(length=length, tmin=tmin, tmax=tmax)
        self.parts = _normalize_selection(parts, allowed=self._allowed_parts, name="parts")
        self.ntheta = _normalize_count(ntheta, name="ntheta", minimum=3)
        self.naxial = _normalize_count(naxial, name="naxial", minimum=2)
        self.nradial = _normalize_count(nradial, name="nradial", minimum=2)
        self.theta_range = theta_range
        self._theta_min, self._theta_max, self._theta_full = _resolve_theta_range(theta_range)

    def _side_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        theta = _sample_theta(self.ntheta, self._theta_min, self._theta_max, self._theta_full)
        t = np.linspace(self.tmin, self.tmax, self.naxial, dtype=np.float64)
        T, TH = np.meshgrid(t, theta, indexing="ij")
        points = (
            self.center[None, None, :]
            + T[..., None] * self.axis[None, None, :]
            + self.radius
            * (
                np.cos(TH)[..., None] * self.e1[None, None, :]
                + np.sin(TH)[..., None] * self.e2[None, None, :]
            )
        )
        return _plane_mesh(points, expected_normal=tuple(self.e1), wrap_u=self._theta_full)

    def _cap_mesh(self, which: str) -> Tuple[np.ndarray, np.ndarray]:
        t = self.tmax if which == "top" else self.tmin
        normal = self.axis if which == "top" else -self.axis
        base = self.center + t * self.axis
        return _disc_mesh(
            base,
            self.e1,
            self.e2,
            radius=self.radius,
            ntheta=self.ntheta,
            nradial=self.nradial,
            expected_normal=tuple(normal),
            theta_range=self.theta_range,
        )

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        meshes: list[Tuple[np.ndarray, np.ndarray]] = []
        if "side" in self.parts:
            meshes.append(self._side_mesh())
        if "top" in self.parts:
            meshes.append(self._cap_mesh("top"))
        if "bottom" in self.parts:
            meshes.append(self._cap_mesh("bottom"))
        return _combine_meshes(meshes)


class HollowCylinderMeshSurface(MeshSurface3D):
    """Rectangular slab pierced by an axial cylindrical hole.

    Despite the historical name, this surface is *not* a hollow cylinder. It
    represents a rectangular block extending along ``axis`` with a circular
    hole of ``inner_radius`` drilled through the middle. Use it when you want
    to highlight "a rectangle with a centered circular hole" plus the hole
    wall, for example to visualize a probe aperture on a planar electrode.

    Geometry
    --------
    * The slab's cross section in the plane perpendicular to ``axis`` is a
      rectangle of size ``width`` (a single float yields a square; a
      ``(width_u, width_v)`` tuple yields an arbitrary rectangle). The
      rectangle is centered on ``center`` in the local ``(e1, e2)`` frame
      derived from ``axis``.
    * Axially the slab extends over ``length`` (centered on ``center``) or,
      equivalently, ``tmin..tmax``.
    * A cylindrical hole of ``inner_radius`` runs through the slab on the
      ``axis`` direction. It must fit inside the rectangle
      (``inner_radius < min(width_u, width_v) / 2``).

    Parts
    -----
    * ``"outer"``  – the 4 rectangular side walls of the slab.
    * ``"inner"``  – the cylindrical hole surface (inward-facing).
    * ``"top"``    – the cap at ``tmax``: rectangle with a central circular hole.
    * ``"bottom"`` – the cap at ``tmin``: rectangle with a central circular hole.

    Optional ``theta_range`` lets you take an angular subrange of every
    theta-parameterized part (inner hole and both caps). The rectangular side
    walls are unaffected — restrict them with ``parts`` if needed.
    """

    _allowed_parts = ("outer", "inner", "top", "bottom")

    def __init__(
        self,
        center: Union[Tuple[float, float], Tuple[float, float, float], np.ndarray],
        axis: AxisSpec,
        width: Union[float, Tuple[float, float]],
        inner_radius: float,
        *,
        length: Optional[float] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        parts: Optional[Union[HollowCylinderPartName, Sequence[HollowCylinderPartName]]] = None,
        ntheta: int = 64,
        naxial: int = 2,
        nradial: int = 8,
        nwall: int = 2,
        theta_range: Optional[Tuple[float, float]] = None,
    ):
        self.center = _center_to_3vec(center)
        self.axis, self.e1, self.e2 = _orthonormal_frame(axis)

        if isinstance(width, (int, float)):
            wu = wv = float(width)
        else:
            try:
                wu_raw, wv_raw = width  # type: ignore[misc]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "width must be a float or a (width_u, width_v) pair"
                ) from exc
            wu = float(wu_raw)
            wv = float(wv_raw)
        if wu <= 0.0 or wv <= 0.0:
            raise ValueError("width must be > 0")
        self.width_u = wu
        self.width_v = wv
        self.half_u = 0.5 * wu
        self.half_v = 0.5 * wv

        self.inner_radius = float(inner_radius)
        if self.inner_radius <= 0.0:
            raise ValueError("inner_radius must be > 0")
        if self.inner_radius >= min(self.half_u, self.half_v):
            raise ValueError(
                "inner_radius must be < half of the smaller rectangle side"
            )

        self.tmin, self.tmax = _axial_range(length=length, tmin=tmin, tmax=tmax)
        self.parts = _normalize_selection(parts, allowed=self._allowed_parts, name="parts")
        self.ntheta = _normalize_count(ntheta, name="ntheta", minimum=3)
        self.naxial = _normalize_count(naxial, name="naxial", minimum=2)
        self.nradial = _normalize_count(nradial, name="nradial", minimum=2)
        self.nwall = _normalize_count(nwall, name="nwall", minimum=2)
        self.theta_range = theta_range
        self._theta_min, self._theta_max, self._theta_full = _resolve_theta_range(theta_range)

    def _inner_side_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        theta = _sample_theta(self.ntheta, self._theta_min, self._theta_max, self._theta_full)
        t = np.linspace(self.tmin, self.tmax, self.naxial, dtype=np.float64)
        T, TH = np.meshgrid(t, theta, indexing="ij")
        points = (
            self.center[None, None, :]
            + T[..., None] * self.axis[None, None, :]
            + self.inner_radius
            * (
                np.cos(TH)[..., None] * self.e1[None, None, :]
                + np.sin(TH)[..., None] * self.e2[None, None, :]
            )
        )
        V, F = _plane_mesh(points, expected_normal=tuple(self.e1), wrap_u=self._theta_full)
        # Flip to face inward (towards the axis of the hole).
        F = F[:, [0, 2, 1]]
        return V, F

    def _wall_meshes(self) -> list:
        t = np.linspace(self.tmin, self.tmax, self.naxial, dtype=np.float64)

        def build(
            offset_vec: np.ndarray,
            tangent_half: float,
            normal_dir: np.ndarray,
            tangent_axis: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            s = np.linspace(-tangent_half, tangent_half, self.nwall, dtype=np.float64)
            T, S = np.meshgrid(t, s, indexing="ij")
            points = (
                self.center[None, None, :]
                + offset_vec[None, None, :]
                + T[..., None] * self.axis[None, None, :]
                + S[..., None] * tangent_axis[None, None, :]
            )
            return _plane_mesh(points, expected_normal=tuple(normal_dir), wrap_u=False)

        return [
            build(-self.half_u * self.e1, self.half_v, -self.e1, self.e2),
            build(+self.half_u * self.e1, self.half_v,  self.e1, self.e2),
            build(-self.half_v * self.e2, self.half_u, -self.e2, self.e1),
            build(+self.half_v * self.e2, self.half_u,  self.e2, self.e1),
        ]

    def _cap_mesh(self, which: str) -> Tuple[np.ndarray, np.ndarray]:
        t = self.tmax if which == "top" else self.tmin
        normal = self.axis if which == "top" else -self.axis
        base = self.center + t * self.axis
        return _rect_with_hole_mesh(
            base,
            self.e1,
            self.e2,
            half_u=self.half_u,
            half_v=self.half_v,
            inner_radius=self.inner_radius,
            ntheta=self.ntheta,
            nradial=self.nradial,
            expected_normal=tuple(normal),
            theta_range=self.theta_range,
        )

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        meshes: list = []
        if "outer" in self.parts:
            meshes.extend(self._wall_meshes())
        if "inner" in self.parts:
            meshes.append(self._inner_side_mesh())
        if "top" in self.parts:
            meshes.append(self._cap_mesh("top"))
        if "bottom" in self.parts:
            meshes.append(self._cap_mesh("bottom"))
        return _combine_meshes(meshes)


class RectangleMeshSurface(MeshSurface3D):
    """Single flat rectangular panel in 3D.

    The rectangle lies in the plane orthogonal to ``axis`` that passes
    through ``center``. Its in-plane frame ``(e1, e2)`` is derived from
    ``axis`` in the same way as cylinder surfaces, and the rectangle
    extends symmetrically about ``center`` with half-widths
    ``(width_u/2, width_v/2)``.

    Unlike ``BoxMeshSurface`` (which only produces axis-aligned faces), this
    class lets you place a single arbitrarily-oriented rectangular panel
    anywhere in space — convenient for highlighting a probe window, a
    transparent cutting plane, or any planar mark.
    """

    def __init__(
        self,
        center: Union[Tuple[float, float], Tuple[float, float, float], np.ndarray],
        axis: AxisSpec,
        width: Union[float, Tuple[float, float]],
        *,
        resolution: Union[int, Tuple[int, int]] = (2, 2),
        flip_normal: bool = False,
    ):
        self.center = _center_to_3vec(center)
        self.axis, self.e1, self.e2 = _orthonormal_frame(axis)

        if isinstance(width, (int, float)):
            wu = wv = float(width)
        else:
            try:
                wu_raw, wv_raw = width  # type: ignore[misc]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "width must be a float or a (width_u, width_v) pair"
                ) from exc
            wu = float(wu_raw)
            wv = float(wv_raw)
        if wu <= 0.0 or wv <= 0.0:
            raise ValueError("width must be > 0")
        self.width_u = wu
        self.width_v = wv
        self.half_u = 0.5 * wu
        self.half_v = 0.5 * wv

        self.resolution = _normalize_resolution(resolution)
        self.flip_normal = bool(flip_normal)

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        nu, nv = self.resolution
        u = np.linspace(-self.half_u, self.half_u, nu, dtype=np.float64)
        v = np.linspace(-self.half_v, self.half_v, nv, dtype=np.float64)
        V_arr, U_arr = np.meshgrid(v, u, indexing="ij")  # shapes (nv, nu)
        points = (
            self.center[None, None, :]
            + U_arr[..., None] * self.e1[None, None, :]
            + V_arr[..., None] * self.e2[None, None, :]
        )
        normal = -self.axis if self.flip_normal else self.axis
        return _plane_mesh(points, expected_normal=tuple(normal), wrap_u=False)


class SphereMeshSurface(MeshSurface3D):
    """Sphere mesh via latitude–longitude parameterization.

    The sphere of ``radius`` is centered on ``center``. The local poles lie
    on ``axis`` (default ``"z"``): latitude ``phi`` runs from ``-π/2`` at the
    south pole to ``+π/2`` at the north pole, and longitude ``theta`` wraps
    around in the ``(e1, e2)`` equatorial plane.

    Optional ``theta_range`` and ``phi_range`` let you carve out a spherical
    wedge, hemisphere, or any rectangular patch on the sphere. When a cut
    exposes the sphere's interior the open flat surfaces are *not* generated
    automatically; combine this class with other mesh surfaces if you need
    to cap them.
    """

    def __init__(
        self,
        center: Union[Tuple[float, float], Tuple[float, float, float], np.ndarray],
        radius: float,
        *,
        axis: AxisSpec = "z",
        ntheta: int = 48,
        nphi: int = 25,
        theta_range: Optional[Tuple[float, float]] = None,
        phi_range: Optional[Tuple[float, float]] = None,
    ):
        self.center = _center_to_3vec(center)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")
        self.axis, self.e1, self.e2 = _orthonormal_frame(axis)
        self.ntheta = _normalize_count(ntheta, name="ntheta", minimum=3)
        self.nphi = _normalize_count(nphi, name="nphi", minimum=2)

        self.theta_range = theta_range
        self._theta_min, self._theta_max, self._theta_full = _resolve_theta_range(theta_range)
        self.phi_range = phi_range
        self._phi_min, self._phi_max = self._resolve_phi_range(phi_range)

    @staticmethod
    def _resolve_phi_range(
        phi_range: Optional[Tuple[float, float]],
    ) -> Tuple[float, float]:
        half_pi = 0.5 * np.pi
        if phi_range is None:
            return -half_pi, half_pi
        pmin, pmax = phi_range
        pmin = float(pmin)
        pmax = float(pmax)
        if not pmin < pmax:
            raise ValueError("phi_range must satisfy phi_min < phi_max")
        if pmin < -half_pi - _FULL_ANGLE_EPS or pmax > half_pi + _FULL_ANGLE_EPS:
            raise ValueError("phi_range must lie within [-π/2, π/2]")
        return max(pmin, -half_pi), min(pmax, half_pi)

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        theta = _sample_theta(self.ntheta, self._theta_min, self._theta_max, self._theta_full)
        phi = np.linspace(self._phi_min, self._phi_max, self.nphi, dtype=np.float64)

        PHI, TH = np.meshgrid(phi, theta, indexing="ij")  # shapes (nphi, ntheta)
        cp = np.cos(PHI)
        sp = np.sin(PHI)
        ct = np.cos(TH)
        st = np.sin(TH)

        points = (
            self.center[None, None, :]
            + self.radius * cp[..., None] * ct[..., None] * self.e1[None, None, :]
            + self.radius * cp[..., None] * st[..., None] * self.e2[None, None, :]
            + self.radius * sp[..., None] * self.axis[None, None, :]
        )

        # _plane_mesh orients faces against a single reference normal, which
        # does not generalize to a closed sphere. We build the grid faces and
        # then flip them to face outward from `self.center` if needed.
        V = points.reshape(-1, 3).astype(np.float64)
        F = _grid_faces(self.nphi, self.ntheta, wrap_u=self._theta_full)
        if F.size:
            sample = F[: min(64, len(F))]
            tris = V[sample]
            face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            centroids = tris.mean(axis=1)
            outward = centroids - self.center[None, :]
            if float((face_normals * outward).sum()) < 0.0:
                F = F[:, [0, 2, 1]]
        return V, F


class CircleMeshSurface(MeshSurface3D):
    """Flat circular disc in a plane perpendicular to ``axis``.

    The disc is centered on ``center`` with the given ``radius``. Its
    orientation is controlled by ``axis`` (the outward normal). Use it for
    the MPIEMSES ``circlex``/``circley``/``circlez`` boundary primitives or
    whenever you need a single filled disk in 3D.
    """

    def __init__(
        self,
        center: Union[Tuple[float, float, float], np.ndarray],
        axis: AxisSpec,
        radius: float,
        *,
        ntheta: int = 64,
        nradial: int = 8,
        theta_range: Optional[Tuple[float, float]] = None,
        flip_normal: bool = False,
    ):
        self.center = _center_to_3vec(center)
        self.axis, self.e1, self.e2 = _orthonormal_frame(axis)
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")
        self.ntheta = _normalize_count(ntheta, name="ntheta", minimum=3)
        self.nradial = _normalize_count(nradial, name="nradial", minimum=2)
        self.theta_range = theta_range
        self.flip_normal = bool(flip_normal)

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        normal = -self.axis if self.flip_normal else self.axis
        return _disc_mesh(
            self.center,
            self.e1,
            self.e2,
            radius=self.radius,
            ntheta=self.ntheta,
            nradial=self.nradial,
            expected_normal=tuple(normal),
            theta_range=self.theta_range,
        )


class DiskMeshSurface(MeshSurface3D):
    """Annular disk (tube) with finite axial thickness.

    Represents the MPIEMSES ``diskx``/``disky``/``diskz`` finbound primitive:
    a ring with outer and inner cylindrical walls plus two annular end caps.
    Unlike :class:`HollowCylinderMeshSurface` (rectangular slab with a hole),
    the outer boundary here is also circular.
    """

    _allowed_parts = ("outer", "inner", "top", "bottom")

    def __init__(
        self,
        center: Union[Tuple[float, float], Tuple[float, float, float], np.ndarray],
        axis: AxisSpec,
        outer_radius: float,
        inner_radius: float,
        *,
        length: Optional[float] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        parts: Optional[Union[HollowCylinderPartName, Sequence[HollowCylinderPartName]]] = None,
        ntheta: int = 64,
        naxial: int = 2,
        nradial: int = 8,
        theta_range: Optional[Tuple[float, float]] = None,
    ):
        self.center = _center_to_3vec(center)
        self.axis, self.e1, self.e2 = _orthonormal_frame(axis)
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius)
        if self.outer_radius <= 0.0:
            raise ValueError("outer_radius must be > 0")
        if self.inner_radius <= 0.0:
            raise ValueError("inner_radius must be > 0")
        if not self.inner_radius < self.outer_radius:
            raise ValueError("Require inner_radius < outer_radius")

        self.tmin, self.tmax = _axial_range(length=length, tmin=tmin, tmax=tmax)
        self.parts = _normalize_selection(parts, allowed=self._allowed_parts, name="parts")
        self.ntheta = _normalize_count(ntheta, name="ntheta", minimum=3)
        self.naxial = _normalize_count(naxial, name="naxial", minimum=2)
        self.nradial = _normalize_count(nradial, name="nradial", minimum=2)
        self.theta_range = theta_range
        self._theta_min, self._theta_max, self._theta_full = _resolve_theta_range(theta_range)

    def _side_mesh(self, radius: float, *, inward: bool) -> Tuple[np.ndarray, np.ndarray]:
        theta = _sample_theta(self.ntheta, self._theta_min, self._theta_max, self._theta_full)
        t = np.linspace(self.tmin, self.tmax, self.naxial, dtype=np.float64)
        T, TH = np.meshgrid(t, theta, indexing="ij")
        points = (
            self.center[None, None, :]
            + T[..., None] * self.axis[None, None, :]
            + radius
            * (
                np.cos(TH)[..., None] * self.e1[None, None, :]
                + np.sin(TH)[..., None] * self.e2[None, None, :]
            )
        )
        V, F = _plane_mesh(points, expected_normal=tuple(self.e1), wrap_u=self._theta_full)
        if inward:
            F = F[:, [0, 2, 1]]
        return V, F

    def _cap_mesh(self, which: str) -> Tuple[np.ndarray, np.ndarray]:
        t = self.tmax if which == "top" else self.tmin
        normal = self.axis if which == "top" else -self.axis
        base = self.center + t * self.axis
        return _annulus_mesh(
            base,
            self.e1,
            self.e2,
            inner_radius=self.inner_radius,
            outer_radius=self.outer_radius,
            ntheta=self.ntheta,
            nradial=self.nradial,
            expected_normal=tuple(normal),
            theta_range=self.theta_range,
        )

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        meshes: list = []
        if "outer" in self.parts:
            meshes.append(self._side_mesh(self.outer_radius, inward=False))
        if "inner" in self.parts:
            meshes.append(self._side_mesh(self.inner_radius, inward=True))
        if "top" in self.parts:
            meshes.append(self._cap_mesh("top"))
        if "bottom" in self.parts:
            meshes.append(self._cap_mesh("bottom"))
        return _combine_meshes(meshes)


class PlaneWithCircleMeshSurface(MeshSurface3D):
    """Flat rectangular plane containing a central circular hole.

    Unlike :class:`HollowCylinderMeshSurface` — which is a *solid slab* with
    a cylindrical hole through it — this surface is a zero-thickness panel.
    It is the natural representation for the MPIEMSES finbound primitives
    ``plane-with-circlex``/``y``/``z``, where a single flat conducting
    plane has a round aperture cut out of it.

    The rectangle is centered on ``center``, lies in the plane perpendicular
    to ``axis`` and spans ``width`` (``(width_u, width_v)`` in the local
    frame). The circular hole of ``inner_radius`` is concentric with the
    rectangle and must satisfy
    ``inner_radius < min(width_u, width_v) / 2``.
    """

    def __init__(
        self,
        center: Union[Tuple[float, float, float], np.ndarray],
        axis: AxisSpec,
        width: Union[float, Tuple[float, float]],
        inner_radius: float,
        *,
        ntheta: int = 64,
        nradial: int = 8,
        theta_range: Optional[Tuple[float, float]] = None,
        flip_normal: bool = False,
    ):
        self.center = _center_to_3vec(center)
        self.axis, self.e1, self.e2 = _orthonormal_frame(axis)

        if isinstance(width, (int, float)):
            wu = wv = float(width)
        else:
            try:
                wu_raw, wv_raw = width  # type: ignore[misc]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "width must be a float or a (width_u, width_v) pair"
                ) from exc
            wu = float(wu_raw)
            wv = float(wv_raw)
        if wu <= 0.0 or wv <= 0.0:
            raise ValueError("width must be > 0")
        self.width_u = wu
        self.width_v = wv
        self.half_u = 0.5 * wu
        self.half_v = 0.5 * wv

        self.inner_radius = float(inner_radius)
        if self.inner_radius <= 0.0:
            raise ValueError("inner_radius must be > 0")
        if self.inner_radius >= min(self.half_u, self.half_v):
            raise ValueError(
                "inner_radius must be < half of the smaller rectangle side"
            )

        self.ntheta = _normalize_count(ntheta, name="ntheta", minimum=3)
        self.nradial = _normalize_count(nradial, name="nradial", minimum=2)
        self.theta_range = theta_range
        self.flip_normal = bool(flip_normal)

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        normal = -self.axis if self.flip_normal else self.axis
        return _rect_with_hole_mesh(
            self.center,
            self.e1,
            self.e2,
            half_u=self.half_u,
            half_v=self.half_v,
            inner_radius=self.inner_radius,
            ntheta=self.ntheta,
            nradial=self.nradial,
            expected_normal=tuple(normal),
            theta_range=self.theta_range,
        )


class CompositeMeshSurface(MeshSurface3D):
    """Composite mesh surface that concatenates child mesh surfaces.

    Construct with a sequence of :class:`MeshSurface3D` instances (or use the
    ``+`` operator on mesh surfaces). The composite's ``mesh()`` calls each
    child's ``mesh()`` and welds them into a single ``(V, F)`` pair using
    ``_combine_meshes``.
    """

    def __init__(self, children: Sequence["MeshSurface3D"]):
        flat: list["MeshSurface3D"] = []
        for child in children:
            if isinstance(child, CompositeMeshSurface):
                flat.extend(child.children)
            else:
                flat.append(child)
        self.children = tuple(flat)

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        return _combine_meshes([child.mesh() for child in self.children])


__all__ = [
    "MeshSurface3D",
    "BoxMeshSurface",
    "RectangleMeshSurface",
    "CircleMeshSurface",
    "CylinderMeshSurface",
    "HollowCylinderMeshSurface",
    "PlaneWithCircleMeshSurface",
    "DiskMeshSurface",
    "SphereMeshSurface",
    "CompositeMeshSurface",
]
