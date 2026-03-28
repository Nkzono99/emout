from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .sdf import AxisSpec, _axis_to_unit


BoxFaceName = Literal["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
CylinderPartName = Literal["side", "top", "bottom"]
HollowCylinderPartName = Literal["outer", "inner", "top", "bottom"]


class MeshSurface3D(ABC):
    """Explicit triangle mesh surface for `plot_surfaces`.

    Unlike `Surface3D`, this API represents already-selected faces directly.
    It is meant for cases where "which face to show" is more important than
    constructing a solid volume and extracting a cross section from it.
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


def _disc_mesh(
    base: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    *,
    radius: float,
    ntheta: int,
    nradial: int,
    expected_normal: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False, dtype=np.float64)
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
        offset += ntheta

    V = np.vstack(vertices).astype(np.float64)

    faces: list[Tuple[int, int, int]] = []
    first = ring_offsets[0]
    for i in range(ntheta):
        i1 = (i + 1) % ntheta
        faces.append((0, first + i1, first + i))

    for layer in range(len(ring_offsets) - 1):
        off0 = ring_offsets[layer]
        off1 = ring_offsets[layer + 1]
        for i in range(ntheta):
            i1 = (i + 1) % ntheta
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
) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False, dtype=np.float64)
    radii = np.linspace(inner_radius, outer_radius, nradial, dtype=np.float64)
    R, T = np.meshgrid(radii, theta, indexing="ij")
    points = (
        base[None, None, :]
        + R[..., None] * np.cos(T)[..., None] * e1[None, None, :]
        + R[..., None] * np.sin(T)[..., None] * e2[None, None, :]
    )
    return _plane_mesh(points, expected_normal=expected_normal, wrap_u=True)


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
    """Finite cylinder surface with selectable side/top/bottom parts."""

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

    def _side_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.linspace(0.0, 2.0 * np.pi, self.ntheta, endpoint=False, dtype=np.float64)
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
        return _plane_mesh(points, expected_normal=tuple(self.e1), wrap_u=True)

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
    """Finite hollow cylinder surface with selectable outer/inner/cap parts."""

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

    def _side_mesh(self, radius: float, *, inward: bool) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.linspace(0.0, 2.0 * np.pi, self.ntheta, endpoint=False, dtype=np.float64)
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
        V, F = _plane_mesh(points, expected_normal=tuple(self.e1), wrap_u=True)
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
        )

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        meshes: list[Tuple[np.ndarray, np.ndarray]] = []
        if "outer" in self.parts:
            meshes.append(self._side_mesh(self.outer_radius, inward=False))
        if "inner" in self.parts:
            meshes.append(self._side_mesh(self.inner_radius, inward=True))
        if "top" in self.parts:
            meshes.append(self._cap_mesh("top"))
        if "bottom" in self.parts:
            meshes.append(self._cap_mesh("bottom"))
        return _combine_meshes(meshes)


__all__ = [
    "MeshSurface3D",
    "BoxMeshSurface",
    "CylinderMeshSurface",
    "HollowCylinderMeshSurface",
]
