"""Concrete boundary classes for MPIEMSES finbound complex mode."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from emout.plot.surface_cut import (
    BoxMeshSurface,
    CircleMeshSurface,
    CylinderMeshSurface,
    DiskMeshSurface,
    MeshSurface3D,
    PlaneWithCircleMeshSurface,
    SphereMeshSurface,
)

from ._base import Boundary
from ._helpers import _domain_extent, _get_scalar, _get_vector


# ---------------------------------------------------------------------------
# Concrete boundary classes
# ---------------------------------------------------------------------------


class SphereBoundary(Boundary):
    """MPIEMSES ``sphere`` boundary → :class:`SphereMeshSurface`."""

    mesh_class = SphereMeshSurface

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        pt = self._ptcond()
        center = _get_vector(pt, "sphere_origin", self.fortran_index)
        radius = _get_scalar(pt, "sphere_radius", self.fortran_index)
        if center is None or radius is None:
            raise ValueError(f"sphere_origin/sphere_radius not set for boundary index {self.index}")
        center = np.asarray(center, dtype=np.float64)
        radius = float(radius)
        if use_si:
            center = self._to_si_length(center)
            radius = self._to_si_length(radius)
        return {"center": center, "radius": radius}

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        return SphereMeshSurface(**params)


class CuboidBoundary(Boundary):
    """MPIEMSES ``cuboid`` boundary → :class:`BoxMeshSurface`."""

    mesh_class = BoxMeshSurface

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        pt = self._ptcond()
        shape = _get_vector(pt, "cuboid_shape", self.fortran_index)
        if shape is None:
            raise ValueError(f"cuboid_shape not set for boundary index {self.index}")
        values = np.asarray([float(v) for v in shape], dtype=np.float64)
        if use_si:
            values = self._to_si_length(values)
        xmin, xmax, ymin, ymax, zmin, zmax = values.tolist()
        return {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
        }

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        return BoxMeshSurface(**params)


class RectangleBoundary(Boundary):
    """MPIEMSES ``rectangle`` boundary → single-face :class:`BoxMeshSurface`.

    The flat face is inferred from whichever of ``x``/``y``/``z`` has
    ``min == max``.
    """

    mesh_class = BoxMeshSurface

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        pt = self._ptcond()
        shape = _get_vector(pt, "rectangle_shape", self.fortran_index)
        if shape is None:
            raise ValueError(f"rectangle_shape not set for boundary index {self.index}")
        values = np.asarray([float(v) for v in shape], dtype=np.float64)
        if use_si:
            values = self._to_si_length(values)
        xmin, xmax, ymin, ymax, zmin, zmax = values.tolist()
        return {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
        }

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        xmin = params["xmin"]
        xmax = params["xmax"]
        ymin = params["ymin"]
        ymax = params["ymax"]
        zmin = params["zmin"]
        zmax = params["zmax"]

        tol = 1e-12
        face: Optional[str] = None
        if abs(xmax - xmin) <= tol:
            face = "xmin"
        elif abs(ymax - ymin) <= tol:
            face = "ymin"
        elif abs(zmax - zmin) <= tol:
            face = "zmin"
        kwargs = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
        }
        if face is not None:
            kwargs["faces"] = (face,)
        # Forward any resolution override the caller may have supplied.
        for extra in ("faces", "resolution"):
            if extra in params:
                kwargs[extra] = params[extra]
        return BoxMeshSurface(**kwargs)


class CircleBoundary(Boundary):
    """MPIEMSES ``circlex``/``circley``/``circlez`` → :class:`CircleMeshSurface`."""

    mesh_class = CircleMeshSurface

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        pt = self._ptcond()
        center = _get_vector(pt, "circle_origin", self.fortran_index)
        radius = _get_scalar(pt, "circle_radius", self.fortran_index)
        if center is None or radius is None:
            raise ValueError(f"circle_origin/circle_radius not set for boundary index {self.index}")
        center = np.asarray(center, dtype=np.float64)
        radius = float(radius)
        if use_si:
            center = self._to_si_length(center)
            radius = self._to_si_length(radius)
        return {
            "center": center,
            "axis": self.btype[-1],
            "radius": radius,
        }

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        return CircleMeshSurface(**params)


class CylinderBoundary(Boundary):
    """MPIEMSES ``cylinderx``/``y``/``z`` and ``open-cylinder*``.

    ``cylinder_origin`` holds the *lower* end of the cylinder along the
    chosen axis; we place that point at the mesh origin with ``tmin=0``,
    ``tmax=cylinder_height`` so the cylinder extends in the ``+axis``
    direction (matching the Fortran convention).
    """

    mesh_class = CylinderMeshSurface

    @property
    def is_open(self) -> bool:
        return self.btype.startswith("open-")

    @property
    def axis_letter(self) -> str:
        return self.btype[-1]

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        pt = self._ptcond()
        origin = _get_vector(pt, "cylinder_origin", self.fortran_index)
        radius = _get_scalar(pt, "cylinder_radius", self.fortran_index)
        height = _get_scalar(pt, "cylinder_height", self.fortran_index)
        if origin is None or radius is None or height is None:
            raise ValueError(f"cylinder_origin/cylinder_radius/cylinder_height not set for boundary index {self.index}")
        origin = np.asarray(origin, dtype=np.float64)
        radius = float(radius)
        height = float(height)
        if use_si:
            origin = self._to_si_length(origin)
            radius = self._to_si_length(radius)
            height = self._to_si_length(height)

        params: Dict[str, Any] = {
            "center": origin,
            "axis": self.axis_letter,
            "radius": radius,
            "tmin": 0.0,
            "tmax": height,
        }
        if self.is_open:
            params["parts"] = ("side",)
        return params

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        return CylinderMeshSurface(**params)


class DiskBoundary(Boundary):
    """MPIEMSES ``diskx``/``y``/``z`` annular disk → :class:`DiskMeshSurface`."""

    mesh_class = DiskMeshSurface

    @property
    def axis_letter(self) -> str:
        return self.btype[-1]

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        pt = self._ptcond()
        origin = _get_vector(pt, "disk_origin", self.fortran_index)
        outer = _get_scalar(pt, "disk_radius", self.fortran_index)
        inner = _get_scalar(pt, "disk_inner_radius", self.fortran_index)
        height = _get_scalar(pt, "disk_height", self.fortran_index)
        if origin is None or outer is None or inner is None or height is None:
            raise ValueError(
                f"disk_origin/disk_radius/disk_inner_radius/disk_height not set for boundary index {self.index}"
            )
        origin = np.asarray(origin, dtype=np.float64)
        outer = float(outer)
        inner = float(inner)
        height = float(height)
        if use_si:
            origin = self._to_si_length(origin)
            outer = self._to_si_length(outer)
            inner = self._to_si_length(inner)
            height = self._to_si_length(height)

        return {
            "center": origin,
            "axis": self.axis_letter,
            "outer_radius": outer,
            "inner_radius": inner,
            "tmin": 0.0,
            "tmax": height,
        }

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        return DiskMeshSurface(**params)


class PlaneWithCircleBoundary(Boundary):
    """MPIEMSES ``plane-with-circlex``/``y``/``z`` → :class:`PlaneWithCircleMeshSurface`.

    These are single flat planes perpendicular to the chosen axis with a
    circular aperture at ``plane_with_circle_origin``. The plane itself is
    drawn large enough to cover the simulation domain (``nx``/``ny``/``nz``
    grid extents).
    """

    mesh_class = PlaneWithCircleMeshSurface

    @property
    def axis_letter(self) -> str:
        return self.btype[-1]

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        pt = self._ptcond()
        origin = _get_vector(pt, "plane_with_circle_origin", self.fortran_index)
        radius = _get_scalar(pt, "plane_with_circle_radius", self.fortran_index)
        if origin is None or radius is None:
            raise ValueError(f"plane_with_circle_origin/radius not set for boundary index {self.index}")
        origin = np.asarray(origin, dtype=np.float64)
        radius = float(radius)

        nx, ny, nz = _domain_extent(self.inp)
        axis = self.axis_letter
        if axis == "x":
            width = (float(ny), float(nz))
        elif axis == "y":
            width = (float(nx), float(nz))
        else:  # z
            width = (float(nx), float(ny))

        if use_si:
            origin = self._to_si_length(origin)
            radius = self._to_si_length(radius)
            width = (
                float(self._to_si_length(width[0])),
                float(self._to_si_length(width[1])),
            )

        return {
            "center": origin,
            "axis": axis,
            "width": width,
            "inner_radius": radius,
        }

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        return PlaneWithCircleMeshSurface(**params)
