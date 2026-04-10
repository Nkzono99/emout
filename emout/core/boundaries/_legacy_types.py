"""Legacy single-body boundary types for MPIEMSES finbound mode."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from emout.plot.surface_cut import (
    BoxMeshSurface,
    CylinderMeshSurface,
    MeshSurface3D,
    RectangleMeshSurface,
)

from ._base import Boundary
from ._helpers import _domain_extent, _safe_attr


class FlatSurfaceBoundary(Boundary):
    """MPIEMSES legacy ``flat-surface`` → :class:`RectangleMeshSurface`.

    Reads the scalar ``zssurf`` (and domain extents ``nx``/``ny``) to build
    a flat rectangular plane at ``z = zssurf`` that spans the entire
    horizontal domain.
    """

    mesh_class = RectangleMeshSurface

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        zssurf = _safe_attr(self.inp, "zssurf")
        if zssurf is None:
            raise ValueError("flat-surface requires `zssurf` in plasma.inp")
        zssurf = float(zssurf)

        nx, ny, _nz = _domain_extent(self.inp)
        center = np.array([0.5 * nx, 0.5 * ny, zssurf], dtype=np.float64)
        width = (float(nx), float(ny))

        if use_si:
            center = self._to_si_length(center)
            width = (
                float(self._to_si_length(width[0])),
                float(self._to_si_length(width[1])),
            )

        return {
            "center": center,
            "axis": "z",
            "width": width,
        }

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        return RectangleMeshSurface(**params)


def _rectangle_hole_bounds(inp) -> Dict[str, float]:
    """Read `[x/y][l/u]rechole` + `zssurf`/`zlrechole` as a flat bounds dict.

    Matches Fortran ``add_rectangle_hole_surface`` in
    ``src/physics/collision/surfaces.F90``:

        xl = xlrechole(1),  xu = xurechole(1),
        yl = ylrechole(1),  yu = yurechole(1),
        zl = zlrechole(2),  zu = zssurf
    """

    def _first_nonnone(name: str, start_from: int = 1):
        if name not in inp.nml["ptcond"]:
            return None
        arr = inp.nml["ptcond"][name]
        si = inp.nml["ptcond"].start_index.get(name) if hasattr(inp.nml["ptcond"], "start_index") else None
        base = si[0] if si and si[0] is not None else 1
        target = max(start_from, base)
        for ib_fortran in range(target, base + len(arr)):
            v = arr[ib_fortran - base]
            if v is not None:
                return float(v)
        return None

    xl = _first_nonnone("xlrechole")
    xu = _first_nonnone("xurechole")
    yl = _first_nonnone("ylrechole")
    yu = _first_nonnone("yurechole")
    # zl is conventionally stored at Fortran index 2 in legacy *-hole modes
    # (zlrechole(1) is reserved for the top surface, matching zssurf).
    zl = _first_nonnone("zlrechole", start_from=2)
    if zl is None:
        zl = _first_nonnone("zlrechole")
    zssurf = _safe_attr(inp, "zssurf")
    if None in (xl, xu, yl, yu, zl, zssurf):
        raise ValueError(
            "rectangle/cylinder-hole requires xlrechole/xurechole/ylrechole/"
            "yurechole/zlrechole and zssurf in plasma.inp"
        )
    return {
        "xl": xl,
        "xu": xu,
        "yl": yl,
        "yu": yu,
        "zl": zl,
        "zu": float(zssurf),
    }


class RectangleHoleBoundary(Boundary):
    """MPIEMSES legacy ``rectangle-hole`` → pit-shaped box.

    The Fortran implementation in ``surfaces.F90`` assembles the hole from
    nine rectangles (bottom of the pit, four walls, and four top-plane
    pieces surrounding the opening). For visualization purposes this class
    returns the rectangular pit itself as a :class:`BoxMeshSurface` with
    the top face omitted — that is the opening — which is the most useful
    shape to overlay on a field plot.
    """

    mesh_class = BoxMeshSurface

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        bounds = _rectangle_hole_bounds(self.inp)
        coords = np.array(
            [bounds["xl"], bounds["xu"], bounds["yl"], bounds["yu"], bounds["zl"], bounds["zu"]],
            dtype=np.float64,
        )
        if use_si:
            coords = self._to_si_length(coords)
        xl, xu, yl, yu, zl, zu = coords.tolist()
        return {
            "xmin": xl,
            "xmax": xu,
            "ymin": yl,
            "ymax": yu,
            "zmin": zl,
            "zmax": zu,
            "faces": ("xmin", "xmax", "ymin", "ymax", "zmin"),
        }

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        return BoxMeshSurface(**params)


class CylinderHoleBoundary(Boundary):
    """MPIEMSES legacy ``cylinder-hole`` → cylindrical pit.

    The Fortran implementation uses the same ``xlrechole``/``xurechole``…
    bounds as :class:`RectangleHoleBoundary` but treats them as defining
    the cylindrical radius (half of ``xu - xl``). We return a
    :class:`CylinderMeshSurface` with only the ``side`` and ``bottom``
    parts, matching the Fortran wall + bottom-circle construction.
    """

    mesh_class = CylinderMeshSurface

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        bounds = _rectangle_hole_bounds(self.inp)
        xl, xu = bounds["xl"], bounds["xu"]
        yl, yu = bounds["yl"], bounds["yu"]
        zl, zu = bounds["zl"], bounds["zu"]
        center = np.array([0.5 * (xl + xu), 0.5 * (yl + yu), zl], dtype=np.float64)
        radius = 0.5 * (xu - xl)
        height = zu - zl
        if use_si:
            center = self._to_si_length(center)
            radius = float(self._to_si_length(radius))
            height = float(self._to_si_length(height))
        return {
            "center": center,
            "axis": "z",
            "radius": radius,
            "tmin": 0.0,
            "tmax": height,
            "parts": ("side", "bottom"),
        }

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        return CylinderMeshSurface(**params)
