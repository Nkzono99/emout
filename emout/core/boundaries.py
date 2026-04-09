"""Boundary wrappers around ``Emout.inp`` for MPIEMSES ``finbound`` mode.

This module exposes :class:`BoundaryCollection` and per-type :class:`Boundary`
subclasses that read their geometry parameters from ``data.inp`` (the
``&ptcond`` namelist) and build a :class:`MeshSurface3D` on demand.

Only the finbound "complex" mode is supported — legacy ``boundary_type``
values such as ``"flat-surface"``, ``"rectangle-hole"``, or
``"hyperboloid-hole"`` are intentionally skipped. Inside complex mode the
following ``boundary_types(*)`` entries are recognised:

    rectangle, cuboid, sphere,
    circlex, circley, circlez,
    cylinderx, cylindery, cylinderz,
    open-cylinderx, open-cylindery, open-cylinderz,
    diskx, disky, diskz

Every boundary exposes ``mesh(use_si=False, **overrides)``. With
``use_si=True`` the geometry is converted to physical (SI) units using
``data.unit.length.reverse``. Extra keyword arguments are forwarded to the
underlying :class:`MeshSurface3D` constructor, overriding values read from
``inp`` (useful for tuning resolution). :class:`BoundaryCollection` exposes
a companion ``mesh()`` that concatenates every boundary into a single
:class:`CompositeMeshSurface` with optional per-index overrides.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Type

import numpy as np

from emout.plot.surface_cut import (
    BoxMeshSurface,
    CircleMeshSurface,
    CompositeMeshSurface,
    CylinderMeshSurface,
    DiskMeshSurface,
    MeshSurface3D,
    PlaneWithCircleMeshSurface,
    RectangleMeshSurface,
    SphereMeshSurface,
)


# ---------------------------------------------------------------------------
# Mesh-class kwarg introspection (for BoundaryCollection.mesh filtering)
# ---------------------------------------------------------------------------


def _accepted_kwargs(mesh_cls: Optional[type]) -> Optional[Set[str]]:
    """Return the set of keyword argument names accepted by ``mesh_cls.__init__``.

    Returns ``None`` if ``mesh_cls`` is ``None`` or its constructor accepts
    arbitrary ``**kwargs`` (in which case the caller should treat the
    boundary as accepting any kwarg). Otherwise returns the explicit set of
    parameter names, with ``self`` removed.
    """
    if mesh_cls is None:
        return None
    try:
        sig = inspect.signature(mesh_cls.__init__)
    except (TypeError, ValueError):
        return None

    accepted: Set[str] = set()
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return None
        accepted.add(name)
    return accepted


# ---------------------------------------------------------------------------
# f90nml sparse-array access helpers
# ---------------------------------------------------------------------------


def _get_scalar(nml_group, name: str, ib_fortran: int):
    """Return the ``ib_fortran``-th element of a sparse 1-D namelist array.

    ``ib_fortran`` is 1-indexed to match Fortran conventions. Returns ``None``
    if the array is absent, the index falls outside the stored range, or the
    stored element itself is ``None``.
    """
    if name not in nml_group:
        return None
    arr = nml_group[name]
    si = nml_group.start_index.get(name) if hasattr(nml_group, "start_index") else None
    start = si[0] if si and si[0] is not None else 1
    idx = ib_fortran - start
    if idx < 0 or idx >= len(arr):
        return None
    value = arr[idx]
    if value is None:
        return None
    return value


def _get_vector(nml_group, name: str, ib_fortran: int) -> Optional[List[float]]:
    """Return the ``ib_fortran``-th vector of a sparse 2-D namelist array.

    MPIEMSES declares boundary parameter arrays as ``param(ndim, nbt)`` in
    Fortran. ``f90nml`` stores that as a list-of-lists keyed by the boundary
    index (second Fortran dimension). Returns ``None`` if the stored entry is
    missing or filled with ``None`` placeholders.
    """
    if name not in nml_group:
        return None
    arr = nml_group[name]
    si = nml_group.start_index.get(name) if hasattr(nml_group, "start_index") else None

    start = 1
    if si:
        # 2-D arrays typically produce [None, start] (first dim fully written,
        # second dim sparse). Fall back to the first entry when only one is
        # available.
        if len(si) >= 2 and si[1] is not None:
            start = si[1]
        elif si[0] is not None:
            start = si[0]

    idx = ib_fortran - start
    if idx < 0 or idx >= len(arr):
        return None
    vec = arr[idx]
    if vec is None:
        return None
    if isinstance(vec, (list, tuple)):
        values = list(vec)
        if all(v is None for v in values):
            return None
        return values
    return None


# ---------------------------------------------------------------------------
# Boundary base class
# ---------------------------------------------------------------------------


class Boundary:
    """Base class for a single finbound sub-boundary.

    Concrete subclasses know which ``&ptcond`` parameters to consult and how
    to construct their :class:`MeshSurface3D` representation. They must
    implement :meth:`_build_params` and :meth:`_build_mesh`, and they should
    declare :attr:`mesh_class` so :class:`BoundaryCollection.mesh` can filter
    broadcast keyword arguments to the kwargs the underlying mesh class
    actually accepts.

    Parameters
    ----------
    inp
        The :class:`emout.utils.InpFile` holding ``data.inp``'s namelist.
    unit
        The :class:`emout.utils.Units` instance for SI conversion (may be
        ``None`` if the simulation has no unit conversion key).
    index
        Zero-indexed position of this boundary inside ``boundary_types``.
    btype
        The MPIEMSES type string (``"sphere"``, ``"cylinderz"``, …). Kept so
        classes handling several axis variants can infer the axis letter.
    """

    #: The :class:`MeshSurface3D` subclass that this boundary builds. Used by
    #: :meth:`BoundaryCollection.mesh` to filter broadcast keyword arguments
    #: so that, for example, ``data.boundaries.mesh(theta_range=...)`` only
    #: forwards ``theta_range`` to boundaries whose mesh class actually
    #: accepts it. Set to ``None`` to opt out of filtering.
    mesh_class: Optional[Type[MeshSurface3D]] = None

    def __init__(self, inp, unit, index: int, btype: str):
        self.inp = inp
        self.unit = unit
        self.index = index
        self.fortran_index = index + 1
        self.btype = btype

    def __repr__(self) -> str:
        return f"<{type(self).__name__} index={self.index} btype={self.btype!r}>"

    # -- inp helpers ---------------------------------------------------------

    def _ptcond(self):
        return self.inp.nml["ptcond"]

    def _to_si_length(self, value):
        if self.unit is None:
            raise ValueError(
                "use_si=True requires a unit conversion key; this simulation "
                "has no data.unit."
            )
        return self.unit.length.reverse(value)

    # -- to be implemented by subclasses ------------------------------------

    def _build_params(self, use_si: bool) -> Dict[str, Any]:
        raise NotImplementedError

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        raise NotImplementedError

    # -- public API ----------------------------------------------------------

    def mesh(self, *, use_si: bool = True, **overrides) -> MeshSurface3D:
        """Build the :class:`MeshSurface3D` for this boundary.

        Parameters auto-detected from ``data.inp`` may be overridden by
        passing keyword arguments. Overrides are interpreted in the same unit
        system as the returned mesh: SI metres by default
        (``use_si=True``), or simulation grid units when ``use_si=False``.
        """
        params = self._build_params(use_si=use_si)
        params.update(overrides)
        return self._build_mesh(params)

    # -- composition ---------------------------------------------------------

    def __add__(self, other):
        """Combine boundaries (or a boundary and another collection/mesh).

        ``boundary1 + boundary2`` and ``boundary + collection`` both return a
        :class:`BoundaryCollection` whose ``mesh()`` builds the composite,
        so the user can pick ``use_si``/``per`` at render time::

            mesh = (data.boundaries[0] + data.boundaries[2]).mesh(use_si=True)

        ``boundary + mesh_surface`` falls back to building this boundary's
        mesh in grid units and concatenating with the existing
        :class:`CompositeMeshSurface` ``+`` operator.
        """
        if isinstance(other, Boundary):
            unit = self.unit if self.unit is not None else other.unit
            return BoundaryCollection.from_boundaries([self, other], unit=unit)
        if isinstance(other, BoundaryCollection):
            unit = self.unit if self.unit is not None else other.unit
            return BoundaryCollection.from_boundaries([self, *other], unit=unit)
        if isinstance(other, MeshSurface3D):
            return self.mesh() + other
        return NotImplemented

    def __radd__(self, other):
        if other == 0:  # enables sum([b1, b2, ...])
            return BoundaryCollection.from_boundaries([self], unit=self.unit)
        if isinstance(other, MeshSurface3D):
            return other + self.mesh()
        return NotImplemented

    def render(self, *, use_si: bool = True, **style_kwargs):
        """Build the mesh and wrap it in a ``RenderItem`` for ``plot_surfaces``.

        Convenience that mirrors :meth:`MeshSurface3D.render`. ``use_si``
        defaults to ``True`` (SI metres) and is forwarded to :meth:`mesh`;
        remaining keyword arguments are passed through to ``RenderItem``
        (``style``, ``solid_color``, ``alpha``, …). For finer
        mesh-construction control, chain explicitly:
        ``boundary.mesh(ntheta=96).render(style="solid")``.
        """
        return self.mesh(use_si=use_si).render(**style_kwargs)


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
            raise ValueError(
                f"sphere_origin/sphere_radius not set for boundary index {self.index}"
            )
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
            raise ValueError(
                f"cuboid_shape not set for boundary index {self.index}"
            )
        values = np.asarray([float(v) for v in shape], dtype=np.float64)
        if use_si:
            values = self._to_si_length(values)
        xmin, xmax, ymin, ymax, zmin, zmax = values.tolist()
        return {
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax,
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
            raise ValueError(
                f"rectangle_shape not set for boundary index {self.index}"
            )
        values = np.asarray([float(v) for v in shape], dtype=np.float64)
        if use_si:
            values = self._to_si_length(values)
        xmin, xmax, ymin, ymax, zmin, zmax = values.tolist()
        return {
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax,
        }

    def _build_mesh(self, params: Mapping[str, Any]) -> MeshSurface3D:
        xmin = params["xmin"]; xmax = params["xmax"]
        ymin = params["ymin"]; ymax = params["ymax"]
        zmin = params["zmin"]; zmax = params["zmax"]

        tol = 1e-12
        face: Optional[str] = None
        if abs(xmax - xmin) <= tol:
            face = "xmin"
        elif abs(ymax - ymin) <= tol:
            face = "ymin"
        elif abs(zmax - zmin) <= tol:
            face = "zmin"
        kwargs = {
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax,
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
            raise ValueError(
                f"circle_origin/circle_radius not set for boundary index {self.index}"
            )
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
            raise ValueError(
                "cylinder_origin/cylinder_radius/cylinder_height not set for "
                f"boundary index {self.index}"
            )
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
                "disk_origin/disk_radius/disk_inner_radius/disk_height not set "
                f"for boundary index {self.index}"
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
            raise ValueError(
                "plane_with_circle_origin/radius not set for boundary index "
                f"{self.index}"
            )
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
        si = inp.nml["ptcond"].start_index.get(name) if hasattr(
            inp.nml["ptcond"], "start_index"
        ) else None
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
        "xl": xl, "xu": xu,
        "yl": yl, "yu": yu,
        "zl": zl, "zu": float(zssurf),
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
            "xmin": xl, "xmax": xu,
            "ymin": yl, "ymax": yu,
            "zmin": zl, "zmax": zu,
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


def _domain_extent(inp) -> Tuple[float, float, float]:
    """Return ``(nx, ny, nz)`` grid extents from ``inp``.

    Falls back to 1.0 for any axis that ``inp`` does not specify, which lets
    the boundary classes still produce *something* visible without crashing
    on incomplete input files.
    """
    nx = _safe_attr(inp, "nx") or 1.0
    ny = _safe_attr(inp, "ny") or 1.0
    nz = _safe_attr(inp, "nz") or 1.0
    return float(nx), float(ny), float(nz)


# ---------------------------------------------------------------------------
# Type → class dispatch
# ---------------------------------------------------------------------------


_BOUNDARY_CLASS_MAP: Dict[str, type] = {
    # Finbound "complex" mode primitives (docs Parameters.md §C).
    "sphere": SphereBoundary,
    "cuboid": CuboidBoundary,
    "rectangle": RectangleBoundary,
    "cylinderx": CylinderBoundary,
    "cylindery": CylinderBoundary,
    "cylinderz": CylinderBoundary,
    "open-cylinderx": CylinderBoundary,
    "open-cylindery": CylinderBoundary,
    "open-cylinderz": CylinderBoundary,
    "diskx": DiskBoundary,
    "disky": DiskBoundary,
    "diskz": DiskBoundary,
    "circlex": CircleBoundary,
    "circley": CircleBoundary,
    "circlez": CircleBoundary,
    "plane-with-circlex": PlaneWithCircleBoundary,
    "plane-with-circley": PlaneWithCircleBoundary,
    "plane-with-circlez": PlaneWithCircleBoundary,
    # Legacy single-body modes that also appear in `boundary_types(*)` in
    # MPIEMSES3D src/physics/collision/surfaces.F90. Each pulls global
    # scalars (`zssurf`, `xlrechole`, …) rather than indexed arrays.
    "flat-surface": FlatSurfaceBoundary,
    "rectangle-hole": RectangleHoleBoundary,
    "cylinder-hole": CylinderHoleBoundary,
}


#: The set of `boundary_type` values that, when set *without* `complex`,
#: describe a single-body simulation. These are the legacy finbound-extension
#: modes handled directly via :class:`BoundaryCollection`.
_LEGACY_SINGLE_BODY_TYPES: Tuple[str, ...] = (
    "flat-surface",
    "rectangle-hole",
    "cylinder-hole",
)


SUPPORTED_BOUNDARY_TYPES: Tuple[str, ...] = tuple(sorted(_BOUNDARY_CLASS_MAP))


# ---------------------------------------------------------------------------
# BoundaryCollection
# ---------------------------------------------------------------------------


def _offset_mesh(mesh_surface, offsets):
    """Return a copy of *mesh_surface* with vertices shifted by *offsets*.

    Parameters
    ----------
    mesh_surface : MeshSurface3D
        Source mesh.
    offsets : tuple of (float or str)
        ``(x_offset, y_offset, z_offset)``.

    Returns
    -------
    MeshSurface3D
        Shifted mesh (a lightweight wrapper).
    """
    from emout.utils.util import apply_offset

    V, F = mesh_surface.mesh()
    V = V.copy()
    for axis_idx in range(min(3, len(offsets))):
        if offsets[axis_idx] is not None:
            V[:, axis_idx] = apply_offset(V[:, axis_idx].copy(), offsets[axis_idx])

    class _OffsetMesh(MeshSurface3D):
        def mesh(self):
            return V, F

    return _OffsetMesh()


class BoundaryCollection:
    """Collection of boundaries discovered in ``data.inp``'s finbound config.

    Supports ``len``, iteration, and integer indexing. ``mesh()`` returns a
    composite :class:`MeshSurface3D` concatenating every boundary, with
    optional per-index or common keyword overrides.

    Boundaries of unsupported types (or legacy single-body modes outside
    ``boundary_type = 'complex'``) are silently skipped. A
    :attr:`skipped` list records the reason per skipped slot for debugging.
    """

    def __init__(self, inp, unit, remote_open_kwargs: Optional[Mapping[str, Any]] = None):
        self.inp = inp
        self.unit = unit
        self._emout_open_kwargs = (
            None if remote_open_kwargs is None else dict(remote_open_kwargs)
        )
        self.skipped: List[Tuple[int, str, str]] = []
        self._boundaries = self._build()

    @classmethod
    def from_boundaries(
        cls,
        boundaries: "Iterable[Boundary]",
        unit=None,
    ) -> "BoundaryCollection":
        """Build a collection from an explicit iterable of boundaries.

        Skips the namelist parsing path used by ``__init__``. Useful when
        composing boundaries with ``+`` (``boundary1 + boundary2``) or
        constructing ad-hoc subsets for plotting and tests.

        Parameters
        ----------
        boundaries
            Iterable of :class:`Boundary` instances. ``BoundaryCollection``
            children are flattened.
        unit
            Optional :class:`emout.utils.Units` for SI conversion. Falls back
            to the first non-``None`` ``unit`` found on the input boundaries.
        """
        flat: List[Boundary] = []
        for item in boundaries:
            if isinstance(item, BoundaryCollection):
                flat.extend(item)
            else:
                flat.append(item)
        if unit is None:
            for b in flat:
                if getattr(b, "unit", None) is not None:
                    unit = b.unit
                    break
        instance = cls.__new__(cls)
        instance.inp = None
        instance.unit = unit
        instance._emout_open_kwargs = None
        instance.skipped = []
        instance._boundaries = flat
        return instance

    # -- construction --------------------------------------------------------

    def _build(self) -> List[Boundary]:
        if self.inp is None:
            return []

        btype = _safe_attr(self.inp, "boundary_type")
        if btype is None:
            return []

        if btype == "complex":
            return self._build_complex()

        # Legacy single-body mode: one boundary whose global scalars
        # (``zssurf``, ``xlrechole``, …) live directly in ``&ptcond``.
        if btype in _LEGACY_SINGLE_BODY_TYPES:
            cls = _BOUNDARY_CLASS_MAP[btype]
            return [cls(self.inp, self.unit, 0, btype)]

        self.skipped.append((0, btype, "unsupported top-level boundary_type"))
        return []

    def _build_complex(self) -> List[Boundary]:
        btypes_raw = _safe_attr(self.inp, "boundary_types")
        if btypes_raw is None:
            return []
        if isinstance(btypes_raw, str):
            btypes_raw = [btypes_raw]

        built: List[Boundary] = []
        for ib, raw in enumerate(btypes_raw):
            if raw is None:
                self.skipped.append((ib, "", "slot is None"))
                continue
            if not isinstance(raw, str):
                self.skipped.append((ib, repr(raw), "non-string boundary type"))
                continue
            name = raw.strip()
            if not name:
                self.skipped.append((ib, raw, "empty boundary type"))
                continue
            cls = _BOUNDARY_CLASS_MAP.get(name)
            if cls is None:
                self.skipped.append((ib, name, "unsupported boundary type"))
                continue
            built.append(cls(self.inp, self.unit, ib, name))
        return built

    # -- container protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self._boundaries)

    def __iter__(self) -> Iterator[Boundary]:
        return iter(self._boundaries)

    def __getitem__(self, idx):
        return self._boundaries[idx]

    def __bool__(self) -> bool:
        return bool(self._boundaries)

    def __repr__(self) -> str:
        types = ", ".join(b.btype for b in self._boundaries)
        return f"<BoundaryCollection [{types}]>"

    # -- composition ---------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, BoundaryCollection):
            unit = self.unit if self.unit is not None else other.unit
            return type(self).from_boundaries([*self, *other], unit=unit)
        if isinstance(other, Boundary):
            unit = self.unit if self.unit is not None else other.unit
            return type(self).from_boundaries([*self, other], unit=unit)
        if isinstance(other, MeshSurface3D):
            return self.mesh() + other
        return NotImplemented

    def __radd__(self, other):
        if other == 0:  # sum([...]) compatibility
            return self
        if isinstance(other, MeshSurface3D):
            return other + self.mesh()
        return NotImplemented

    def render(
        self,
        *,
        use_si: bool = True,
        per: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **style_kwargs,
    ):
        """Build the composite mesh and wrap it in a ``RenderItem``.

        ``use_si`` defaults to ``True`` (SI metres). ``use_si`` and ``per``
        are forwarded to :meth:`mesh`; remaining keyword arguments are
        forwarded to ``RenderItem`` (``style``, ``solid_color``, ``alpha``,
        …).
        """
        return self.mesh(use_si=use_si, per=per).render(**style_kwargs)

    def plot(
        self,
        *,
        ax=None,
        use_si: bool = True,
        offsets=None,
        per: Optional[Mapping[int, Mapping[str, Any]]] = None,
        style: str = "solid",
        solid_color="0.7",
        alpha: float = 0.6,
        **kwargs,
    ):
        """Plot boundary meshes in 3-D.

        Use this to inspect boundary geometry without a field overlay.
        To overlay on field data, use ``Data3d.plot_surfaces(data.boundaries)``.

        Parameters
        ----------
        ax : Axes3D, optional
            Target 3-D axes. Created if ``None``.
        use_si : bool, default True
            Convert to SI metres.
        offsets : tuple of (float or str), optional
            ``(x_offset, y_offset, z_offset)`` applied to mesh vertices.
            Accepts ``"left"``, ``"center"``, ``"right"`` or numeric values.
        per : dict, optional
            Per-boundary mesh overrides.
        style, solid_color, alpha
            Rendering style forwarded to :class:`RenderItem`.
        """
        from emout.distributed.remote_figure import (
            is_recording,
            record_boundary_plot,
            request_session,
        )
        import matplotlib.pyplot as plt
        from emout.plot.surface_cut import plot_surfaces as _plot_surfaces, RenderItem

        if is_recording():
            emout_kw = getattr(self, "_emout_open_kwargs", None)
            request_session(emout_kw)
            record_boundary_plot(
                {
                    "use_si": use_si,
                    "offsets": offsets,
                    "per": per,
                    "style": style,
                    "solid_color": solid_color,
                    "alpha": alpha,
                    **kwargs,
                },
                emout_kwargs=emout_kw,
            )
            return None

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        composite = self.mesh(use_si=use_si, per=per)

        if offsets is not None:
            composite = _offset_mesh(composite, offsets)

        item = RenderItem(composite, style=style, solid_color=solid_color, alpha=alpha, **kwargs)
        _plot_surfaces(ax, field=None, surfaces=item)
        return ax

    # -- composite mesh ------------------------------------------------------

    def mesh(
        self,
        *,
        use_si: bool = True,
        per: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **common_overrides,
    ) -> MeshSurface3D:
        """Return the composite mesh of all recognised boundaries.

        Parameters
        ----------
        use_si : bool, default True
            Convert geometry to SI metres via ``data.unit.length.reverse``.
            Pass ``use_si=False`` to keep simulation grid units instead.
        per : dict, optional
            Mapping from boundary index (0-based) to a dict of overrides
            passed to that boundary's ``mesh()`` call. Per-index entries are
            merged on top of (and override) the broadcast ``common_overrides``
            and **bypass the kwarg filter** — they go straight through to
            ``boundary.mesh()``.
        **common_overrides
            Overrides broadcast to every boundary. Each boundary only sees
            the subset of these kwargs that its underlying ``mesh_class``
            ``__init__`` actually accepts (introspected via
            :func:`inspect.signature`). For example::

                data.boundaries.mesh(theta_range=[0, np.pi])

            forwards ``theta_range`` to the sphere/cylinder/disk/circle/
            plane-with-circle entries that support it and silently drops it
            for cuboid/rectangle/flat-surface entries.
        """
        per = per or {}
        children: List[MeshSurface3D] = []
        for boundary in self._boundaries:
            accepted = _accepted_kwargs(getattr(type(boundary), "mesh_class", None))
            if accepted is None:
                # No introspection possible — broadcast everything verbatim.
                extra: Dict[str, Any] = dict(common_overrides)
            else:
                extra = {k: v for k, v in common_overrides.items() if k in accepted}
            # Per-boundary overrides always win, and bypass the filter so the
            # caller can target a specific entry with kwargs the broadcast
            # filter would otherwise drop.
            extra.update(per.get(boundary.index, {}))
            children.append(boundary.mesh(use_si=use_si, **extra))
        return CompositeMeshSurface(children)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _safe_attr(inp, name: str):
    """Fetch ``inp.<name>`` tolerating both ``KeyError`` and ``AttributeError``.

    :class:`emout.utils.emsesinp.InpFile` raises :class:`KeyError` for missing
    keys instead of :class:`AttributeError`, so plain ``getattr(inp, name,
    default)`` cannot catch the absence.
    """
    try:
        return getattr(inp, name)
    except (KeyError, AttributeError):
        return None


__all__ = [
    "Boundary",
    "BoundaryCollection",
    "SphereBoundary",
    "CuboidBoundary",
    "RectangleBoundary",
    "CircleBoundary",
    "CylinderBoundary",
    "DiskBoundary",
    "PlaneWithCircleBoundary",
    "FlatSurfaceBoundary",
    "RectangleHoleBoundary",
    "CylinderHoleBoundary",
    "SUPPORTED_BOUNDARY_TYPES",
]
