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

from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from emout.plot.surface_cut import (
    BoxMeshSurface,
    CircleMeshSurface,
    CompositeMeshSurface,
    CylinderMeshSurface,
    DiskMeshSurface,
    MeshSurface3D,
    SphereMeshSurface,
)


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
    implement :meth:`_build_params` and :meth:`_build_mesh`.

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

    def mesh(self, *, use_si: bool = False, **overrides) -> MeshSurface3D:
        """Build the :class:`MeshSurface3D` for this boundary.

        Parameters auto-detected from ``data.inp`` may be overridden by
        passing keyword arguments. Overrides are interpreted in the same unit
        system as the returned mesh (grid units by default, SI metres when
        ``use_si=True``).
        """
        params = self._build_params(use_si=use_si)
        params.update(overrides)
        return self._build_mesh(params)


# ---------------------------------------------------------------------------
# Concrete boundary classes
# ---------------------------------------------------------------------------


class SphereBoundary(Boundary):
    """MPIEMSES ``sphere`` boundary → :class:`SphereMeshSurface`."""

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


# ---------------------------------------------------------------------------
# Type → class dispatch
# ---------------------------------------------------------------------------


_BOUNDARY_CLASS_MAP: Dict[str, type] = {
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
}


SUPPORTED_BOUNDARY_TYPES: Tuple[str, ...] = tuple(sorted(_BOUNDARY_CLASS_MAP))


# ---------------------------------------------------------------------------
# BoundaryCollection
# ---------------------------------------------------------------------------


class BoundaryCollection:
    """Collection of boundaries discovered in ``data.inp``'s finbound config.

    Supports ``len``, iteration, and integer indexing. ``mesh()`` returns a
    composite :class:`MeshSurface3D` concatenating every boundary, with
    optional per-index or common keyword overrides.

    Boundaries of unsupported types (or legacy single-body modes outside
    ``boundary_type = 'complex'``) are silently skipped. A
    :attr:`skipped` list records the reason per skipped slot for debugging.
    """

    def __init__(self, inp, unit):
        self.inp = inp
        self.unit = unit
        self.skipped: List[Tuple[int, str, str]] = []
        self._boundaries = self._build()

    # -- construction --------------------------------------------------------

    def _build(self) -> List[Boundary]:
        if self.inp is None:
            return []

        btype = _safe_attr(self.inp, "boundary_type")
        if btype != "complex":
            return []

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

    # -- composite mesh ------------------------------------------------------

    def mesh(
        self,
        *,
        use_si: bool = False,
        per: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **common_overrides,
    ) -> MeshSurface3D:
        """Return the composite mesh of all recognised boundaries.

        Parameters
        ----------
        use_si : bool, default False
            Convert geometry to SI metres via ``data.unit.length.reverse``.
        per : dict, optional
            Mapping from boundary index (0-based) to a dict of overrides
            passed to that boundary's ``mesh()`` call.
        **common_overrides
            Overrides broadcast to every boundary. Unknown kwargs for a
            particular boundary type will raise ``TypeError`` from the
            underlying mesh constructor — use ``per=`` to target.
        """
        per = per or {}
        children: List[MeshSurface3D] = []
        for boundary in self._boundaries:
            extra: Dict[str, Any] = dict(common_overrides)
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
    "SUPPORTED_BOUNDARY_TYPES",
]
