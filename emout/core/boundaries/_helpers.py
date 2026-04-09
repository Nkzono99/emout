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


def _domain_extent(inp):
    """Return ``(nx, ny, nz)`` grid extents from ``inp``.

    Falls back to 1.0 for any axis that ``inp`` does not specify, which lets
    the boundary classes still produce *something* visible without crashing
    on incomplete input files.
    """
    nx = _safe_attr(inp, "nx") or 1.0
    ny = _safe_attr(inp, "ny") or 1.0
    nz = _safe_attr(inp, "nz") or 1.0
    return float(nx), float(ny), float(nz)


