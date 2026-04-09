"""Boundary wrappers for MPIEMSES finbound mode.

Re-exports all public classes and constants for backward compatibility.
"""

from ._base import Boundary
from ._collection import (
    SUPPORTED_BOUNDARY_TYPES,
    BoundaryCollection,
    _BOUNDARY_CLASS_MAP,
    _LEGACY_SINGLE_BODY_TYPES,
)
from ._complex_types import (
    CircleBoundary,
    CuboidBoundary,
    CylinderBoundary,
    DiskBoundary,
    PlaneWithCircleBoundary,
    RectangleBoundary,
    SphereBoundary,
)
from ._legacy_types import (
    CylinderHoleBoundary,
    FlatSurfaceBoundary,
    RectangleHoleBoundary,
)

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
