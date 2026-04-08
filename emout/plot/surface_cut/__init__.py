"""surface_cut

Utilities for:
- Uniform cell-centered 3D grid data (nz,ny,nx)
- SciPy-based interpolation (Field3D)
- Implicit surfaces via SDF-like functions + boolean composition
- Clipping / sampling helpers
- Optional visualization utilities (marching cubes + colormap/contours)

Star-import control is provided via __all__.
"""

from .grid import UniformCellCenteredGrid
from .field import Field3D
from .cutter import SurfaceCutter, KeepSide
from .mesh import (
    MeshSurface3D,
    BoxMeshSurface,
    RectangleMeshSurface,
    CircleMeshSurface,
    CylinderMeshSurface,
    HollowCylinderMeshSurface,
    DiskMeshSurface,
    SphereMeshSurface,
    CompositeMeshSurface,
)
from .sdf import (
    Surface3D,
    HeightFieldSurface,
    DEMHeightFieldSurface,
    PlaneSurface,
    BoxSurface,
    PlaneBoxSurface,
    SphereSurface,
    CylinderSurface,
    UnionSurface,
    IntersectionSurface,
    DifferenceSurface,
    ComplementSurface,
    XorSurface,
    xor,
    OffsetSurface,
    TransformSurface,
    ShiftSurface,
    RotateSurface,
)

__all__ = [
    # core data
    "UniformCellCenteredGrid",
    "Field3D",
    "SurfaceCutter",
    "KeepSide",
    "MeshSurface3D",
    "BoxMeshSurface",
    "RectangleMeshSurface",
    "CircleMeshSurface",
    "CylinderMeshSurface",
    "HollowCylinderMeshSurface",
    "DiskMeshSurface",
    "SphereMeshSurface",
    "CompositeMeshSurface",
    # surfaces
    "Surface3D",
    "HeightFieldSurface",
    "DEMHeightFieldSurface",
    "PlaneSurface",
    "BoxSurface",
    "PlaneBoxSurface",
    "SphereSurface",
    "CylinderSurface",
    # booleans
    "UnionSurface",
    "IntersectionSurface",
    "DifferenceSurface",
    "ComplementSurface",
    "XorSurface",
    "xor",
    # transforms
    "OffsetSurface",
    "TransformSurface",
    "ShiftSurface",
    "RotateSurface",
]

# Optional viz (depends on scikit-image)
try:
    from .viz import Bounds3D, RenderItem, plot_surfaces, add_colorbar

    __all__ += ["Bounds3D", "RenderItem", "plot_surfaces", "add_colorbar"]
except Exception:
    # Keep the package importable without viz deps.
    pass
