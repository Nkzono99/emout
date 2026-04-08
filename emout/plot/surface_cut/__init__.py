"""surface_cut

Utilities for:
- Uniform cell-centered 3D grid data (nz,ny,nx)
- SciPy-based interpolation (Field3D)
- Explicit triangle mesh surfaces (MeshSurface3D and friends)
- Optional visualization utilities (plot_surfaces, RenderItem, …)

The legacy SDF-based implicit surface system (``Surface3D`` and
``SurfaceCutter``) was removed; everything goes through ``MeshSurface3D``
now. The high-level entry points for boundary meshes built from
``data.inp`` live in :mod:`emout.emout.boundaries`.

Star-import control is provided via ``__all__``.
"""

from .grid import UniformCellCenteredGrid
from .field import Field3D
from .mesh import (
    MeshSurface3D,
    BoxMeshSurface,
    RectangleMeshSurface,
    CircleMeshSurface,
    CylinderMeshSurface,
    HollowCylinderMeshSurface,
    PlaneWithCircleMeshSurface,
    DiskMeshSurface,
    SphereMeshSurface,
    CompositeMeshSurface,
)

__all__ = [
    # core data
    "UniformCellCenteredGrid",
    "Field3D",
    # mesh surfaces
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

# Optional viz (depends on matplotlib; safe to import without scikit-image now
# that the SDF/marching-cubes code path is gone).
try:
    from .viz import Bounds3D, RenderItem, plot_surfaces, add_colorbar

    __all__ += ["Bounds3D", "RenderItem", "plot_surfaces", "add_colorbar"]
except Exception:
    # Keep the package importable without viz deps.
    pass
