"""PyVista plotting helpers for scalar and vector fields.

Re-exports from submodules for backward compatibility.
"""

from ._pyvista_scalar import (
    create_plane_mesh,
    create_volume_mesh,
    plot_scalar_plane,
    plot_scalar_volume,
)
from ._pyvista_vector import (
    create_vector_mesh3d,
    plot_vector_quiver3d,
    plot_vector_streamlines3d,
)

__all__ = [
    "create_plane_mesh",
    "create_volume_mesh",
    "plot_scalar_plane",
    "plot_scalar_volume",
    "create_vector_mesh3d",
    "plot_vector_quiver3d",
    "plot_vector_streamlines3d",
]
