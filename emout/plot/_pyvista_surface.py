"""PyVista helpers for explicit boundary mesh surfaces."""

from __future__ import annotations

import numpy as np

from ._pyvista_helpers import _configure_plotter, _offseted, _require_pyvista


def create_surface_mesh(surface, offsets=None):
    """Create a PyVista triangular surface mesh from a ``MeshSurface3D``."""
    pv = _require_pyvista()
    vertices, faces = surface.mesh()
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("surface mesh vertices must have shape (n_vertices, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("surface mesh faces must have shape (n_faces, 3)")

    if offsets is not None:
        vertices = vertices.copy()
        for axis in range(3):
            vertices[:, axis] = _offseted(vertices[:, axis], offsets[axis])

    vtk_faces = np.empty((faces.shape[0], 4), dtype=np.int64)
    vtk_faces[:, 0] = 3
    vtk_faces[:, 1:] = faces
    return pv.PolyData(vertices, vtk_faces.ravel())


def plot_surface_mesh(
    surface,
    plotter=None,
    offsets=None,
    show: bool = False,
    color="0.7",
    opacity: float = 0.6,
    show_edges: bool = False,
    add_axes: bool = True,
    show_bounds=None,
    bounds_kwargs=None,
    camera=None,
    camera_position=None,
    axis_labels=None,
    **kwargs,
):
    """Draw a ``MeshSurface3D`` on a PyVista plotter and return it."""
    pv = _require_pyvista()
    if plotter is None:
        plotter = pv.Plotter()

    mesh = create_surface_mesh(surface, offsets=offsets)
    add_mesh_kwargs = {
        "color": color,
        "opacity": opacity,
        "show_edges": show_edges,
    }
    add_mesh_kwargs.update(kwargs)
    plotter.add_mesh(mesh, **add_mesh_kwargs)
    _configure_plotter(
        plotter,
        axis_labels,
        show_bounds=show_bounds,
        bounds_kwargs=bounds_kwargs,
        camera=camera,
        camera_position=camera_position,
    )

    if add_axes:
        plotter.add_axes()
    if show:
        plotter.show()
    return plotter
