"""Compatibility shim.

Old usage:
    from viz_surfaces import plot_surfaces, add_colorbar, Bounds3D

New preferred usage:
    from surface_cut import plot_surfaces, add_colorbar, Bounds3D
    # or: from surface_cut.viz import ...
"""

from surface_cut.viz import Bounds3D, plot_surfaces, add_colorbar

__all__ = ["Bounds3D", "plot_surfaces", "add_colorbar"]
