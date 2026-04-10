"""Compatibility tests for the legacy ``emout.plot.viz_surfaces`` shim."""

from emout.plot import viz_surfaces
from emout.plot.surface_cut import Bounds3D, add_colorbar, plot_surfaces


def test_viz_surfaces_reexports_surface_cut_symbols():
    """The compatibility shim should re-export the canonical surface_cut API."""
    assert viz_surfaces.Bounds3D is Bounds3D
    assert viz_surfaces.plot_surfaces is plot_surfaces
    assert viz_surfaces.add_colorbar is add_colorbar
