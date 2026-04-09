"""Low-level plotting helpers for scalar and vector fields.

Re-exports from :mod:`._plot_2d` and :mod:`._plot_3d` for backward
compatibility.
"""

from ._plot_2d import (
    figsize_with_2d,
    mycmap,
    plot_2d_contour,
    plot_2d_streamline,
    plot_2d_vector,
    plot_2dmap,
    plot_line,
    plot_surface,
)
from ._plot_3d import plot_3d_quiver, plot_3d_streamline

__all__ = [
    "figsize_with_2d",
    "mycmap",
    "plot_2dmap",
    "plot_2d_contour",
    "plot_surface",
    "plot_line",
    "plot_2d_vector",
    "plot_2d_streamline",
    "plot_3d_quiver",
    "plot_3d_streamline",
]
