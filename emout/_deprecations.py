"""Internal deprecation helpers."""

from __future__ import annotations

import warnings


def warn_plot_pyvista_deprecated() -> None:
    """Warn that ``plot_pyvista()`` has been superseded by ``plot3d()``."""
    warnings.warn(
        "plot_pyvista() is deprecated; use plot3d() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
