"""Helpers for plotting and recording bounded 3-D surface fields."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def is_spatial_3d_selection(selectors: tuple[Any, ...]) -> bool:
    """Return whether selectors represent one time and all spatial axes."""
    return (
        len(selectors) == 4
        and isinstance(selectors[0], (int, np.integer))
        and all(not isinstance(selector, (int, np.integer)) for selector in selectors[1:])
    )


def surface_base_spacing(axisunits, valunit, use_si: bool) -> tuple[float, float, float, bool]:
    """Return base ``(dx, dy, dz, effective_si)`` for surface plotting."""
    effective_si = bool(use_si) and valunit is not None and axisunits is not None
    if effective_si and all(unit is not None for unit in axisunits[-3:]):
        dz = float(axisunits[-3].reverse(1.0))
        dy = float(axisunits[-2].reverse(1.0))
        dx = float(axisunits[-1].reverse(1.0))
        return dx, dy, dz, True
    return 1.0, 1.0, 1.0, False


def surface_grid_params(
    xslice: slice,
    yslice: slice,
    zslice: slice,
    axisunits,
    valunit,
    use_si: bool,
) -> tuple[float, float, float, float, float, float, bool]:
    """Return ``(x0, y0, z0, dx, dy, dz, effective_si)`` for sliced data."""
    base_dx, base_dy, base_dz, effective_si = surface_base_spacing(axisunits, valunit, use_si)
    x_step = 1 if xslice.step is None else xslice.step
    y_step = 1 if yslice.step is None else yslice.step
    z_step = 1 if zslice.step is None else zslice.step
    x_start = 0 if xslice.start is None else xslice.start
    y_start = 0 if yslice.start is None else yslice.start
    z_start = 0 if zslice.start is None else zslice.start
    return (
        float(x_start * base_dx),
        float(y_start * base_dy),
        float(z_start * base_dz),
        float(x_step * base_dx),
        float(y_step * base_dy),
        float(z_step * base_dz),
        effective_si,
    )


def plot_surfaces_roi_selectors(
    selectors: tuple[Any, ...],
    axis_lengths: tuple[int, int, int, int],
    axisunits,
    valunit,
    use_si: bool,
    bounds,
    *,
    padding: int = 1,
) -> tuple[Any, ...]:
    """Return selectors cropped to the ``plot_surfaces(bounds=...)`` ROI."""
    if not is_spatial_3d_selection(selectors):
        raise ValueError("plot_surfaces article ROI requires one time index and all spatial axes")
    return surface_roi_selectors(
        selectors,
        axis_lengths,
        axisunits,
        valunit,
        use_si,
        bounds,
        padding=padding,
    )


def surface_roi_selectors(
    selectors: tuple[Any, ...],
    axis_lengths: tuple[int, int, int, int],
    axisunits,
    valunit,
    use_si: bool,
    bounds,
    *,
    padding: int = 1,
) -> tuple[Any, ...]:
    """Return selectors cropped to the spatial ``plot_surfaces`` ROI."""
    if bounds is None:
        return selectors

    dx, dy, dz, _effective_si = surface_base_spacing(axisunits, valunit, use_si)
    tsel, zsel, ysel, xsel = selectors
    return (
        tsel,
        _axis_roi_selector(bounds.z, zsel, axis_lengths[1], dz, padding=padding, axis_name="z"),
        _axis_roi_selector(bounds.y, ysel, axis_lengths[2], dy, padding=padding, axis_name="y"),
        _axis_roi_selector(bounds.x, xsel, axis_lengths[3], dx, padding=padding, axis_name="x"),
    )


def _axis_roi_selector(
    interval: tuple[float | None, float | None],
    selector,
    axis_length: int,
    base_spacing: float,
    *,
    padding: int,
    axis_name: str,
) -> slice:
    if not isinstance(selector, slice):
        raise ValueError(f"plot_surfaces article ROI requires a regular slice on {axis_name}")

    rng = range(*selector.indices(axis_length))
    selected_length = len(rng)
    if selected_length == 0:
        raise ValueError(f"plot_surfaces article ROI has an empty {axis_name} selection")

    step = rng.step
    spacing = float(step * base_spacing)
    origin = float(rng.start * base_spacing)
    lo, hi = interval

    if lo is None:
        local_start = 0
    else:
        local_start = math.floor((float(lo) - origin) / spacing) - padding

    if hi is None:
        local_stop = selected_length
    else:
        local_stop = math.ceil((float(hi) - origin) / spacing) + padding

    local_start = max(0, local_start)
    local_stop = min(selected_length, local_stop)
    if local_stop <= local_start:
        raise ValueError(f"plot_surfaces bounds do not overlap the {axis_name} field range")

    return slice(rng.start + step * local_start, rng.start + step * local_stop, step)
