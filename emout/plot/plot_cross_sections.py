from typing import Callable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_cross_sections(
    data,
    axis: str = "z",
    coord: float = 0.0,
    ax: Optional[plt.Axes] = None,
    use_si: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Plot boundaries (spheres, flat surfaces, rectangle holes) sliced by plane axis=coord.

    Parameters
    ----------
    data : object
        Must have `inp.boundary_type`, `inp.boundary_types`, and relevant params.
    axis : {'x','y','z'}
        Normal of slicing plane.
    coord : float
        Coordinate along `axis` of the slicing plane.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    kwargs
        Passed to individual plot functions for styling.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    # Early exit if no complex boundaries
    inp = getattr(data, "inp", None)
    if not inp or getattr(inp, "boundary_type", None) != "complex":
        return ax or plt.gca()

    # Prepare axes
    ax = ax or plt.gca()

    if use_si:
        coord = data.unit.length.reverse(coord)

    # Dispatch table for boundary types
    handlers: dict[str, Callable] = {
        "sphere": lambda ib: _handle_sphere(
            data, ib, axis, coord, ax, use_si, **kwargs
        ),
        "flat-surface": lambda ib: _handle_flat(
            data, axis, coord, ax, use_si, **kwargs
        ),
        "rectangle-hole": lambda ib: _handle_rect(
            data, axis, coord, ax, use_si, **kwargs
        ),
    }

    for ib, btype in enumerate(inp.boundary_types or []):
        handler = handlers.get(btype)
        if handler:
            handler(ib)

    return ax


def _handle_sphere(
    data, ib: int, axis: str, coord: float, ax: plt.Axes, use_si: bool, **kwargs
) -> None:
    """Plot sphere cross-section."""
    nml = data.inp.nml["ptcond"]
    si = nml.start_index["sphere_origin"][1]
    idx = ib - (si - 1)
    # fetch and convert
    center = np.array(data.inp.sphere_origin[idx])
    radius = float(data.inp.sphere_radius[idx])
    # unit conversion
    if use_si:
        center = data.unit.length.reverse(center)
        radius = data.unit.length.reverse(radius)
    # delegate to generic sphere plotter
    _plot_spheres(np.column_stack(([center], [radius])), axis, coord, ax, **kwargs)


def _handle_flat(
    data, axis: str, coord: float, ax: plt.Axes, use_si: bool, **kwargs
) -> None:
    """Plot flat-surface boundary (horizontal)."""
    zs = float(data.inp.zssurf)

    if use_si:
        zs = data.unit.length.reverse(zs)

    if axis == "z":
        return  # no-op for z plane
    # For x/y slice, draw horizontal line in 2D plot coords
    ax = ax or plt.gca()
    ax.axhline(zs, **kwargs)


def _handle_rect(
    data, axis: str, coord: float, ax: plt.Axes, use_si: bool, **kwargs
) -> None:
    """Plot a rectangular hole cross-section."""
    inp = data.inp
    # bounds in model units
    xl, xu = inp.xlrechole[1], inp.xurechole[1]
    yl, yu = inp.ylrechole[1], inp.yurechole[1]
    zl, zu = inp.zlrechole[1], inp.zurechole[1]
    nx, ny = inp.nx, inp.ny

    # convert to physical units
    if use_si:
        to_phys = data.unit.length.reverse
        xl, xu, yl, yu = map(to_phys, (xl, xu, yl, yu))
        zl, zu = to_phys(zl), to_phys(zu)
        nx, ny = to_phys(nx), to_phys(ny)

    # prepare coords
    if axis == "x" and xl <= coord <= xu:
        ys = np.array([0, yl, yl, yu, yu, ny - 1])
        zs = np.array([zu, zu, zl, zl, zu, zu])
        ax.plot(ys, zs, **kwargs)

    elif axis == "y" and yl <= coord <= yu:
        xs = np.array([0, xl, xl, xu, xu, nx - 1])
        zs = np.array([zu, zu, zl, zl, zu, zu])
        ax.plot(xs, zs, **kwargs)

    elif axis == "z" and zl <= coord <= zu:
        xs = np.array([xl, xl, xu, xu, xl])
        ys = np.array([yl, yu, yu, yl, yl])
        ax.plot(xs, ys, **kwargs)


def _plot_spheres(
    spheres: Sequence[Tuple[float, float, float, float]],
    axis: str = "z",
    coord: float = 0.0,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Generic sphere cross-section plotter.

    Parameters
    ----------
    spheres : sequence of (xc, yc, zc, r)
    axis : {'x','y','z'}
    coord : float
    ax : Axes, optional
    kwargs : passed to Circle patch

    Returns
    -------
    ax : Axes
    """
    ax = ax or plt.gca()
    # map axis to index and remaining axes
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    other = [i for i in (0,1,2) if i != idx]

    for xc, yc, zc, r in np.atleast_2d(spheres):
        center = np.array((xc, yc, zc), float)
        d = center[idx] - coord
        if abs(d) > r:
            continue
        rr = np.sqrt(r**2 - d**2)
        cxy = (center[other[0]], center[other[1]])
        circle = plt.Circle(cxy, rr, fill=False, **kwargs)
        ax.add_patch(circle)

    return ax
