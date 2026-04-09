"""3-D iso-surface (contour) rendering via matplotlib tri-surface plots."""

# isosurface_mpl_roi.py
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

DxLike = Union[float, Sequence[float]]
Bounds = Tuple[Tuple[Optional[float], Optional[float]],
               Tuple[Optional[float], Optional[float]],
               Tuple[Optional[float], Optional[float]]]
LevelFormatter = Union[str, Callable[[float], str]]


def _as_spacing_xyz(dx: DxLike) -> Tuple[float, float, float]:
    """Normalize grid spacing to `(dx, dy, dz)` form.

    Parameters
    ----------
    dx : DxLike
        Grid spacing. If scalar, interpreted as `(dx, dx, dx)`;
        if 3-element, interpreted as `(dx, dy, dz)`.
    Returns
    -------
    Tuple[float, float, float]
        A 3-element tuple `(dx, dy, dz)`.
    """
    if np.isscalar(dx):
        d = float(dx)
        return (d, d, d)
    t = tuple(float(v) for v in dx)
    if len(t) != 3:
        raise ValueError("dx must be a scalar or (dx, dy, dz).")
    return t


def _sanitize_volume(vol: np.ndarray) -> np.ndarray:
    """Sanitize a 3-D volume array for rendering.

    Parameters
    ----------
    vol : np.ndarray
        3-D volume array.
    Returns
    -------
    np.ndarray
        A float-type `(nz, ny, nx)` array with NaN/Inf replaced.
    """
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"data3d must be 3D array (nz, ny, nx). Got ndim={vol.ndim}.")
    if not np.issubdtype(vol.dtype, np.floating):
        vol = vol.astype(np.float32, copy=False)
    if not np.isfinite(vol).all():
        vol = np.nan_to_num(vol, nan=0.0,
                            posinf=np.finfo(vol.dtype).max,
                            neginf=np.finfo(vol.dtype).min)
    return vol


def _format_level_value(
    value: float,
    *,
    fmt: Optional[LevelFormatter] = None,
    sigfigs: Optional[int] = None,
) -> str:
    """Format an iso-surface level value as a display string.

    Parameters
    ----------
    value : float
        The numeric value to format.
    fmt : Optional[LevelFormatter], optional
        Format specifier string or callable.
    sigfigs : Optional[int], optional
        Number of significant figures used when `fmt` is not given.
    Returns
    -------
    str
        Formatted string representation.
    """
    if callable(fmt):
        return str(fmt(value))

    if isinstance(fmt, str):
        # old-style format like '%1.2f'
        if "%" in fmt:
            try:
                return fmt % value
            except (TypeError, ValueError):
                pass
        # format mini-language, e.g. '.3g'
        try:
            return format(value, fmt)
        except (TypeError, ValueError):
            pass
        # str.format style, e.g. '{:.3g}' or '{value:.3g}'
        try:
            return fmt.format(value)
        except (IndexError, KeyError, ValueError):
            return fmt.format(value=value)

    if sigfigs is not None:
        if sigfigs < 1:
            raise ValueError("clabel_sigfigs must be >= 1.")
        return f"{value:.{sigfigs}g}"
    return f"{value:g}"


def _resolve_shared_exponent(levels: Sequence[float], shared_exponent: Union[str, int, None]) -> int:
    """Determine the shared exponent for label display.

    Parameters
    ----------
    levels : Sequence[float]
        Array of level values used for iso-surface labels.
    shared_exponent : Union[str, int, None]
        Shared exponent setting.
    Returns
    -------
    int
        The shared exponent to use.
    """
    if shared_exponent is None:
        return 0
    if isinstance(shared_exponent, int):
        return shared_exponent
    if shared_exponent != "auto":
        raise ValueError("clabel_shared_exponent must be None, 'auto', or int.")

    finite_levels = np.asarray(levels, dtype=np.float64)
    finite_levels = finite_levels[np.isfinite(finite_levels)]
    if finite_levels.size == 0:
        return 0

    max_abs = np.max(np.abs(finite_levels))
    if max_abs == 0.0:
        return 0
    return int(np.floor(np.log10(max_abs)))


def _slice_from_bounds_1d(
    vmin: Optional[float],
    vmax: Optional[float],
    origin: float,
    d: float,
    n: int,
    axis_name: str,
) -> Tuple[slice, float]:
    """Compute an index range from 1-D boundary specification.

    Parameters
    ----------
    vmin : Optional[float]
        Minimum value of the display range.
    vmax : Optional[float]
        Maximum value of the display range.
    origin : float
        Reference point for the axis origin.
    d : float
        Grid spacing.
    n : int
        Number of samples or grid points.
    axis_name : str
        Axis name (`x`/`y`/`z`).
    Returns
    -------
    Tuple[slice, float]
        `(index_slice, new_axis_origin)`.
    """
    if (vmin is not None) and (vmax is not None) and (vmax < vmin):
        raise ValueError(f"{axis_name} bounds invalid: max < min")

    # index i corresponds to x = origin + i*d (node-based)
    i0 = 0 if vmin is None else int(np.ceil((vmin - origin) / d))
    i1 = n if vmax is None else int(np.floor((vmax - origin) / d) + 1)  # end-exclusive

    i0 = max(0, min(n, i0))
    i1 = max(0, min(n, i1))

    if i1 - i0 < 2:
        raise ValueError(
            f"{axis_name} ROI too small (needs >=2 samples). got [{i0}:{i1}] over n={n}"
        )

    new_origin = origin + i0 * d
    return slice(i0, i1), new_origin


def _apply_roi(
    vol_zyx: np.ndarray,
    dx_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
    bounds_xyz: Optional[Bounds],
    roi_zyx: Optional[Tuple[slice, slice, slice]],
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """Crop the array and origin based on the ROI specification.

    Parameters
    ----------
    vol_zyx : np.ndarray
        Volume array in `(z, y, x)` order.
    dx_xyz : Tuple[float, float, float]
        Grid spacing in `(x, y, z)` order.
    origin_xyz : Tuple[float, float, float]
        Origin coordinates in `(x, y, z)` order.
    bounds_xyz : Optional[Bounds]
        Crop range in `((xmin, xmax), (ymin, ymax), (zmin, zmax))` form.
        Use `None` for any element to include the full range along that axis.
    roi_zyx : Optional[Tuple[slice, slice, slice]]
        Direct index slices in `(z, y, x)` order.
        Takes precedence over `bounds_xyz` when given.
    Returns
    -------
    Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]
        `(cropped_volume, dx_xyz, updated_origin_xyz)`.
    """
    nz, ny, nx = vol_zyx.shape
    dx, dy, dz = dx_xyz
    x0, y0, z0 = origin_xyz

    if roi_zyx is not None:
        sz, sy, sx = roi_zyx
        sub = vol_zyx[sz, sy, sx]

        # slice.start can be None
        iz0 = 0 if sz.start is None else int(sz.start)
        iy0 = 0 if sy.start is None else int(sy.start)
        ix0 = 0 if sx.start is None else int(sx.start)

        new_origin = (x0 + ix0 * dx, y0 + iy0 * dy, z0 + iz0 * dz)
        return sub, dx_xyz, new_origin

    if bounds_xyz is None:
        return vol_zyx, dx_xyz, origin_xyz

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds_xyz

    sx, new_x0 = _slice_from_bounds_1d(xmin, xmax, x0, dx, nx, "x")
    sy, new_y0 = _slice_from_bounds_1d(ymin, ymax, y0, dy, ny, "y")
    sz, new_z0 = _slice_from_bounds_1d(zmin, zmax, z0, dz, nz, "z")

    sub = vol_zyx[sz, sy, sx]
    return sub, dx_xyz, (new_x0, new_y0, new_z0)


def contour3d(
    data3d: np.ndarray,          # (nz, ny, nx)
    dx: DxLike,                  # scalar or (dx, dy, dz) in x,y,z
    levels: Sequence[float],
    *,
    ax=None,
    origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    # ROI options (choose one)
    bounds_xyz: Optional[Bounds] = None,  # ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    roi_zyx: Optional[Tuple[slice, slice, slice]] = None,  # (slice_z, slice_y, slice_x)
    # rendering
    opacity: float = 0.35,
    step: int = 1,
    title: Optional[str] = None,
    save: Optional[str] = None,
    show: bool = True,
    xlabel: str = None,
    ylabel: str = None,
    zlabel: str = None,
    clabel: bool = False,
    clabel_fmt: Optional[LevelFormatter] = None,
    clabel_fontsize: float = 10,
    clabel_sigfigs: Optional[int] = None,
    clabel_shared_exponent: Union[str, int, None] = None,
    clabel_text_kwargs: Optional[Dict[str, Any]] = None,
    clabel_exponent_pos: Tuple[float, float] = (1.02, 0.98),
    clabel_exponent_text: Optional[str] = None,
    clabel_exponent_kwargs: Optional[Dict[str, Any]] = None,
):
    """Render iso-surfaces of a 3-D volume with Matplotlib.

    The input array is expected in `(z, y, x)` axis order. Use `bounds_xyz`
    (physical coordinates) or `roi_zyx` (index slices) to crop the render
    region.

    Parameters
    ----------
    data3d : np.ndarray
        3-D array to render, shaped `(nz, ny, nx)`.
    dx : DxLike
        Grid spacing. When scalar, treated as `(dx, dy, dz)`.
    levels : Sequence[float]
        Array of iso-surface levels to render.
    ax : object, optional
        Target 3-D Axes. If `None`, a new one is created.
    origin_xyz : Tuple[float, float, float], optional
        Origin offset in physical coordinates `(x0, y0, z0)`.
    bounds_xyz : Optional[Bounds], optional
        Physical-coordinate ROI. `((xmin, xmax), (ymin, ymax), (zmin, zmax))`.
    roi_zyx : Optional[Tuple[slice, slice, slice]], optional
        Index-based ROI. `(slice_z, slice_y, slice_x)`.
    opacity : float, optional
        Iso-surface opacity (0.0 to 1.0).
    step : int, optional
        Down-sampling step after ROI. `step=2` samples every 2 points per axis.
    title : Optional[str], optional
        Plot title.
    save : Optional[str], optional
        Output file path. When specified, `fig.savefig` is called.
    show : bool, optional
        If `True`, call `plt.show()`.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    zlabel : str, optional
        Z-axis label.
    clabel : bool, optional
        If `True`, draw level-value labels near each iso-surface.
    clabel_fmt : Optional[LevelFormatter], optional
        Format specifier for label strings.
    clabel_fontsize : float, optional
        Font size for iso-surface labels.
    clabel_sigfigs : Optional[int], optional
        Number of significant figures for labels (when `clabel_fmt` is not set).
    clabel_shared_exponent : Union[str, int, None], optional
        Shared exponent for labels. `'auto'` determines it automatically.
    clabel_text_kwargs : Optional[Dict[str, Any]], optional
        Extra keyword arguments forwarded to `ax.text` for each label.
    clabel_exponent_pos : Tuple[float, float], optional
        2-D axes-coordinate position `(x, y)` for the shared exponent text.
    clabel_exponent_text : Optional[str], optional
        Text for the shared exponent display. If `None`, auto-generated.
    clabel_exponent_kwargs : Optional[Dict[str, Any]], optional
        Extra keyword arguments forwarded to `ax.text2D` for the exponent text.

    Returns
    -------
    tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
        The `(fig, ax)` pair used for rendering.
    """
    if not levels:
        raise ValueError("levels must be non-empty.")
    if bounds_xyz is not None and roi_zyx is not None:
        raise ValueError("Specify only one of bounds_xyz or roi_zyx.")
    if not (0.0 <= opacity <= 1.0):
        raise ValueError("opacity must be in [0, 1].")
    step = int(step)
    if step < 1:
        raise ValueError("step must be >= 1.")
    if clabel and clabel_fontsize <= 0:
        raise ValueError("clabel_fontsize must be > 0.")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    vol = _sanitize_volume(data3d)
    dx_xyz = _as_spacing_xyz(dx)

    # ROI crop first (fast + predictable)
    vol, (dx_, dy_, dz_), (x0, y0, z0) = _apply_roi(vol, dx_xyz, origin_xyz, bounds_xyz, roi_zyx)

    # Optional downsample after ROI
    if step > 1:
        vol = vol[::step, ::step, ::step]
        dx_, dy_, dz_ = dx_ * step, dy_ * step, dz_ * step

    nz, ny, nx = vol.shape
    spacing_zyx = (dz_, dy_, dx_)  # array axis order: z, y, x

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = plt.gcf()

    label_text_kwargs = {} if clabel_text_kwargs is None else dict(clabel_text_kwargs)
    exponent_kwargs = {} if clabel_exponent_kwargs is None else dict(clabel_exponent_kwargs)
    shared_exp = _resolve_shared_exponent(levels, clabel_shared_exponent)
    scale = 10.0 ** shared_exp if shared_exp != 0 else 1.0

    for lv in levels:
        verts_zyx, faces, _, _ = marching_cubes(
            vol, level=float(lv), spacing=spacing_zyx, allow_degenerate=False
        )

        # (z,y,x) -> (x,y,z) and apply origin
        verts_xyz = verts_zyx[:, [2, 1, 0]]
        verts_xyz[:, 0] += x0
        verts_xyz[:, 1] += y0
        verts_xyz[:, 2] += z0

        mesh = Poly3DCollection(verts_xyz[faces], alpha=opacity)
        ax.add_collection3d(mesh)

        if clabel:
            label_level = float(lv) / scale
            label_text = _format_level_value(
                label_level,
                fmt=clabel_fmt,
                sigfigs=clabel_sigfigs,
            )
            # label near the center of each surface
            center = verts_xyz.mean(axis=0)
            ax.text(
                float(center[0]),
                float(center[1]),
                float(center[2]),
                label_text,
                fontsize=clabel_fontsize,
                **label_text_kwargs,
            )

    # ax.set_xlim(x0, x0 + (nx - 1) * dx_)
    # ax.set_ylim(y0, y0 + (ny - 1) * dy_)
    # ax.set_zlim(z0, z0 + (nz - 1) * dz_)
    
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zlabel:
        ax.set_zlabel(zlabel)
        
    if title:
        ax.set_title(title)

    if clabel and shared_exp != 0:
        exponent_text = clabel_exponent_text or rf"$\times 10^{{{shared_exp}}}$"
        x_pos, y_pos = clabel_exponent_pos
        ax.text2D(
            x_pos,
            y_pos,
            exponent_text,
            transform=ax.transAxes,
            fontsize=clabel_fontsize,
            ha="left",
            va="top",
            **exponent_kwargs,
        )

    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax
