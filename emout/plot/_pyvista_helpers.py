"""PyVista-based 3-D visualisation helpers for scalar and vector fields.

All functions in this module require the optional ``pyvista`` dependency.
They are called by :meth:`Data3d.plot_pyvista` and
:meth:`VectorData.plot_pyvista`.
"""

import importlib
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

import emout.utils as utils

_SPATIAL_AXES = ("x", "y", "z")
_AXIS_TO_INDEX = {"t": 0, "z": 1, "y": 2, "x": 3}


def _require_pyvista():
    """Lazily import and return the pyvista module.

    Raises
    ------
    ModuleNotFoundError
        If pyvista is not installed.
    """
    try:
        return importlib.import_module("pyvista")
    except Exception as exc:
        raise ModuleNotFoundError(
            "pyvista is required for 3D plotting. Install it via `pip install pyvista`."
        ) from exc


def _offseted(line: np.ndarray, offset: Union[float, str, None]) -> np.ndarray:
    """Apply a positional offset to a coordinate array.

    Parameters
    ----------
    line : np.ndarray
        1-D array of coordinate values.
    offset : float or str or None
        Offset specification. A float is added directly. String values
        ``'left'``, ``'center'``, and ``'right'`` shift so that the
        first, middle, or last element becomes zero.

    Returns
    -------
    np.ndarray
        A new array with the offset applied.
    """
    line = np.array(line, dtype=float, copy=True)
    if offset is None:
        return line
    if offset == "left":
        line -= line.ravel()[0]
    elif offset == "center":
        line -= line.ravel()[line.size // 2]
    elif offset == "right":
        line -= line.ravel()[-1]
    elif isinstance(offset, (float, int)):
        line += float(offset)
    else:
        raise ValueError(f"Invalid offset: {offset}")
    return line


def _as_scalar_array(data, use_si: bool) -> np.ndarray:
    """Return the data values, optionally converted to SI units.

    Parameters
    ----------
    data : Data2d or Data3d
        Grid data object with optional ``valunit`` attribute.
    use_si : bool
        If True and ``data.valunit`` is available, convert values to SI.

    Returns
    -------
    np.ndarray
        Scalar values as a float array.
    """
    if use_si and getattr(data, "valunit", None) is not None:
        return np.asarray(data.valunit.reverse(data), dtype=float)
    return np.asarray(data, dtype=float)


def _axis_values(data, use_si: bool, offsets=None):
    """Build coordinate values for each spatial axis with optional SI conversion and offset.

    Parameters
    ----------
    data : Data2d or Data3d
        Grid data object carrying slice and axis-unit metadata.
    use_si : bool
        If True and axis units are available, convert coordinates to SI.
    offsets : tuple of (float or str or None), optional
        Per-axis ``(x, y, z)`` positional offsets forwarded to
        :func:`_offseted`.

    Returns
    -------
    coords : dict[str, np.ndarray]
        Mapping from axis name (``'x'``, ``'y'``, ``'z'``) to coordinate
        arrays.
    labels : dict[str, str]
        Mapping from axis name to a human-readable label string.
    """
    coords = {}
    labels = {}
    if offsets is None:
        offsets = (None, None, None)
    offset_map = {"x": offsets[0], "y": offsets[1], "z": offsets[2]}

    for axis in _SPATIAL_AXES:
        axis_idx = _AXIS_TO_INDEX[axis]
        values = np.arange(*utils.slice2tuple(data.slices[axis_idx]), dtype=float)
        axisunits = getattr(data, "axisunits", None)
        if use_si and axisunits is not None and axisunits[axis_idx] is not None:
            axisunit = axisunits[axis_idx]
            values = axisunit.reverse(values)
            labels[axis] = f"{axis} [{axisunit.unit}]"
        else:
            labels[axis] = axis
        coords[axis] = _offseted(values, offset_map[axis])

    return coords, labels


def _scalar_label(data, use_si: bool) -> str:
    """Return a human-readable label including units if *use_si* is True.

    Parameters
    ----------
    data : Data2d or Data3d
        Grid data object with optional ``valunit`` attribute.
    use_si : bool
        If True and value units are available, append the unit string.

    Returns
    -------
    str
        Label of the form ``"name [unit]"`` or just ``"name"``.
    """
    name = data.name or "value"
    if use_si and getattr(data, "valunit", None) is not None:
        return f"{name} [{data.valunit.unit}]"
    return name


def _spacing(values: np.ndarray) -> float:
    """Compute uniform grid spacing from a 1-D coordinate array.

    Parameters
    ----------
    values : np.ndarray
        1-D array of coordinate values (assumed uniformly spaced).

    Returns
    -------
    float
        The spacing between consecutive values, or ``1.0`` when the
        array has fewer than two elements.
    """
    if len(values) <= 1:
        return 1.0
    return float(values[1] - values[0])


def _show_bounds(plotter, axis_labels):
    """Annotate a PyVista plotter with axis labels and tick marks.

    Parameters
    ----------
    plotter : pyvista.Plotter
        The plotter instance to annotate.
    axis_labels : dict[str, str]
        Mapping from ``'x'``, ``'y'``, ``'z'`` to label strings.
    """
    try:
        plotter.show_bounds(
            xlabel=axis_labels["x"],
            ylabel=axis_labels["y"],
            zlabel=axis_labels["z"],
        )
    except TypeError:
        plotter.show_bounds()


