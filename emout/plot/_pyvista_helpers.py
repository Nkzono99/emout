"""PyVista-based 3-D visualisation helpers for scalar and vector fields.

These helpers are called by :meth:`Data3d.plot_pyvista` and
:meth:`VectorData.plot_pyvista`.
"""

import importlib
import inspect
from typing import Union

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
        raise ModuleNotFoundError("pyvista is required for 3D plotting. Install it via `pip install pyvista`.") from exc


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
            xtitle=axis_labels["x"],
            ytitle=axis_labels["y"],
            ztitle=axis_labels["z"],
        )
    except AttributeError:
        return
    except TypeError:
        try:
            plotter.show_bounds(
                xlabel=axis_labels["x"],
                ylabel=axis_labels["y"],
                zlabel=axis_labels["z"],
            )
        except TypeError:
            plotter.show_bounds()


def _surface_items(surfaces):
    """Return surface overlay inputs as a list without splitting boundary collections."""
    if surfaces is None:
        return []
    if isinstance(surfaces, (list, tuple)):
        return list(surfaces)
    return [surfaces]


def _call_mesh(surface, *, use_si: bool, per=None):
    """Call ``surface.mesh`` with the kwargs accepted by that object."""
    mesh_method = getattr(surface, "mesh", None)
    if mesh_method is None:
        raise TypeError(f"Expected a mesh-like surface with a .mesh() method, got {type(surface).__name__}")

    kwargs = {}
    try:
        parameters = inspect.signature(mesh_method).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "use_si" in parameters:
        kwargs["use_si"] = use_si
    if per is not None and "per" in parameters:
        kwargs["per"] = per

    return mesh_method(**kwargs)


def _surface_mesh_arrays(surface, *, use_si: bool, per=None):
    """Extract ``(vertices, faces)`` from a mesh surface, boundary, or collection."""
    mesh_or_arrays = _call_mesh(surface, use_si=use_si, per=per)
    if isinstance(mesh_or_arrays, tuple) and len(mesh_or_arrays) == 2:
        return mesh_or_arrays

    mesh_method = getattr(mesh_or_arrays, "mesh", None)
    if mesh_method is None:
        raise TypeError(
            "surface.mesh() must return a (vertices, faces) tuple or another mesh-like object, "
            f"got {type(mesh_or_arrays).__name__}"
        )
    return mesh_method()


def _polydata_faces(faces: np.ndarray) -> np.ndarray:
    """Convert an ``(n, k)`` face array to PyVista's flat face format."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty(0, dtype=np.int64)
    if faces.ndim != 2:
        raise ValueError(f"Surface faces must be a 2-D array, got shape {faces.shape}")
    counts = np.full((faces.shape[0], 1), faces.shape[1], dtype=np.int64)
    return np.hstack((counts, faces)).ravel()


def _surface_to_polydata(surface, *, use_si: bool, offsets=None, per=None):
    """Convert a mesh surface, boundary, or collection into ``pyvista.PolyData``."""
    vertices, faces = _surface_mesh_arrays(surface, use_si=use_si, per=per)
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Surface vertices must have shape (n, 3), got {vertices.shape}")

    if offsets is not None:
        vertices = vertices.copy()
        for axis_idx, offset in enumerate(offsets[:3]):
            vertices[:, axis_idx] = _offseted(vertices[:, axis_idx], offset)

    pv = _require_pyvista()
    return pv.PolyData(vertices, _polydata_faces(faces))


def _add_surface_overlays(
    plotter,
    surfaces,
    *,
    use_si: bool = True,
    offsets=None,
    per=None,
    surface_color="0.7",
    surface_opacity=0.35,
    **kwargs,
):
    """Add solid mesh-surface overlays to a PyVista plotter."""
    if surfaces is None:
        return plotter

    from emout.plot.surface_cut import RenderItem

    for item in _surface_items(surfaces):
        color = surface_color
        opacity = surface_opacity
        surface = item
        if isinstance(item, RenderItem):
            surface = item.surface
            color = item.solid_color
            if item.alpha is not None:
                opacity = item.alpha

        poly = _surface_to_polydata(surface, use_si=use_si, offsets=offsets, per=per)
        add_mesh_kwargs = {
            "color": color,
            "opacity": opacity,
        }
        add_mesh_kwargs.update(kwargs)
        plotter.add_mesh(poly, **add_mesh_kwargs)

    return plotter
