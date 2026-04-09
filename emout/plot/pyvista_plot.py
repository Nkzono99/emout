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


def create_plane_mesh(data2d, use_si=True, offsets=None, scalar_name=None):
    """Create a PyVista plane mesh from a 2-D scalar field.

    Parameters
    ----------
    data2d : Data2d
        2-D grid data to visualise.
    use_si : bool, optional
        Convert coordinates and values to SI units.
    offsets : tuple of (float or str or None), optional
        Per-axis positional offsets ``(x, y, z)``.
    scalar_name : str, optional
        Name for the scalar data array attached to the mesh.  Defaults
        to ``data2d.name``.

    Returns
    -------
    mesh : pyvista.StructuredGrid
        Plane mesh with scalar values attached.
    scalar_name : str
        Name of the scalar array on the mesh.
    axis_labels : dict[str, str]
        Axis label strings.
    scalar_label : str
        Human-readable scalar label including units.
    """
    if len(data2d.use_axes) != 2:
        raise ValueError("plot_pyvista for Data2d requires 2D data.")
    if "t" in data2d.use_axes:
        raise ValueError("Data2d with time axis is not supported by plot_pyvista.")

    axes = tuple(data2d.use_axes)
    axis0, axis1 = axes
    if axis0 not in _SPATIAL_AXES or axis1 not in _SPATIAL_AXES:
        raise ValueError(f"Unsupported axes for Data2d plot_pyvista: {axes}")

    coords, axis_labels = _axis_values(data2d, use_si=use_si, offsets=offsets)
    c0 = coords[axis0]
    c1 = coords[axis1]
    n0, n1 = len(c0), len(c1)

    mesh0 = np.broadcast_to(c0[:, None], (n0, n1))
    mesh1 = np.broadcast_to(c1[None, :], (n0, n1))

    missing = list(set(_SPATIAL_AXES) - set(axes))[0]
    missing_value = coords[missing][0]
    missing_mesh = np.full((n0, n1), missing_value, dtype=float)

    xyz_mesh = {axis0: mesh0, axis1: mesh1, missing: missing_mesh}
    xmesh = xyz_mesh["x"]
    ymesh = xyz_mesh["y"]
    zmesh = xyz_mesh["z"]

    pv = _require_pyvista()
    grid = pv.StructuredGrid(xmesh, ymesh, zmesh)

    scalars = _as_scalar_array(data2d, use_si=use_si)
    scalar_name = scalar_name or (data2d.name or "value")
    grid.point_data[scalar_name] = scalars.ravel(order="F")

    return grid, scalar_name, axis_labels, _scalar_label(data2d, use_si=use_si)


def create_volume_mesh(data3d, use_si=True, offsets=None, scalar_name=None):
    """Create a PyVista volume mesh from a 3-D scalar field.

    Parameters
    ----------
    data3d : Data3d
        3-D grid data to visualise.
    use_si : bool, optional
        Convert coordinates and values to SI units.
    offsets : tuple of (float or str or None), optional
        Per-axis positional offsets ``(x, y, z)``.
    scalar_name : str, optional
        Name for the scalar data array attached to the mesh.  Defaults
        to ``data3d.name``.

    Returns
    -------
    mesh : pyvista.ImageData
        Volume mesh with scalar values attached.
    scalar_name : str
        Name of the scalar array on the mesh.
    axis_labels : dict[str, str]
        Axis label strings.
    scalar_label : str
        Human-readable scalar label including units.
    """
    axes = tuple(data3d.use_axes)
    if set(axes) != set(_SPATIAL_AXES):
        raise ValueError(
            f"plot_pyvista for Data3d requires spatial axes x,y,z. got: {axes}"
        )

    coords, axis_labels = _axis_values(data3d, use_si=use_si, offsets=offsets)
    x = coords["x"]
    y = coords["y"]
    z = coords["z"]

    pv = _require_pyvista()
    mesh = pv.ImageData(
        dimensions=(len(x), len(y), len(z)),
        spacing=(_spacing(x), _spacing(y), _spacing(z)),
        origin=(x[0], y[0], z[0]),
    )

    data = _as_scalar_array(data3d, use_si=use_si)
    data_zyx = np.transpose(
        data,
        (
            axes.index("z"),
            axes.index("y"),
            axes.index("x"),
        ),
    )

    scalar_name = scalar_name or (data3d.name or "value")
    mesh.point_data[scalar_name] = data_zyx.transpose(2, 1, 0).ravel(order="F")
    return mesh, scalar_name, axis_labels, _scalar_label(data3d, use_si=use_si)


def plot_scalar_plane(
    data2d,
    plotter=None,
    use_si=True,
    offsets=None,
    show=False,
    cmap="viridis",
    clim=None,
    scalar_name=None,
    show_edges=False,
    add_scalar_bar=True,
    **kwargs,
):
    """Plot a 2-D scalar field on a PyVista plane.

    Parameters
    ----------
    data2d : Data2d
        2-D grid data to visualise.
    plotter : pyvista.Plotter, optional
        Existing plotter to draw into.  A new one is created if *None*.
    use_si : bool, optional
        Convert coordinates and values to SI units.
    offsets : tuple of (float or str or None), optional
        Per-axis positional offsets ``(x, y, z)``.
    show : bool, optional
        If True, call ``plotter.show()`` before returning.
    cmap : str, optional
        Colour-map name forwarded to PyVista.
    clim : tuple of float, optional
        ``(vmin, vmax)`` colour limits.
    scalar_name : str, optional
        Name for the scalar data array on the mesh.
    show_edges : bool, optional
        Show mesh edge lines.
    add_scalar_bar : bool, optional
        Add a colour-bar to the plotter.
    **kwargs
        Additional keyword arguments forwarded to
        ``plotter.add_mesh``.

    Returns
    -------
    pyvista.Plotter
        The plotter instance used for rendering.
    """
    pv = _require_pyvista()
    if plotter is None:
        plotter = pv.Plotter()

    mesh, scalar_name, axis_labels, scalar_label = create_plane_mesh(
        data2d,
        use_si=use_si,
        offsets=offsets,
        scalar_name=scalar_name,
    )

    add_mesh_kwargs = dict(
        scalars=scalar_name,
        cmap=cmap,
        show_edges=show_edges,
        show_scalar_bar=False,
    )
    if clim is not None:
        add_mesh_kwargs["clim"] = clim
    add_mesh_kwargs.update(kwargs)

    plotter.add_mesh(mesh, **add_mesh_kwargs)
    if add_scalar_bar:
        plotter.add_scalar_bar(title=scalar_label)
    _show_bounds(plotter, axis_labels)
    plotter.add_axes()

    if show:
        plotter.show()
    return plotter


def plot_scalar_volume(
    data3d,
    mode: Literal["box", "volume", "slice", "contour"] = "box",
    plotter=None,
    use_si=True,
    offsets=None,
    show=False,
    cmap="viridis",
    clim=None,
    opacity: Union[float, str] = "sigmoid",
    contour_levels: Union[int, np.ndarray] = 8,
    scalar_name=None,
    add_outline=True,
    outline_color="white",
    add_scalar_bar=True,
    show_edges=False,
    **kwargs,
):
    """Plot a 3-D scalar field as a volume rendering.

    Parameters
    ----------
    data3d : Data3d
        3-D grid data to visualise.
    mode : {'box', 'volume', 'slice', 'contour'}, optional
        Rendering mode.  ``'box'`` extracts the outer surface,
        ``'volume'`` uses GPU volume rendering, ``'slice'`` shows
        orthogonal slices, and ``'contour'`` draws iso-surfaces.
    plotter : pyvista.Plotter, optional
        Existing plotter to draw into.  A new one is created if *None*.
    use_si : bool, optional
        Convert coordinates and values to SI units.
    offsets : tuple of (float or str or None), optional
        Per-axis positional offsets ``(x, y, z)``.
    show : bool, optional
        If True, call ``plotter.show()`` before returning.
    cmap : str, optional
        Colour-map name forwarded to PyVista.
    clim : tuple of float, optional
        ``(vmin, vmax)`` colour limits.
    opacity : float or str, optional
        Opacity for volume rendering.  Can be a transfer function name
        such as ``'sigmoid'``.
    contour_levels : int or np.ndarray, optional
        Number of iso-surfaces (or explicit levels) when
        *mode='contour'*.
    scalar_name : str, optional
        Name for the scalar data array on the mesh.
    add_outline : bool, optional
        Draw an outline box around the volume.
    outline_color : str, optional
        Colour of the outline box.
    add_scalar_bar : bool, optional
        Add a colour-bar to the plotter.
    show_edges : bool, optional
        Show mesh edge lines (used in ``'box'`` mode).
    **kwargs
        Additional keyword arguments forwarded to
        ``plotter.add_mesh`` or ``plotter.add_volume``.

    Returns
    -------
    pyvista.Plotter
        The plotter instance used for rendering.
    """
    pv = _require_pyvista()
    if plotter is None:
        plotter = pv.Plotter()

    mesh, scalar_name, axis_labels, scalar_label = create_volume_mesh(
        data3d,
        use_si=use_si,
        offsets=offsets,
        scalar_name=scalar_name,
    )

    if mode == "box":
        surface = mesh.extract_surface()
        add_mesh_kwargs = dict(
            scalars=scalar_name,
            cmap=cmap,
            show_edges=show_edges,
            show_scalar_bar=False,
        )
        if clim is not None:
            add_mesh_kwargs["clim"] = clim
        if isinstance(opacity, (float, int)):
            add_mesh_kwargs["opacity"] = float(opacity)
        add_mesh_kwargs.update(kwargs)
        plotter.add_mesh(surface, **add_mesh_kwargs)
    elif mode == "volume":
        add_volume_kwargs = dict(
            scalars=scalar_name,
            cmap=cmap,
            opacity=opacity,
            show_scalar_bar=False,
        )
        if clim is not None:
            add_volume_kwargs["clim"] = clim
        add_volume_kwargs.update(kwargs)
        plotter.add_volume(mesh, **add_volume_kwargs)
    elif mode == "slice":
        sliced = mesh.slice_orthogonal()
        add_mesh_kwargs = dict(
            scalars=scalar_name,
            cmap=cmap,
            show_scalar_bar=False,
        )
        if clim is not None:
            add_mesh_kwargs["clim"] = clim
        add_mesh_kwargs.update(kwargs)
        plotter.add_mesh(sliced, **add_mesh_kwargs)
    elif mode == "contour":
        contour = mesh.contour(isosurfaces=contour_levels, scalars=scalar_name)
        add_mesh_kwargs = dict(
            scalars=scalar_name,
            cmap=cmap,
            show_scalar_bar=False,
        )
        if clim is not None:
            add_mesh_kwargs["clim"] = clim
        add_mesh_kwargs.update(kwargs)
        plotter.add_mesh(contour, **add_mesh_kwargs)
    else:
        raise ValueError(f'Unsupported mode "{mode}" for Data3d plot_pyvista.')

    if add_outline:
        plotter.add_mesh(mesh.outline(), color=outline_color)
    if add_scalar_bar:
        plotter.add_scalar_bar(title=scalar_label)
    _show_bounds(plotter, axis_labels)
    plotter.add_axes()

    if show:
        plotter.show()
    return plotter


def create_vector_mesh3d(
    x_data3d,
    y_data3d,
    z_data3d,
    use_si=True,
    offsets=None,
    vector_name="vectors",
    magnitude_name="magnitude",
):
    """Create a PyVista mesh from three vector-component arrays.

    Parameters
    ----------
    x_data3d, y_data3d, z_data3d : Data3d
        X-, Y-, and Z-component 3-D grid data.  All three must share
        the same shape and spatial axes.
    use_si : bool, optional
        Convert coordinates and values to SI units.
    offsets : tuple of (float or str or None), optional
        Per-axis positional offsets ``(x, y, z)``.
    vector_name : str, optional
        Name for the vector data array attached to the mesh.
    magnitude_name : str, optional
        Name for the magnitude scalar array attached to the mesh.

    Returns
    -------
    mesh : pyvista.ImageData
        Volume mesh with vector and magnitude arrays.
    vector_name : str
        Name of the vector array on the mesh.
    magnitude_name : str
        Name of the magnitude array on the mesh.
    axis_labels : dict[str, str]
        Axis label strings.
    """
    if x_data3d.shape != y_data3d.shape or x_data3d.shape != z_data3d.shape:
        raise ValueError("All vector components must have the same shape.")

    axes = tuple(x_data3d.use_axes)
    if set(axes) != set(_SPATIAL_AXES):
        raise ValueError(
            f"plot_pyvista for VectorData3d requires spatial axes x,y,z. got: {axes}"
        )

    coords, axis_labels = _axis_values(x_data3d, use_si=use_si, offsets=offsets)
    x = coords["x"]
    y = coords["y"]
    z = coords["z"]

    pv = _require_pyvista()
    mesh = pv.ImageData(
        dimensions=(len(x), len(y), len(z)),
        spacing=(_spacing(x), _spacing(y), _spacing(z)),
        origin=(x[0], y[0], z[0]),
    )

    def _component_values(component):
        """Flatten one vector component into F-order for PyVista."""
        if use_si and getattr(component, "valunit", None) is not None:
            arr = np.asarray(component.valunit.reverse(component), dtype=float)
        else:
            arr = np.asarray(component, dtype=float)
        arr_zyx = np.transpose(
            arr,
            (
                axes.index("z"),
                axes.index("y"),
                axes.index("x"),
            ),
        )
        return arr_zyx.transpose(2, 1, 0).ravel(order="F")

    ux = _component_values(x_data3d)
    uy = _component_values(y_data3d)
    uz = _component_values(z_data3d)

    vectors = np.column_stack((ux, uy, uz))
    mesh.point_data[vector_name] = vectors
    mesh.point_data[magnitude_name] = np.linalg.norm(vectors, axis=1)

    return mesh, vector_name, magnitude_name, axis_labels


def plot_vector_quiver3d(
    x_data3d,
    y_data3d,
    z_data3d,
    plotter=None,
    use_si=True,
    offsets=None,
    show=False,
    skip: Union[int, Tuple[int, int, int]] = 2,
    scale_by_magnitude=True,
    factor: Optional[float] = None,
    cmap="viridis",
    clim=None,
    add_scalar_bar=True,
    **kwargs,
):
    """Plot a 3-D vector field as quiver arrows.

    Parameters
    ----------
    x_data3d, y_data3d, z_data3d : Data3d
        X-, Y-, and Z-component 3-D grid data.
    plotter : pyvista.Plotter, optional
        Existing plotter to draw into.  A new one is created if *None*.
    use_si : bool, optional
        Convert coordinates and values to SI units.
    offsets : tuple of (float or str or None), optional
        Per-axis positional offsets ``(x, y, z)``.
    show : bool, optional
        If True, call ``plotter.show()`` before returning.
    skip : int or tuple of int, optional
        Down-sampling factor for quiver arrows.  A single integer
        applies to all axes; a 3-tuple ``(skip_x, skip_y, skip_z)``
        allows per-axis control.
    scale_by_magnitude : bool, optional
        Scale arrow length by vector magnitude.
    factor : float, optional
        Glyph scale factor.  When *None* an automatic value is
        computed from the domain size and mean magnitude.
    cmap : str, optional
        Colour-map name forwarded to PyVista.
    clim : tuple of float, optional
        ``(vmin, vmax)`` colour limits.
    add_scalar_bar : bool, optional
        Add a colour-bar to the plotter.
    **kwargs
        Additional keyword arguments forwarded to
        ``plotter.add_mesh``.

    Returns
    -------
    pyvista.Plotter
        The plotter instance used for rendering.
    """
    pv = _require_pyvista()
    if plotter is None:
        plotter = pv.Plotter()

    mesh, vector_name, magnitude_name, axis_labels = create_vector_mesh3d(
        x_data3d,
        y_data3d,
        z_data3d,
        use_si=use_si,
        offsets=offsets,
    )

    if isinstance(skip, int):
        skip_x = skip_y = skip_z = max(skip, 1)
    else:
        skip_x = max(skip[0], 1)
        skip_y = max(skip[1], 1)
        skip_z = max(skip[2], 1)

    nx, ny, nz = mesh.dimensions
    ids = np.arange(mesh.n_points).reshape((nx, ny, nz), order="F")
    ids = ids[::skip_x, ::skip_y, ::skip_z].ravel(order="F")

    points = mesh.points[ids]
    vectors = mesh.point_data[vector_name][ids]
    magnitudes = mesh.point_data[magnitude_name][ids]

    sampled = pv.PolyData(points)
    sampled.point_data[vector_name] = vectors
    sampled.point_data[magnitude_name] = magnitudes

    if factor is None:
        lengths = np.array(
            [
                mesh.bounds[1] - mesh.bounds[0],
                mesh.bounds[3] - mesh.bounds[2],
                mesh.bounds[5] - mesh.bounds[4],
            ]
        )
        diag = float(np.linalg.norm(lengths))
        mag_mean = float(np.nanmean(magnitudes)) if len(magnitudes) else 0.0
        if mag_mean > 0:
            factor = 0.07 * diag / mag_mean
        else:
            factor = 1.0

    scale_name = magnitude_name if scale_by_magnitude else False
    glyph = sampled.glyph(orient=vector_name, scale=scale_name, factor=factor)

    add_mesh_kwargs = dict(show_scalar_bar=False)
    if cmap is not None and magnitude_name in glyph.array_names:
        add_mesh_kwargs["scalars"] = magnitude_name
        add_mesh_kwargs["cmap"] = cmap
    if clim is not None:
        add_mesh_kwargs["clim"] = clim
    add_mesh_kwargs.update(kwargs)

    plotter.add_mesh(glyph, **add_mesh_kwargs)
    if add_scalar_bar and cmap is not None and magnitude_name in glyph.array_names:
        plotter.add_scalar_bar(title="|v|")
    _show_bounds(plotter, axis_labels)
    plotter.add_axes()

    if show:
        plotter.show()
    return plotter


def plot_vector_streamlines3d(
    x_data3d,
    y_data3d,
    z_data3d,
    plotter=None,
    use_si=True,
    offsets=None,
    show=False,
    n_points=200,
    source_radius=None,
    source_center=None,
    tube_radius=None,
    cmap="viridis",
    clim=None,
    add_scalar_bar=True,
    **kwargs,
):
    """Plot a 3-D vector field as streamlines.

    Parameters
    ----------
    x_data3d, y_data3d, z_data3d : Data3d
        X-, Y-, and Z-component 3-D grid data.
    plotter : pyvista.Plotter, optional
        Existing plotter to draw into.  A new one is created if *None*.
    use_si : bool, optional
        Convert coordinates and values to SI units.
    offsets : tuple of (float or str or None), optional
        Per-axis positional offsets ``(x, y, z)``.
    show : bool, optional
        If True, call ``plotter.show()`` before returning.
    n_points : int, optional
        Number of seed points for streamline integration.
    source_radius : float, optional
        Radius of the spherical seed source.  Defaults to 25 % of the
        smallest domain extent.
    source_center : tuple of float, optional
        Centre of the seed source.  Defaults to the mesh centre.
    tube_radius : float, optional
        If given, render streamlines as tubes with this radius.
    cmap : str, optional
        Colour-map name forwarded to PyVista.
    clim : tuple of float, optional
        ``(vmin, vmax)`` colour limits.
    add_scalar_bar : bool, optional
        Add a colour-bar to the plotter.
    **kwargs
        Additional keyword arguments forwarded to
        ``mesh.streamlines``.

    Returns
    -------
    pyvista.Plotter
        The plotter instance used for rendering.

    Raises
    ------
    RuntimeError
        If no streamlines could be generated with the given parameters.
    """
    pv = _require_pyvista()
    if plotter is None:
        plotter = pv.Plotter()

    mesh, vector_name, magnitude_name, axis_labels = create_vector_mesh3d(
        x_data3d,
        y_data3d,
        z_data3d,
        use_si=use_si,
        offsets=offsets,
    )

    lengths = np.array(
        [
            mesh.bounds[1] - mesh.bounds[0],
            mesh.bounds[3] - mesh.bounds[2],
            mesh.bounds[5] - mesh.bounds[4],
        ]
    )
    if source_center is None:
        source_center = mesh.center
    if source_radius is None:
        source_radius = float(0.25 * np.min(lengths))

    streamline = mesh.streamlines(
        vectors=vector_name,
        source_center=source_center,
        source_radius=source_radius,
        n_points=n_points,
        **kwargs,
    )
    if streamline.n_points == 0:
        raise RuntimeError(
            "No streamlines were generated. Try increasing n_points/source_radius."
        )

    if magnitude_name not in streamline.array_names and vector_name in streamline.array_names:
        vecs = np.asarray(streamline[vector_name])
        streamline[magnitude_name] = np.linalg.norm(vecs, axis=1)

    stream_mesh = streamline
    if tube_radius is not None:
        stream_mesh = streamline.tube(radius=tube_radius)

    add_mesh_kwargs = dict(show_scalar_bar=False)
    if cmap is not None and magnitude_name in stream_mesh.array_names:
        add_mesh_kwargs["scalars"] = magnitude_name
        add_mesh_kwargs["cmap"] = cmap
    if clim is not None:
        add_mesh_kwargs["clim"] = clim
    add_mesh_kwargs.update(kwargs)

    plotter.add_mesh(stream_mesh, **add_mesh_kwargs)
    if add_scalar_bar and cmap is not None and magnitude_name in stream_mesh.array_names:
        plotter.add_scalar_bar(title="|v|")
    plotter.add_mesh(mesh.outline(), color="white")
    _show_bounds(plotter, axis_labels)
    plotter.add_axes()

    if show:
        plotter.show()
    return plotter
