"""PyVista scalar field plotting helpers."""

from typing import Literal, Union

import numpy as np

from ._pyvista_helpers import (
    _SPATIAL_AXES,
    _as_scalar_array,
    _axis_values,
    _require_pyvista,
    _scalar_label,
    _show_bounds,
    _spacing,
)


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
        raise ValueError(f"plot_pyvista for Data3d requires spatial axes x,y,z. got: {axes}")

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
