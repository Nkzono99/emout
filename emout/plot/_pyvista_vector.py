"""PyVista vector field plotting helpers."""

from typing import Optional, Tuple, Union

import numpy as np

from ._pyvista_helpers import (
    _SPATIAL_AXES,
    _add_surface_overlays,
    _surface_items,
    _surface_to_polydata,
    _axis_values,
    _require_pyvista,
    _save_or_show_plotter,
    _show_bounds,
    _spacing,
)

_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


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
        raise ValueError(f"plot_pyvista for VectorData3d requires spatial axes x,y,z. got: {axes}")

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


def _seed_resolution(seed_resolution, *, dims: int, n_points: int):
    """Return per-axis seed counts for a regular seed grid."""
    if seed_resolution is None:
        count = max(int(np.ceil(float(n_points) ** (1.0 / dims))), 1)
        return (count,) * dims
    if isinstance(seed_resolution, int):
        return (max(int(seed_resolution), 1),) * dims
    if len(seed_resolution) != dims:
        raise ValueError(f"seed_resolution must have {dims} entries for this seed mode.")
    return tuple(max(int(value), 1) for value in seed_resolution)


def _axis_points(mesh, axis: str, count: int):
    """Return regular points along one mesh axis."""
    axis_idx = _AXIS_TO_INDEX[axis]
    lower = mesh.bounds[2 * axis_idx]
    upper = mesh.bounds[2 * axis_idx + 1]
    return np.linspace(lower, upper, count)


def _as_seed_points(points):
    """Validate and return an explicit seed point array."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"seed_points must have shape (n, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("seed_points must contain at least one point.")
    return points


def _plane_position(mesh, axis: str, seed_position, source_center=None):
    """Resolve the fixed coordinate for a seed plane."""
    if seed_position is None:
        seed_position = "center"
    if isinstance(seed_position, str):
        if seed_position != "center":
            raise ValueError('seed_position must be "center" or a numeric coordinate.')
        if source_center is not None:
            return float(source_center[_AXIS_TO_INDEX[axis]])
        return float(mesh.center[_AXIS_TO_INDEX[axis]])
    return float(seed_position)


def _volume_seed_points(mesh, *, n_points: int, seed_resolution=None):
    """Generate regular seed points across the full mesh volume."""
    nx, ny, nz = _seed_resolution(seed_resolution, dims=3, n_points=n_points)
    x = _axis_points(mesh, "x", nx)
    y = _axis_points(mesh, "y", ny)
    z = _axis_points(mesh, "z", nz)
    grids = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([grid.ravel() for grid in grids])


def _plane_seed_points(
    mesh, *, seed_plane: str, seed_position="center", n_points: int, seed_resolution=None, source_center=None
):
    """Generate regular seed points on one coordinate plane."""
    plane = seed_plane.lower()
    if len(plane) != 2 or len(set(plane)) != 2 or any(axis not in _AXIS_TO_INDEX for axis in plane):
        raise ValueError('seed_plane must be one of "xy", "xz", "yz" or reversed equivalents.')

    axes = tuple(plane)
    fixed_axis = next(axis for axis in _SPATIAL_AXES if axis not in axes)
    n0, n1 = _seed_resolution(seed_resolution, dims=2, n_points=n_points)
    values0 = _axis_points(mesh, axes[0], n0)
    values1 = _axis_points(mesh, axes[1], n1)
    mesh0, mesh1 = np.meshgrid(values0, values1, indexing="ij")

    points = np.empty((mesh0.size, 3), dtype=float)
    points[:, _AXIS_TO_INDEX[axes[0]]] = mesh0.ravel()
    points[:, _AXIS_TO_INDEX[axes[1]]] = mesh1.ravel()
    points[:, _AXIS_TO_INDEX[fixed_axis]] = _plane_position(
        mesh,
        fixed_axis,
        seed_position,
        source_center=source_center,
    )
    return points


def _surface_seed_points(
    pv,
    *,
    seed_surface,
    n_points: int,
    use_si: bool = True,
    offsets=None,
    surface_per=None,
):
    """Generate seed points from one or more mesh-like surfaces."""
    if seed_surface is None:
        raise ValueError("seed_mode='surface' requires seed_surface or surfaces.")

    point_arrays = []
    for item in _surface_items(seed_surface):
        poly = _surface_to_polydata(item, use_si=use_si, offsets=offsets, per=surface_per)
        point_arrays.append(np.asarray(poly.points, dtype=float))

    points = np.vstack(point_arrays) if point_arrays else np.empty((0, 3), dtype=float)
    points = _as_seed_points(points)
    if n_points is not None and len(points) > n_points:
        indices = np.linspace(0, len(points) - 1, int(n_points), dtype=int)
        points = points[indices]
    return pv.PolyData(points)


def _make_streamline_source(
    mesh,
    pv,
    *,
    seed_mode="sphere",
    n_points=200,
    source_center=None,
    seed_points=None,
    seed_plane="xy",
    seed_position="center",
    seed_resolution=None,
    seed_surface=None,
    surfaces=None,
    use_si=True,
    offsets=None,
    surface_per=None,
):
    """Create a PyVista source object for non-spherical streamline seeds."""
    if seed_points is not None:
        return pv.PolyData(_as_seed_points(seed_points))

    mode = seed_mode.lower()
    if mode == "sphere":
        return None
    if mode == "volume":
        return pv.PolyData(_volume_seed_points(mesh, n_points=n_points, seed_resolution=seed_resolution))
    if mode == "plane":
        return pv.PolyData(
            _plane_seed_points(
                mesh,
                seed_plane=seed_plane,
                seed_position=seed_position,
                n_points=n_points,
                seed_resolution=seed_resolution,
                source_center=source_center,
            )
        )
    if mode == "surface":
        surface = seed_surface if seed_surface is not None else surfaces
        return _surface_seed_points(
            pv,
            seed_surface=surface,
            n_points=n_points,
            use_si=use_si,
            offsets=offsets,
            surface_per=surface_per,
        )
    raise ValueError(f'Unsupported seed_mode "{seed_mode}" for VectorData3d streamlines.')


def _tube_streamline_mesh(streamline, *, tube_radius, magnitude_name, default_radius, tube_radius_factor=10.0):
    """Return a fixed- or variable-radius tube mesh for streamlines."""
    if tube_radius is None:
        return streamline
    if isinstance(tube_radius, str):
        mode = tube_radius.lower()
        if mode not in ("auto", "magnitude"):
            raise ValueError('tube_radius must be numeric, None, "auto", or "magnitude".')
        return streamline.tube(
            radius=default_radius,
            scalars=magnitude_name,
            radius_factor=float(tube_radius_factor),
        )
    return streamline.tube(radius=float(tube_radius))


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
    surfaces=None,
    surface_color="0.7",
    surface_opacity=0.35,
    surface_per=None,
    surface_kwargs=None,
    filename=None,
    savefilename=None,
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
    surfaces : object, optional
        Boundary or mesh surface overlay to draw as a solid transparent mesh.
    surface_color, surface_opacity : optional
        Default style for ``surfaces``. ``RenderItem`` inputs override these.
    surface_per : dict, optional
        Per-boundary mesh overrides when ``surfaces`` is a
        ``BoundaryCollection``.
    surface_kwargs : dict, optional
        Additional keyword arguments forwarded to PyVista for surface
        overlays.
    filename, savefilename : str, optional
        Save the rendered scene. Static image suffixes use
        ``plotter.screenshot``; ``.html`` uses ``plotter.export_html``.
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
    _add_surface_overlays(
        plotter,
        surfaces,
        use_si=use_si,
        offsets=offsets,
        per=surface_per,
        surface_color=surface_color,
        surface_opacity=surface_opacity,
        **(surface_kwargs or {}),
    )
    _show_bounds(plotter, axis_labels)
    plotter.add_axes()

    return _save_or_show_plotter(plotter, show=show, filename=filename, savefilename=savefilename)


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
    seed_mode="sphere",
    seed_points=None,
    seed_plane="xy",
    seed_position="center",
    seed_resolution=None,
    seed_surface=None,
    tube_radius=None,
    tube_radius_factor=10.0,
    cmap="viridis",
    clim=None,
    add_scalar_bar=True,
    surfaces=None,
    surface_color="0.7",
    surface_opacity=0.35,
    surface_per=None,
    surface_kwargs=None,
    filename=None,
    savefilename=None,
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
    seed_mode : {'sphere', 'volume', 'plane', 'surface'}, optional
        Seed placement mode. ``'sphere'`` keeps the historical PyVista
        spherical source, ``'volume'`` fills the domain with a regular
        seed grid, ``'plane'`` seeds one coordinate plane, and
        ``'surface'`` seeds mesh-surface vertices.
    seed_points : array-like, optional
        Explicit ``(n, 3)`` seed points. When supplied, this overrides
        ``seed_mode``.
    seed_plane : {'xy', 'xz', 'yz'}, optional
        Coordinate plane used by ``seed_mode='plane'``. Reversed forms
        such as ``'zx'`` are also accepted.
    seed_position : {'center'} or float, optional
        Fixed coordinate of the missing axis for ``seed_mode='plane'``.
    seed_resolution : int or tuple of int, optional
        Seed grid resolution. A 3-tuple is used for ``'volume'`` and a
        2-tuple for ``'plane'``. Defaults are derived from ``n_points``.
    seed_surface : object, optional
        Boundary or mesh-like object used by ``seed_mode='surface'``.
        Falls back to ``surfaces`` when omitted.
    tube_radius : float or {'auto', 'magnitude'}, optional
        If numeric, render fixed-radius tubes. ``'auto'`` and
        ``'magnitude'`` scale tube radius by vector magnitude.
    tube_radius_factor : float, optional
        Ratio between minimum and maximum tube radius for magnitude
        scaled tubes.
    cmap : str, optional
        Colour-map name forwarded to PyVista.
    clim : tuple of float, optional
        ``(vmin, vmax)`` colour limits.
    add_scalar_bar : bool, optional
        Add a colour-bar to the plotter.
    surfaces : object, optional
        Boundary or mesh surface overlay to draw as a solid transparent mesh.
    surface_color, surface_opacity : optional
        Default style for ``surfaces``. ``RenderItem`` inputs override these.
    surface_per : dict, optional
        Per-boundary mesh overrides when ``surfaces`` is a
        ``BoundaryCollection``.
    surface_kwargs : dict, optional
        Additional keyword arguments forwarded to PyVista for surface
        overlays.
    filename, savefilename : str, optional
        Save the rendered scene. Static image suffixes use
        ``plotter.screenshot``; ``.html`` uses ``plotter.export_html``.
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

    source = _make_streamline_source(
        mesh,
        pv,
        seed_mode=seed_mode,
        n_points=n_points,
        source_center=source_center,
        seed_points=seed_points,
        seed_plane=seed_plane,
        seed_position=seed_position,
        seed_resolution=seed_resolution,
        seed_surface=seed_surface,
        surfaces=surfaces,
        use_si=use_si,
        offsets=offsets,
        surface_per=surface_per,
    )
    if source is None:
        streamline = mesh.streamlines(
            vectors=vector_name,
            source_center=source_center,
            source_radius=source_radius,
            n_points=n_points,
            **kwargs,
        )
    else:
        streamline = mesh.streamlines_from_source(
            source,
            vectors=vector_name,
            **kwargs,
        )
    if streamline.n_points == 0:
        raise RuntimeError("No streamlines were generated. Try changing seed_mode, n_points, or source_radius.")

    if magnitude_name not in streamline.array_names and vector_name in streamline.array_names:
        vecs = np.asarray(streamline[vector_name])
        streamline[magnitude_name] = np.linalg.norm(vecs, axis=1)

    default_tube_radius = float(0.005 * np.min(lengths)) if np.min(lengths) > 0 else 1.0
    stream_mesh = _tube_streamline_mesh(
        streamline,
        tube_radius=tube_radius,
        magnitude_name=magnitude_name,
        default_radius=default_tube_radius,
        tube_radius_factor=tube_radius_factor,
    )

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
    _add_surface_overlays(
        plotter,
        surfaces,
        use_si=use_si,
        offsets=offsets,
        per=surface_per,
        surface_color=surface_color,
        surface_opacity=surface_opacity,
        **(surface_kwargs or {}),
    )
    _show_bounds(plotter, axis_labels)
    plotter.add_axes()

    return _save_or_show_plotter(plotter, show=show, filename=filename, savefilename=savefilename)
