"""PyVista vector field plotting helpers."""

from typing import Optional, Tuple, Union

import numpy as np

from ._pyvista_helpers import (
    _as_scalar_array,
    _axis_values,
    _offseted,
    _require_pyvista,
    _scalar_label,
    _show_bounds,
    _spacing,
)


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
