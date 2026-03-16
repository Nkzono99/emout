import importlib
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

import emout.utils as utils

_SPATIAL_AXES = ("x", "y", "z")
_AXIS_TO_INDEX = {"t": 0, "z": 1, "y": 2, "x": 3}


def _require_pyvista():
    """pyvista を遅延 import して返す。"""
    try:
        return importlib.import_module("pyvista")
    except Exception as exc:
        raise ModuleNotFoundError(
            "pyvista is required for 3D plotting. Install it via `pip install pyvista`."
        ) from exc


def _offseted(line: np.ndarray, offset: Union[float, str, None]) -> np.ndarray:
    """位置指定を実座標オフセットへ変換する。"""
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
    if use_si and getattr(data, "valunit", None) is not None:
        return np.asarray(data.valunit.reverse(data), dtype=float)
    return np.asarray(data, dtype=float)


def _axis_values(data, use_si: bool, offsets=None):
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
    name = data.name or "value"
    if use_si and getattr(data, "valunit", None) is not None:
        return f"{name} [{data.valunit.unit}]"
    return name


def _spacing(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 1.0
    return float(values[1] - values[0])


def _show_bounds(plotter, axis_labels):
    try:
        plotter.show_bounds(
            xlabel=axis_labels["x"],
            ylabel=axis_labels["y"],
            zlabel=axis_labels["z"],
        )
    except TypeError:
        plotter.show_bounds()


def create_plane_mesh(data2d, use_si=True, offsets=None, scalar_name=None):
    """Data2d 用の平面 StructuredGrid を生成する。"""
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
    """Data3d 用の ImageData を生成する。"""
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
    """Data2d を 3D 空間上の平面として pyvista 描画する。"""
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
    """Data3d を pyvista で描画する。"""
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
    """3成分 Data3d から pyvista.ImageData ベクトル場を生成する。"""
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
    """3次元 quiver を pyvista で描画する。"""
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
    """3次元流線を pyvista で描画する。"""
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
