# PyVista Visualization (`plot3d` / `plot_pyvista`)

The PyVista backend is the 3-D visualization API for placing 2-D slices in 3-D space, rendering 3-D scalar fields as volume / slice / contour views, and drawing 3-D vector fields as streamlines or quiver arrows. Use the regular `plot()` / `cmap()` / `contour()` APIs from [Plotting](plotting.md) for ordinary 1-D/2-D analysis, and switch to PyVista when you need interactive 3-D camera control or overlays.

## Choosing an Entry Point

| Target | Recommended API | Return value |
| --- | --- | --- |
| 2D scalar slice | `data.phisp[-1, 100, :, :].plot3d(...)` | `pyvista.Plotter` |
| 3D scalar volume | `data.phisp[-1].plot3d(mode=...)` | `pyvista.Plotter` |
| 3D vector field | `data.j1xyz[-1].plot3d(mode=..., backend="pyvista")` | `pyvista.Plotter` |
| Boundary meshes | `data.boundaries.plot3d(plotter=...)` | `pyvista.Plotter` |
| Backtrace / trace paths | `trace.plot3d(plotter=...)` | `pyvista.Plotter` |
| Mesh construction only | `emout.plot.pyvista_plot.create_*_mesh(...)` | PyVista mesh object |

`Data2d.plot3d()` and `Data3d.plot3d()` are aliases for `plot_pyvista()`. `VectorData.plot3d()` uses the PyVista backend by default; pass `backend="mpl"` when you want the Matplotlib 3-D backend.

## Installation

PyVista is installed as a regular emout dependency, so no extra selector is needed for 3-D views.

```bash
pip install emout
```

If an old environment or editable install raises `ModuleNotFoundError`, reinstall emout so the dependency set is refreshed.

## Scalar Fields

### Place a 2D Slice in 3D Space

A 2-D slice is rendered as a `pyvista.StructuredGrid` plane. `plot3d()` returns the plotter, so you can adjust the camera or take screenshots afterwards.

```python
import emout

data = emout.Emout("output_dir")

plotter = data.phisp[-1, 100, :, :].plot3d(
    cmap="viridis",
    clim=(-20, 20),
    show_edges=False,
    show=False,
)
plotter.show()
```

### Render a 3D Volume

Use `mode` to choose how a 3-D scalar field is rendered.

| `mode` | Rendering | Main options |
| --- | --- | --- |
| `"box"` | outer surface | `opacity`, `show_edges` |
| `"volume"` | volume rendering | `opacity` |
| `"slice"` | orthogonal slices | `cmap`, `clim` |
| `"contour"` | iso-surfaces | `contour_levels`, `opacity` |

```python
data.phisp[-1].plot3d(mode="box", opacity=0.4, show=True)
data.phisp[-1].plot3d(mode="volume", opacity="sigmoid", show=True)
data.phisp[-1].plot3d(mode="slice", cmap="coolwarm", clim=(-50, 50), show=True)
data.phisp[-1].plot3d(mode="contour", contour_levels=12, show=True)
```

To save an image, create the plotter with `show=False` and use PyVista's `screenshot()`.

```python
plotter = data.phisp[-1].plot3d(mode="contour", contour_levels=10, show=False)
plotter.screenshot("phisp_contour.png")
plotter.close()
```

## Vector Fields

A three-component `VectorData` can draw streamlines with `mode="stream"` / `"streamline"` or glyph arrows with `mode="quiver"` / `"vec"`.

```python
data.j1xyz[-1].plot3d(
    mode="stream",
    backend="pyvista",
    n_points=300,
    tube_radius=0.02,
    show=True,
)

data.j1xyz[-1].plot3d(
    mode="quiver",
    backend="pyvista",
    skip=(3, 3, 2),
    factor=0.4,
    show=True,
)
```

Streamline seeds are passed to PyVista's `mesh.streamlines()`. Tune `source_center`, `source_radius`, and `n_points` when needed. Quiver arrows are down-sampled with `skip` and scaled with `factor`.

## Overlays

### Add Layers to the Same Plotter

Every PyVista API accepts `plotter=`. Add multiple layers to the same plotter to combine scalar slices, volume outlines, and vector streamlines in one 3-D scene.

```python
plotter = data.phisp[-1].plot3d(
    mode="slice",
    cmap="coolwarm",
    clim=(-50, 50),
    show=False,
)

data.j1xyz[-1].plot3d(
    mode="stream",
    backend="pyvista",
    plotter=plotter,
    tube_radius=0.02,
    color="white",
    show=True,
)
```

Boundary meshes can be added to the same plotter. `data.boundaries.plot3d()` draws the whole collection, `data.boundaries[0].plot3d()` draws one boundary, and `data.boundaries.mesh().plot3d()` draws an already built `MeshSurface3D`.

```python
plotter = data.phisp[-1].plot3d(mode="slice", show=False)
data.boundaries.plot3d(
    plotter=plotter,
    color="0.7",
    opacity=0.35,
    show_edges=True,
    show=True,
)
```

Pass boundary mesh-construction arguments via `mesh_kwargs`, and per-boundary overrides via `per`. See [Boundary Meshes](boundaries.md) for the Matplotlib / `plot_surfaces()` path.

### Overlay Backtrace / Trace Paths

Results from `data.trace.*(..., get_trace=True)` have `plot3d()` and can add trajectories to an existing plotter. When probabilities are available, `alpha="auto"` uses the per-trajectory probability as opacity.

```python
trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=(-5.0, 5.0, 16),
    vy=0.0,
    vz=(-5.0, 5.0, 16),
    get_trace=True,
)

plotter = data.phisp[-1].plot3d(mode="slice", show=False)
trace.plot3d(
    plotter=plotter,
    direction="forward",
    tube_radius=0.05,
    color="black",
    show=True,
)
```

If you build traces with `get_probabilities=False`, probability-derived alpha is unavailable. Pass an explicit value such as `alpha=0.3` when needed.

## Units and Axis Order

Grid-data axis order is `(t, z, y, x)`, and a 3-D volume is `(z, y, x)`. The PyVista helpers read that order and rearrange coordinates into PyVista's `(x, y, z)` convention.

`use_si=True` is the default. When unit metadata is available, coordinates and values are converted to SI units. If unit metadata is absent, passing `use_si=True` still renders grid / raw values internally.

`offsets=(x_offset, y_offset, z_offset)` accepts numbers or `"left"` / `"center"` / `"right"`. Use the same `use_si` and `offsets` values across all layers in an overlay.

## Remote Execution and HPC

PyVista rendering creates a plotter in the local Python process. `data.remote()` / `remote_figure()` target Matplotlib image rendering, so they do not keep a `Data3d.plot3d()` PyVista scene on the worker for repeated redraws.

On KUDPC and other login-node environments, do not run heavy PyVista rendering or large 3-D data reads directly on the login node. Route them to a compute node or a supported visualization environment. In scripts, `show=False` plus `screenshot()` is usually easier to run in batch.

```python
plotter = data.phisp[-1].plot3d(mode="slice", show=False)
plotter.screenshot("phisp_slice.png")
plotter.close()
```

## Low-Level Helpers

Usually you should use `plot3d()` / `plot_pyvista()`. Use helpers from `emout.plot.pyvista_plot` only when you want to modify the PyVista mesh yourself.

```python
from emout.plot.pyvista_plot import (
    create_plane_mesh,
    create_surface_mesh,
    create_volume_mesh,
    create_vector_mesh3d,
)

mesh, scalar_name, axis_labels, scalar_label = create_volume_mesh(data.phisp[-1])
```

Low-level helpers return PyVista objects directly; they are not high-level emout remote-rendering or article-recording APIs.

## Common Errors

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: pyvista` | old environment or dependencies not refreshed | `pip install -U emout` |
| `Data2d with time axis is not supported` | the 2-D slice still includes `t` | fix time to a single index and pass a spatial 2-D slice |
| `plot_pyvista ... requires spatial axes x,y,z` | the 3-D spatial axes are not all present | select only time, e.g. `data.phisp[-1]` |
| no streamlines appear | too few seeds or too small a seed radius | increase `n_points` / `source_radius` |
| layers are offset | different layers use different `use_si` or `offsets` | use the same values for every layer |
