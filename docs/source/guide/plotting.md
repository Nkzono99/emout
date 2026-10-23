# Plotting (`plot`)

`plot()` is the most frequently used feature in emout. It automatically selects the visualization type based on data dimensionality.

## 2D Color Map

Slicing a 3D volume to 2D produces a color-mapped plot:

```python
import emout

data = emout.Emout("output_dir")

# xz-plane (y = ny//2) at the last timestep
data.phisp[-1, :, data.inp.ny // 2, :].plot()

# xy-plane at z = 100
data.phisp[-1, 100, :, :].plot()
```

## 1D Line Plot

Slicing to 1D produces a line plot:

```python
# Profile along z-axis at x=32, y=32
data.phisp[-1, :, 32, 32].plot()
```

## Common Options

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| `use_si` | `bool` | Display axis labels and values in SI units | `True` |
| `show` | `bool` | Call `plt.show()` after plotting | `False` |
| `savefilename` | `str` | Save plot to file instead of displaying | `None` |
| `vmin` | `float` | Minimum value for colorbar | auto |
| `vmax` | `float` | Maximum value for colorbar | auto |
| `cmap` | colormap | Matplotlib colormap | custom gray-jet |
| `norm` | `str` | `'log'` for logarithmic color scale | `None` |
| `mode` | `str` | `'cm'` (colormap), `'cont'` (contour), `'cm+cont'` (both) | `'cm'` |
| `title` | `str` | Custom plot title | auto-generated |
| `xlabel` | `str` | Custom x-axis label | auto-generated |
| `ylabel` | `str` | Custom y-axis label | auto-generated |

## Examples

### Save to File

```python
data.phisp[-1, 100, :, :].plot(savefilename="phisp.png")
```

### Logarithmic Scale

```python
data.nd1p[-1, 100, :, :].plot(norm="log", vmin=1e-3, vmax=20)
```

### Contour Lines

```python
data.phisp[-1, 100, :, :].plot(mode="cont")
```

### Colormap with Contour Overlay

```python
data.phisp[-1, 100, :, :].plot(mode="cm+cont")
```

### Vector Field (Streamlines)

2D vector data is plotted as streamlines:

```python
data.j1xy[-1, 100, :, :].plot()
```

## SI Units and Raw EMSES Units

By default, `plot()` converts axis labels and values to SI units. To use raw EMSES units:

```python
data.phisp[-1, 100, :, :].plot(use_si=False)
```

## Accessing SI Values Directly

The `.val_si` property returns a NumPy array in SI units:

```python
phisp_V = data.phisp[-1].val_si       # Potential [V]
j1z_A_m2 = data.j1z[-1].val_si        # Current density [A/m^2]
nd1p_m3 = data.nd1p[-1].val_si        # Number density [/m^3]
```

## Data Masking

Mask specific regions before plotting:

```python
# Mask values below the mean
data.phisp[1].masked(lambda phi: phi < phi.mean()).plot()
```

## Sections After Spatial Averaging

Pass `keep_metadata=True` when plotting a spatial mean over a time selection.
It removes the reduced axes and preserves the remaining coordinates, units,
and the `Data` class appropriate to the result's dimensions. The default `False`
preserves legacy NumPy reduction behavior and should not be used for plotting
reduced fields. The default time-average behavior of `mean()` is unchanged.

```python
yz = data.phisp[:].mean(axis="x", keep_metadata=True)
yz[-1].plot()  # Horizontal: y; vertical: z
profile = data.phisp[:].mean(axis=("x", "y"), keep_metadata=True)
profile[-1].plot()  # z profile
```

Spatial averaging loads the selected range into memory. The same calls work
with article recording and replay; recording stores the selection before reduction.

## 3D Iso-surfaces with Matplotlib

Use `coordinates="grid"` with `Data3d.plot(mode="cont")` to place iso-surfaces
from cropped or strided data in their original grid coordinates. Each axis uses
its own start coordinate and spacing, converted to SI when `use_si=True`.
Data without unit metadata uses grid units. The default `coordinates="local"`
keeps the legacy zero origin and unit spacing (the x-axis unit length in SI).

```python
roi = data.phisp[-1, 2:8:2, 3:9:2, 4:16:2]
roi.plot(mode="cont", coordinates="grid", use_si=False, levels=[8.0], show=False)
```

`offsets=(x, y, z)` adds shifts in display units. The strings `"left"`, `"center"`,
and `"right"` move the first, middle, or last sample of each selected axis to zero.
Specify `bounds_xyz` in display coordinates after applying offsets.

## 3D Plotting with PyVista

The PyVista backend can place 2-D slices in 3-D space and render 3-D scalar / vector fields. This page only shows the entry points. For modes, overlays, saving, and HPC usage, see [PyVista Visualization](pyvista.md).

```python
# 3D scalar volume surface
data.phisp[-1, :, :, :].plot3d(mode="box", show=True)

# 2D slice placed in 3D space
data.phisp[-1, 100, :, :].plot3d(show=True)

# 3D vector field
data.j1xyz[-1].plot3d(mode="stream", show=True)
```

## Mesh Surface Rendering

When you want to overlay boundaries on a 3-D field, start with `data.boundaries.plot3d()` or `plot3d(..., surfaces=data.boundaries)`. For boundary mesh composition, per-boundary styling, and field-sampled `plot_surfaces()` rendering, see [boundary meshes](boundaries.md).

```python
data.phisp[-1].plot3d(mode="contour", levels=[0.0], filename="phisp_iso.png")
data.j1xyz[-1].plot3d(surfaces=data.boundaries, filename="j1_stream.png")
```

## Exporting VTK

Use `ordering="vtk"` to write `(z, y, x)` values in VTK's x-fastest order,
and `coordinates="grid"` to retain a crop's starting coordinates and strides.
Both options use the same convention in the PyVista and XML fallback paths.

```python
roi = data.phisp[-1, 2:10:2, 3:12:3, 4:16:2]
roi.to_vtk("phisp_roi.vti", ordering="vtk", coordinates="grid", use_si=True)
```

`use_si=True` converts values and each axis's lengths to SI when metadata is available.
The output remains cell data. Saved coordinates denote the lower cell corners;
cell centres sit half an exported spacing higher. Strides also increase the coarse
cell widths. Reversed slices reverse both coordinates and values to produce positive
spacing. This API does not change EMSES samples to point data.
The defaults, `ordering="legacy"` and `coordinates="local"`, preserve the historical
z-fastest output, zero origin, and spacing without strides. Specify both options above
for quantitative spatial comparisons.

## Hole-Surface Helpers

`consistent=True` on `plot_surface_with_hole()` and `plot_surface_with_hole_half()`
uses matching units for plotted values and colour limits, and assembles surfaces on
one 3-D Axes. When `use_si` is omitted, it uses SI if unit metadata exists, otherwise
raw values. An explicit `ax3d` is respected, as are `vmin=0` / `vmax=0`.

```python
from emout.plot.extra_plot import plot_surface_with_hole

plot_surface_with_hole(data.phisp[-1], data.inp, consistent=True)
```

The default `consistent=False` preserves historical rendering. For legacy calls,
specify `use_si` to avoid inconsistent colour limits, and also supply a shared
`ax3d` to the full-surface helper.
