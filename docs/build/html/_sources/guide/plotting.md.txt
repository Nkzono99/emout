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

## 3D Plotting with PyVista

For 3D visualization, install the optional PyVista dependency:

```bash
pip install "emout[pyvista]"
```

```python
# 3D volume rendering
data.phisp[-1, :, :, :].plot3d(mode="box", show=True)

# 2D slice placed in 3D space
data.phisp[-1, 100, :, :].plot3d(show=True)

# 3D vector field
data.j1xyz[-1].plot3d(mode="stream", show=True)
data.j1xyz[-1].plot3d(mode="quiver", show=True)
```

### Mesh Surface Rendering

For face-oriented 3D rendering with explicit mesh surfaces:

```python
import matplotlib.pyplot as plt
from emout.plot.surface_cut import (
    BoxMeshSurface,
    CylinderMeshSurface,
    HollowCylinderMeshSurface,
    RenderItem,
    plot_surfaces,
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

plot_surfaces(
    ax,
    field=field3d,  # surface_cut.Field3D
    surfaces=[
        RenderItem(
            BoxMeshSurface(0, 10, 0, 6, 0, 4, faces=("zmax", "xmax")),
            style="field",
        ),
        RenderItem(
            CylinderMeshSurface(
                center=(5, 3, 2), axis="z", radius=1.5, length=4.0,
                parts=("side", "top"),
            ),
            style="solid",
            solid_color="0.7",
            alpha=0.5,
        ),
    ],
)
```
