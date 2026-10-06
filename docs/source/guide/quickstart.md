# Quick Start

## Installation

```bash
pip install emout
```

PyVista-based 3D visualization is included in the standard install.

> Dask-based remote execution is automatically available on Python 3.10+ (no extra install needed).

## Loading Simulation Data

```python
import emout

data = emout.Emout("output_dir")
```

`Emout` scans the directory for HDF5 files and the parameter file (`plasma.inp` or `plasma.toml`).
Variable names are resolved from the EMSES filename convention:

| Attribute | Source file pattern | Description |
| --- | --- | --- |
| `data.phisp` | `phisp00_0000.h5` | Electrostatic potential |
| `data.nd1p` | `nd1p00_0000.h5` | Species-1 number density |
| `data.j1x` | `j1x00_0000.h5` | Species-1 current density (x) |
| `data.ex` | `ex00_0000.h5` | Electric field (x) |
| `data.bz` | `bz00_0000.h5` | Magnetic field (z) |
| `data.rex` | relocated from `ex` | Relocated electric field (x) |
| `data.j1xy` | `j1x` + `j1y` | 2D vector (auto-combined) |
| `data.j1xyz` | `j1x` + `j1y` + `j1z` | 3D vector (auto-combined) |
| `data.icur` | `icur` (text) | Inward-current data (pandas DataFrame, SI conversion via `.val_si`) |
| `data.ocur` | `ocur` (text) | Outward-current data (pandas DataFrame, SI conversion via `.val_si`) |
| `data.pbody` | `pbody` (text) | Conductor-potential data (pandas DataFrame, SI conversion via `.val_si`) |

Each HDF5-backed attribute is a time-series object. Indexing by timestep returns a NumPy-compatible array:

```python
len(data.phisp)       # Number of timesteps
data.phisp[0].shape   # (nz, ny, nx)
data.phisp[-1]        # Last timestep
```

## Your First Plot

```python
# 2D color map of potential on the xz-plane (y = ny/2) at the last timestep
data.phisp[-1, :, data.inp.ny // 2, :].plot()
```

After slicing out a 2D or 1D array, call `.plot()` to visualize it with SI unit labels.

> **Note: slice axis order is `(t, z, y, x)`** — this is the reverse of the
> `(x, y, z)` convention you may be used to from NumPy. The example above
> reads as `t=-1` (last step), `z=:` (all), `y=ny/2` (fixed), `x=:` (all),
> which produces an xz-plane. Every slice expression in `emout` uses this
> order, so rewrite slices copied in from other code before using them.

## Vector Fields and Component Values

Choose a type according to the role of the data.

| Type | Role |
| --- | --- |
| `Data1d`–`Data4d` | One component of grid data: a NumPy array with coordinate and unit metadata |
| `VectorData` | Two or three field components sharing a shape and grid coordinates; supports slicing, plotting, and component-wise arithmetic |
| `ComponentValues` | Point samples, reductions, or arrays without grid metadata; retains physical component names |
| `Group` | Element-wise operations on arbitrary objects, without physical component names or grid assumptions |

`data.exz` and `data.exyz` continue to return `VectorData`.
Slice axis order is `(t, z, y, x)`. After slicing, index in the order of the
remaining axes; the two-dimensional `field` below uses `(z, x)`.

```python
import numpy as np

field = data.exz[-1, :, data.inp.ny // 2, :]
field.components["x"]       # Physical x component
field.components["z"]       # Physical z component
field.component_axes        # ("x", "z")
field.to_numpy()            # Shape: (component, z, x)

(-field).plot()
np.add(field, 1.0)          # Component-wise NumPy arithmetic

sample = field[0, 0]        # ComponentValues: one point
means = field.mean()       # ComponentValues: per-component means
means.components["z"]
means.objs                  # Existing element-wise access remains available
```

Arithmetic between named operands matches physical components regardless of
storage order. For example, adding `data.exz` and `data.ezx` adds x to x and z to z.
Operations between fields with different component sets, array shapes, or grid
coordinates raise `ValueError`. Ordinary NumPy arrays broadcast to each component,
while `Group` operands retain positional matching. Operations between `Group`
objects of different lengths also raise `ValueError`, rather than silently dropping
trailing elements. Arithmetic operates on the stored values and does not
automatically convert between different unit systems.

Boolean or integer array indexing, or a new axis inserted with `None`, on a loaded
`Data` array returns an ordinary NumPy array without coordinate metadata. For a
vector field, the component arrays are returned together as `ComponentValues`.
To mask a plot region while keeping its grid, use `.masked()` as described in the
[plotting guide](plotting.md).

`VectorData2d` and `VectorData3d` remain aliases for `VectorData`.
`.objs`, `.attrs`, `.x_data`, `.y_data`, and `.z_data` remain available.
The `*_data` aliases retain their **first, second, and third storage positions**:
for `exz`, `.y_data` is the z component. Use `.components["z"]` to select a physical
component explicitly. `.components` is a read-only mapping, but its arrays remain
accessible as before. Use `.axis(i)` to obtain the shared grid coordinates for one
current array axis.

## Appended Simulation Outputs

If the simulation continued into additional directories:

```python
# Automatic detection
data = emout.Emout("output_dir", ad="auto")

# Manual specification
data = emout.Emout("output_dir", append_directories=["output_dir_2", "output_dir_3"])
```

## Particle Data

EMSES particle outputs are automatically grouped by species:

```python
p4 = data.p4              # Species 4
p4.x, p4.y, p4.z          # Position time series
p4.vx, p4.vy, p4.vz       # Velocity time series
p4.tid                     # Trace ID

# Convert to pandas Series
data.p4.vx[0].val_si.to_series().hist(bins=200)
```
