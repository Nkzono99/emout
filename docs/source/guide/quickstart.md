Lang: [English](quickstart.md) | [日本語](quickstart.ja.md)

# Quick Start

## Installation

```bash
pip install emout
```

For 3D visualization with PyVista:

```bash
pip install "emout[pyvista]"
```

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
| `data.icur` | `icur` (text) | Current data (pandas DataFrame) |
| `data.pbody` | `pbody` (text) | Conductor data (pandas DataFrame) |

Each attribute is a time-series object. Indexing by timestep returns a NumPy-compatible array:

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

Slicing follows the axis order `(t, z, y, x)`. After slicing out a 2D or 1D array, call `.plot()` to visualize it with SI unit labels.

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
