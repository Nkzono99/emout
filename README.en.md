Lang: [English](README.en.md) | [日本語](README.md)

# emout

[![PyPI version](https://img.shields.io/pypi/v/emout.svg)](https://pypi.org/project/emout/)
[![Python](https://img.shields.io/pypi/pyversions/emout.svg)](https://pypi.org/project/emout/)
[![Docs](https://github.com/Nkzono99/emout/actions/workflows/docs.yaml/badge.svg)](https://nkzono99.github.io/emout/)
[![CodeQL](https://github.com/Nkzono99/emout/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/Nkzono99/emout/actions/workflows/codeql-analysis.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Python library for analyzing and visualizing [EMSES](https://github.com/Nkzono99/MPIEMSES3D) simulation outputs**

emout is:

- a one-line facade for loading `.h5` grid output + `plasma.inp` / `plasma.toml`
- 1D / 2D / 3D plotting that auto-selects the right view from data dimensionality
- an EMSES ⇄ SI unit conversion system driven by the `!!key dx=...,to_c=...` header
- a Python-native interface to EMSES `finbound` boundary geometry

---

- **Documentation (English):** [User Guide](https://nkzono99.github.io/emout/guide/quickstart.html) | [API Reference](https://nkzono99.github.io/emout/api/emout.html)
- **Notebook example:** [Visualization of lunar surface charging simulation](https://nbviewer.org/github/Nkzono99/examples/blob/main/examples/emout/example.ipynb)

emout reads EMSES output files (`.h5`) and parameter files (`plasma.inp` / `plasma.toml`),
letting you browse, plot, animate, and convert data to SI units in just a few lines of code.

---

## Highlights

- **Plot in one line** — `data.phisp[-1, :, 100, :].plot()` produces a 2D colormap with SI unit labels
- **Time-series animations** — `gifplot()` writes GIF / HTML in a single call, `build_frame_updater()` composes multi-panel layouts
- **Automatic SI conversion** — parses `!!key dx=...,to_c=...` from `plasma.inp` and builds 30+ unit translators on the fly
- **Dynamic variable resolution** — access EMSES outputs by filename: `data.phisp`, `data.nd1p`, `data.j1xy`, `data.j1xyz`
- **3D visualization** — PyVista integration, plus auto-generated finbound boundary meshes via `data.boundaries`
- **Parameter file flexibility** — reads both legacy `plasma.inp` and modern `plasma.toml`; nested TOML structure stays accessible via `data.toml`

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Loading Data](#loading-data)
4. [Plotting (`plot`)](#plotting-plot)
5. [Animations (`gifplot`)](#animations-gifplot)
6. [Parameter File (`data.inp`)](#parameter-file-datainp)
7. [Unit Conversion (`data.unit`)](#unit-conversion-dataunit)
8. [Particle Data](#particle-data)
9. [Data Masking](#data-masking)
10. [Appended Outputs](#appended-outputs)
11. [3D Plotting (PyVista)](#3d-plotting-pyvista)
12. [Boundary Meshes (`data.boundaries`)](#boundary-meshes-databoundaries)
13. [Solving Poisson's Equation (Experimental)](#solving-poissons-equation-experimental)
14. [Backtrace (Experimental)](#backtrace-experimental)
15. [Contributing](#contributing)

---

## Installation

```bash
pip install emout
```

For 3D visualization with PyVista:

```bash
pip install "emout[pyvista]"
```

---

## Quick Start

```python
import emout

data = emout.Emout("output_dir")

# Plot the potential on the xz-plane (y = ny/2) at the last timestep
data.phisp[-1, :, data.inp.ny // 2, :].plot()
```

This single call displays a color-mapped 2D plot with SI unit labels.

---

## Loading Data

Assumes the following directory structure:

```
output_dir/
├── plasma.inp          # Parameter file
├── phisp00_0000.h5     # Potential
├── nd1p00_0000.h5      # Species-1 number density
├── nd2p00_0000.h5      # Species-2 number density
├── j1x00_0000.h5       # Species-1 current density (x)
├── ex00_0000.h5        # Electric field (x)
├── bz00_0000.h5        # Magnetic field (z)
└── ...
```

```python
import emout

data = emout.Emout("output_dir")

# Separate input file and output directory
data = emout.Emout(input_path="/path/to/plasma.toml", output_directory="output_dir")

# Variable names are resolved automatically from EMSES filenames
data.phisp          # Potential (time series)
len(data.phisp)     # Number of timesteps
data.phisp[0].shape # (nz, ny, nx)

data.nd1p           # Species-1 number density
data.j1x            # Species-1 current density (x)
data.bz             # Magnetic field (z)

# Vector data (auto-combined)
data.j1xy           # j1x + j1y → 2D vector
data.j1xyz          # j1x + j1y + j1z → 3D vector

# Relocated data
data.rex            # Relocated electric field (x)

# Text outputs (pandas DataFrame)
data.icur           # Current data
data.pbody          # Conductor data
```

---

## Plotting (`plot`)

`plot()` automatically selects the appropriate visualization based on data dimensionality.
**This is the most frequently used feature.**

### 2D Color Map

```python
# xz-plane at y = ny//2, last timestep
data.phisp[-1, :, data.inp.ny // 2, :].plot()

# xy-plane at z = 100
data.phisp[-1, 100, :, :].plot()
```

### 1D Line Plot

```python
# Profile along z-axis at x=32, y=32
data.phisp[-1, :, 32, 32].plot()
```

### Common Options

| Parameter | Description | Default |
| --- | --- | --- |
| `use_si` | Display axis labels and values in SI units | `True` |
| `show` | Call `plt.show()` | `False` |
| `savefilename` | Save as image file | `None` |
| `vmin`, `vmax` | Colorbar range | auto |
| `cmap` | Colormap | custom gray-jet |
| `norm` | `'log'` for logarithmic scale | `None` |
| `mode` | `'cm'`, `'cont'`, `'cm+cont'` (2D) | `'cm'` |

```python
# Save with SI units
data.phisp[-1, 100, :, :].plot(savefilename="phisp.png")

# Logarithmic scale
data.nd1p[-1, 100, :, :].plot(norm="log", vmin=1e-3, vmax=20)

# Contour plot
data.phisp[-1, 100, :, :].plot(mode="cont")

# Vector field (streamlines)
data.j1xy[-1, 100, :, :].plot()
```

#### 2D shortcut methods

On 2D data you can skip the stringly-typed `mode=` argument and call a named
method that makes the intent obvious when you re-read the script later. Both
shortcuts accept every keyword `plot()` does — except `mode`, which is fixed.

```python
# equivalent to mode='cm'
data.phisp[-1, 100, :, :].cmap()

# equivalent to mode='cont'
data.phisp[-1, 100, :, :].contour()
```

For an overlay (`plot(mode='cm+cont')`) or for 1D / 3D data, use `plot()` directly.

---

## Animations (`gifplot`)

Create GIF / HTML animations from time-series data. **The second most used feature.**

### Basic Usage

```python
# Inline display in Jupyter Notebook (default)
data.phisp[:, 100, :, :].gifplot()

# Save as GIF
data.phisp[:, 100, :, :].gifplot(action="save", filename="phisp.gif")

# Display in matplotlib window
data.phisp[:, 100, :, :].gifplot(action="show")
```

### Common Options

| Parameter | Description | Default |
| --- | --- | --- |
| `action` | `'to_html'`, `'save'`, `'show'`, `'return'`, `'frames'` | `'to_html'` |
| `filename` | Save path when `action='save'` | `None` |
| `axis` | Animation axis | `0` |
| `interval` | Frame interval [ms] | `200` |
| `use_si` | Use SI units | `True` |
| `vmin`, `vmax` | Colorbar range | auto |
| `norm` | `'log'` for logarithmic scale | `None` |

### Multi-Panel Animations

```python
# Create frame updaters
updater0 = data.phisp[:, 100, :, :].gifplot(action="frames", mode="cmap")
updater1 = data.phisp[:, 100, :, :].build_frame_updater(mode="cont")
updater2 = data.nd1p[:, 100, :, :].build_frame_updater(mode="cmap", vmin=1e-3, vmax=20, norm="log")
updater3 = data.nd2p[:, 100, :, :].build_frame_updater(mode="cmap", vmin=1e-3, vmax=20, norm="log")
updater4 = data.j2xy[:, 100, :, :].build_frame_updater(mode="stream")

# Define layout (triple-nested list: [row][col][overlay])
layout = [
    [
        [updater0, updater1],
        [updater2],
        [updater3, updater4],
    ]
]

animator = updater0.to_animator(layout=layout)
animator.plot(action="to_html")  # or "save", "show"
```

---

## Parameter File (`data.inp`)

Reads `plasma.inp` (or `plasma.toml`) as a dictionary-like object.

```python
# Access by group name + parameter name
data.inp["tmgrid"]["nx"]    # → e.g., 256
data.inp["plasma"]["wp"]    # → e.g., [1.0, 0.05]

# Group name can be omitted if unambiguous
data.inp["nx"]

# Attribute access
data.inp.tmgrid.nx
data.inp.nx
```

### Commonly Used Parameters

```python
# Grid size
nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz

# Time step
dt = data.inp.dt
ifdiag = data.inp.ifdiag  # Output interval

# Number of species
nspec = data.inp.nspec

# Boundary conditions
data.inp.mtd_vbnd  # Per-axis boundary type (0=periodic, 1=Dirichlet, 2=Neumann)
```

### TOML Format (`plasma.toml`)

When `plasma.toml` exists in the directory, it is automatically converted to `plasma.inp` via the `toml2inp` command before loading:

```python
data = emout.Emout("output_dir")  # Runs toml2inp if plasma.toml exists
data.inp.nx  # Same interface
data.toml    # Direct access to structured TOML data (TomlData)
data.toml.species[0].wp  # Access nested structures
```

> **Note:** The `toml2inp` command must be available on PATH (bundled with [MPIEMSES3D](https://github.com/Nkzono99/MPIEMSES3D)).
> Group tables such as `species_groups` are expanded into each entry on load and omitted from `data.toml`.

---

## Unit Conversion (`data.unit`)

Converts between EMSES internal units and SI units.

> **Prerequisite:** The first line of `plasma.inp` must contain `!!key dx=[0.5],to_c=[10000.0]`,
> where `dx` is the grid spacing [m] and `to_c` is the internal speed of light in EMSES units.

### Using Unit Translators

```python
# SI → EMSES
data.unit.v.trans(1.0)      # 1 m/s → EMSES velocity unit

# EMSES → SI
data.unit.v.reverse(1.0)    # 1 EMSES velocity unit → m/s
```

### Direct SI Values

```python
# .val_si property converts data to SI units
phisp_V = data.phisp[-1].val_si         # Potential [V]
j1z_A_m2 = data.j1z[-1].val_si          # Current density [A/m^2]
nd1p_m3 = data.nd1p[-1].val_si          # Number density [/m^3]
```

### Available Units

<details>
<summary>Click to expand</summary>

| Name | Quantity | SI Unit |
| --- | --- | --- |
| `phi` | Potential | V |
| `E` | Electric field | V/m |
| `B` | Magnetic flux density | T |
| `J` | Current density | A/m^2 |
| `n` | Number density | /m^3 |
| `rho` | Charge density | C/m^3 |
| `v` | Velocity | m/s |
| `t` | Time | s |
| `f` | Frequency | Hz |
| `length` | Length | m |
| `q` | Charge | C |
| `m` | Mass | kg |
| `W` | Energy | J |
| `w` | Energy density | J/m^3 |
| `P` | Power | W |
| `T` | Temperature | K |
| `F` | Force | N |
| `a` | Acceleration | m/s^2 |
| `i` | Current | A |
| `N` | Flux | /m^2s |
| `c` | Speed of light | m/s |
| `eps` | Permittivity | F/m |
| `mu` | Permeability | H/m |
| `C` | Capacitance | F |
| `L` | Inductance | H |
| `G` | Conductance | S |
| `q_m` | Charge-to-mass ratio | C/kg |
| `qe` | Elementary charge | C |
| `qe_me` | Electron charge-to-mass ratio | C/kg |
| `kB` | Boltzmann constant | J/K |
| `e0` | Vacuum permittivity | F/m |
| `m0` | Vacuum permeability | N/A^2 |

</details>

---

## Particle Data

EMSES particle outputs (`p4xe00_0000.h5`, `p4vxe00_0000.h5`, etc.) are automatically grouped.

```python
# Species-4 particle data
p4 = data.p4

# Component time series
p4.x, p4.y, p4.z         # Position
p4.vx, p4.vy, p4.vz      # Velocity
p4.tid                     # Trace ID

# Convert to pandas Series (useful for histograms, etc.)
data.p4.vx[0].val_si.to_series().hist(bins=200)
```

---

## Data Masking

<details>
<summary>Show examples</summary>

```python
# Mask values below the mean
data.phisp[1].masked(lambda phi: phi < phi.mean())

# Manual equivalent
phi = data.phisp[1].copy()
phi[phi < phi.mean()] = float("nan")
```

</details>

---

## Separating Input and Output Directories

When the input parameter file and output files are in different locations:

```python
# Specify input file path explicitly
data = emout.Emout(input_path="/path/to/plasma.toml", output_directory="output_dir")

# Specify only output directory (input file is searched within it)
data = emout.Emout("output_dir")

# Traditional usage (everything in the same directory)
data = emout.Emout("output_dir")
```

| Parameter | Description | Default |
| --- | --- | --- |
| `directory` | Base directory | `"./"` |
| `input_path` | Full path to the input file (e.g., `/path/to/plasma.toml`) | `None` |
| `output_directory` | Directory containing output files | Same as `directory` |

---

## Appended Outputs

<details>
<summary>Show examples</summary>

When a simulation continues into additional output directories:

```python
# Manual specification
data = emout.Emout("output_dir", append_directories=["output_dir_2", "output_dir_3"])

# Automatic detection
data = emout.Emout("output_dir", ad="auto")
```

</details>

---

## 3D Plotting (PyVista)

<details>
<summary>Show examples</summary>

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

```python
import matplotlib.pyplot as plt
from emout.plot.surface_cut import (
    BoxMeshSurface, CylinderMeshSurface, HollowCylinderMeshSurface,
    RenderItem, plot_surfaces,
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

plot_surfaces(
    ax,
    field=field3d,
    surfaces=[
        RenderItem(BoxMeshSurface(0, 10, 0, 6, 0, 4, faces=("zmax", "xmax")), style="field"),
        RenderItem(
            CylinderMeshSurface(center=(5, 3, 2), axis="z", radius=1.5, length=4.0, parts=("side", "top")),
            style="solid", solid_color="0.7", alpha=0.5,
        ),
    ],
)
```

</details>

---

## Boundary Meshes (`data.boundaries`)

The `finbound` / legacy boundaries defined under MPIEMSES's `&ptcond` are exposed
as Python objects. Each boundary produces a `MeshSurface3D` that you can overlay
onto a 3D field plot or style individually.

### Access

```python
# The whole collection
data.boundaries                 # BoundaryCollection (iterable, indexable)
len(data.boundaries)
data.boundaries.types           # list of boundary_types strings
data.boundaries.skipped         # [(index, type_name, reason), ...]

# An individual boundary
data.boundaries[0]              # subclass (SphereBoundary, CylinderBoundary, ...)
data.boundaries[0].mesh()       # → MeshSurface3D (SI units by default)
data.boundaries[0].mesh(use_si=False)  # keep grid units

# All boundaries fused into one composite mesh
data.boundaries.mesh()          # → CompositeMeshSurface
```

### Overlay on a 3D field plot

Pass `data.boundaries` straight to `Data3d.plot_surfaces` to draw the
geometries on top of isosurfaces or slices.

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

data.phisp[-1].plot_surfaces(
    ax=ax,
    surfaces=data.boundaries,         # auto-wrapped into RenderItems
)
plt.show()
```

### Composition and per-boundary styling

```python
# Boundary + Boundary → BoundaryCollection
combined = data.boundaries[0] + data.boundaries[1]

# Different style per boundary
data.phisp[-1].plot_surfaces(
    ax=ax,
    surfaces=data.boundaries.render(
        per={
            0: dict(style="solid", solid_color="0.7"),
            1: dict(alpha=0.5),
        },
    ),
)
```

### Supported boundary types

Both `boundary_type = "complex"` with `boundary_types(i)` and the legacy
single-body modes (`boundary_type = "rectangle-hole"` etc.) are supported.

| Category | Type name |
| --- | --- |
| Closed solids | `sphere`, `cuboid` |
| Cylinders | `cylinderx`, `cylindery`, `cylinderz`, `open-cylinderx/y/z` |
| Flat panels | `rectangle`, `circlex/y/z`, `diskx/y/z`, `plane-with-circlex/y/z` |
| Legacy single-body | `flat-surface`, `rectangle-hole`, `cylinder-hole` |

See [MPIEMSES3D Parameters.md](https://github.com/Nkzono99/MPIEMSES3D/blob/main/docs/Parameters.md) for the full parameter specification. Unregistered types are recorded in `data.boundaries.skipped` as `(index, type_name, reason)` tuples rather than raising.

---

## Solving Poisson's Equation (Experimental)

<details>
<summary>Show examples</summary>

```python
import numpy as np
import scipy.constants as cn
from emout import Emout
from emout.utils import poisson

data = Emout("output_dir")
dx = data.inp.dx
rho = data.rho[-1].val_si
btypes = ["pdn"[i] for i in data.inp.mtd_vbnd]

phisp = poisson(rho, dx=dx, btypes=btypes, epsilon_0=cn.epsilon_0)
```

</details>

---

## Backtrace (Experimental)

<details>
<summary>Show examples</summary>

```bash
pip install git+https://github.com/Nkzono99/vdist-solver-fortran.git
```

```python
# Compute probability distribution
probability_result = data.backtrace.get_probabilities(
    128, 128, 60,
    (-data.inp.path[0] * 3, data.inp.path[0] * 3, 10),
    0,
    (-data.inp.path[0] * 3, 0, 10),
    ispec=0,
)
probability_result.vxvz.plot()

# Compute and plot backtrace trajectories
particles = probability_result.particles
prob_1d = probability_result.probabilities.ravel()
alpha_values = np.nan_to_num(prob_1d / prob_1d.max())

backtrace_result = data.backtrace.get_backtraces_from_particles(particles, ispec=0)
backtrace_result.xz.plot(color="black", alpha=alpha_values)
```

### Dask Cluster Integration

```python
from emout.distributed import start_cluster, stop_cluster

client = start_cluster(
    partition="gr20001a",
    processes=1, cores=112, memory="60G",
    walltime="03:00:00",
    scheduler_port=32332,
)

# Subsequent data.backtrace API calls run on compute nodes
result = data.backtrace.get_probabilities(...)
stop_cluster()
```

</details>

---

## Contributing

Bug reports, feature requests, and pull requests are welcome.

- **Bugs / questions:** open a [GitHub Issue](https://github.com/Nkzono99/emout/issues) with a minimal reproduction
- **Pull requests:** branch off `main`, keep `pytest -q` green (current baseline in `AGENTS.md §10`), and submit
- **Docs:** `README.md` (Japanese) and `README.en.md` (English) are kept in sync by hand — please update both when touching user-facing content

The development environment and repo layout are covered in [AGENTS.md](AGENTS.md).

---

## License

[MIT License](LICENSE)

## Links

- [User Guide (English)](https://nkzono99.github.io/emout/guide/quickstart.html)
- [API Reference](https://nkzono99.github.io/emout/api/emout.html)
- [EMSES (MPIEMSES3D)](https://github.com/Nkzono99/MPIEMSES3D)
- [Example Notebook](https://nbviewer.org/github/Nkzono99/examples/blob/main/examples/emout/example.ipynb)
