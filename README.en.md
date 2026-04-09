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
- an EMSES ⇄ SI unit conversion system driven by the `!!key dx=...,to_c=...` header (30+ quantities)
- a Python-native interface to EMSES `finbound` boundary geometry

---

- **Documentation:** [User Guide (English)](https://nkzono99.github.io/emout/guide/quickstart.html) | [API Reference](https://nkzono99.github.io/emout/api/emout.html) | [ユーザーガイド（日本語）](https://nkzono99.github.io/emout/guide/quickstart.ja.html)
- **Notebook example:** [Visualization of lunar surface charging simulation](https://nbviewer.org/github/Nkzono99/examples/blob/main/examples/emout/example.ipynb)

---

## Installation

```bash
pip install emout

# For 3D visualization with PyVista
pip install "emout[pyvista]"
```

---

## Quick Start

```python
import emout

data = emout.Emout("output_dir")

# Plot potential on the xz-plane (y=ny/2) at the last timestep — SI units auto-applied
data.phisp[-1, :, data.inp.ny // 2, :].plot()
```

Variable names are resolved automatically from EMSES filenames:

```python
data.phisp          # Potential (GridDataSeries — time series)
data.nd1p           # Species-1 number density
data.j1x            # Species-1 current density (x)
data.j1xy           # j1x + j1y auto-combined → 2D vector
data.j1xyz          # 3D vector
data.icur, data.pbody  # Text outputs (pandas DataFrame)
```

Axis order is `(t, z, y, x)`.

---

## Feature Guide

See the user guide for detailed usage of each feature.

| Feature | What it does | Guide |
| --- | --- | --- |
| **Plotting** | `plot()` / `cmap()` / `contour()` for 1D/2D plots | [→ Plotting](https://nkzono99.github.io/emout/guide/plotting.html) |
| **Animations** | `gifplot()` for GIF/HTML, multi-panel layouts | [→ Animations](https://nkzono99.github.io/emout/guide/animation.html) |
| **Parameters** | `data.inp.nx`, `data.toml.species[0].wp` | [→ Parameters](https://nkzono99.github.io/emout/guide/inp.html) |
| **Unit conversion** | `data.unit.v.reverse(1.0)`, `data.phisp[-1].val_si` | [→ Units](https://nkzono99.github.io/emout/guide/units.html) |
| **Boundary meshes** | `data.boundaries.mesh()`, overlay on `plot_surfaces` | [→ Boundaries](https://nkzono99.github.io/emout/guide/boundaries.html) |
| **3D (PyVista)** | `plot3d(mode="box"/"stream"/"quiver")` | [→ Quick Start](https://nkzono99.github.io/emout/guide/quickstart.html) |

---

## Recipes

### Plotting

```python
data.phisp[-1, 100, :, :].plot()                       # 2D color map
data.phisp[-1, 100, :, :].contour()                     # contour lines
data.nd1p[-1, 100, :, :].plot(norm="log", vmin=1e-3)    # log scale
data.j1xy[-1, 100, :, :].plot()                          # streamlines
data.phisp[-1, :, 32, 32].plot()                         # 1D profile
```

### Animations

```python
data.phisp[:, 100, :, :].gifplot()                                  # Jupyter inline
data.phisp[:, 100, :, :].gifplot(action="save", filename="out.gif") # save as GIF
```

### Unit conversion

```python
data.unit.v.trans(1.0)       # SI → EMSES
data.phisp[-1].val_si        # full 3D array in SI [V]
```

### Boundary meshes

```python
data.boundaries[0].mesh()                   # single boundary → MeshSurface3D
data.phisp[-1].plot_surfaces(               # overlay on field
    ax=ax, surfaces=data.boundaries,
)
```

### Particle data

```python
p4 = data.p4                               # species 4
p4.vx[0].val_si.to_series().hist(bins=200)  # velocity distribution
```

<details>
<summary>Appended outputs / input–output path separation</summary>

```python
# Combine appended simulation outputs
data = emout.Emout("output_dir", ad="auto")

# Separate input file and output directory
data = emout.Emout(input_path="/path/to/plasma.toml", output_directory="output_dir")
```

</details>

<details>
<summary>Experimental features (Poisson solver / backtrace / Dask cluster)</summary>

```python
# Poisson solver
from emout.utils import poisson
phi = poisson(rho, dx=dx, btypes=btypes, epsilon_0=cn.epsilon_0)

# Backtrace (requires vdist-solver-fortran)
result = data.backtrace.get_probabilities(ix, iy, iz, vx_range, vy_center, vz_range, ispec=0)
result.vxvz.plot()

# Dask cluster integration
from emout.distributed import start_cluster, stop_cluster
client = start_cluster(partition="gr20001a", processes=1, cores=112, memory="60G", walltime="03:00:00")
```

</details>

---

## Contributing

Bug reports, feature requests, and pull requests are welcome.

- **Bugs / questions:** open a [GitHub Issue](https://github.com/Nkzono99/emout/issues) with a minimal reproduction
- **Pull requests:** branch off `main`, keep `pytest -q` green, and submit
- **Docs:** `README.md` (Japanese) and `README.en.md` (English) are kept in sync — please update both

The development environment and repo layout are covered in [AGENTS.md](AGENTS.md).

---

## License

[MIT License](LICENSE)

## Links

- [User Guide (English)](https://nkzono99.github.io/emout/guide/quickstart.html) | [ユーザーガイド（日本語）](https://nkzono99.github.io/emout/guide/quickstart.ja.html)
- [API Reference](https://nkzono99.github.io/emout/api/emout.html)
- [EMSES (MPIEMSES3D)](https://github.com/Nkzono99/MPIEMSES3D)
- [Example Notebook](https://nbviewer.org/github/Nkzono99/examples/blob/main/examples/emout/example.ipynb)
