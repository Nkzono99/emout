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

- **Documentation:** [User Guide (English/日本語)](https://nkzono99.github.io/emout/guide/quickstart.html) | [API Reference](https://nkzono99.github.io/emout/api/emout.html)
- **Notebook example:** [Visualization of lunar surface charging simulation](https://nbviewer.org/github/Nkzono99/examples/blob/main/examples/emout/example.ipynb)
- **Codex plugin:** [emout Context](https://github.com/Nkzono99/emout/blob/main/plugins/emout-context/README.en.md) — install with the standard Codex `codex plugin marketplace add` / `codex plugin add` flow, then ask Codex to create or improve visualization scripts, use unit conversion, apply `remote_figure` for large outputs, and troubleshoot emout outside this repository ([installation guide](https://github.com/Nkzono99/emout/blob/main/plugins/README.en.md))

---

## Installation

```bash
pip install emout

# For 3D visualization with PyVista
pip install "emout[pyvista]"

# Check the installed version and whether PyPI has a newer release
emout version --check-update
```

> Dask-based remote execution is automatically available on Python 3.10+ (no extra install needed).

---

## Codex Plugin

The emout Codex plugin, `emout Context`, makes emout's axis order, unit conversion, plotting, `remote_figure`, and troubleshooting context available when Codex is started in simulation output directories or other repositories.

```bash
codex plugin marketplace add Nkzono99/emout \
  --ref main \
  --sparse .agents/plugins \
  --sparse plugins/emout-context
codex plugin add emout-context@emout
```

To update, run `codex plugin marketplace upgrade emout` and then reinstall with `codex plugin add emout-context@emout`. See [Codex plugin installation](https://github.com/Nkzono99/emout/blob/main/plugins/README.en.md) for the full install and update flow.

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
| **Backtrace** | `data.backtrace.get_probabilities(...)`, `get_backtrace(...)` | [→ Backtrace](https://nkzono99.github.io/emout/guide/backtrace.html) |
| **3D (PyVista)** | `plot3d(mode="box"/"stream"/"quiver")` | [→ Quick Start](https://nkzono99.github.io/emout/guide/quickstart.html) |
| **Remote exec** | Dask Actor offloads processing to compute nodes | [→ Remote Execution](https://nkzono99.github.io/emout/guide/distributed.html) |
| **Article data** | Record and replay the minimum slices consumed by `plot()` / `to_numpy()` | [→ Article Data](https://nkzono99.github.io/emout/guide/article.html) |

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

### Recording And Replaying Article Data

Figure scripts can still start with the usual `emout.Emout()` call.
Switch article mode with environment variables to save only the minimum
slices consumed by `plot()` and `to_numpy()` under a records path, then
replay the same script from those recorded slices.

```python
import emout

data = emout.Emout("output_dir")
ymid = data.inp.ny // 2

data.phisp[-1, :, ymid, :].plot(cmap="viridis")
arr = data.ex[-1, :, ymid, :].to_numpy()
```

```bash
# Normal run
python fig1.py

# Record: saves to article-records/datasets/<output_dir>-<hash>/fig1/
EMOUT_ARTICLE_MODE=record \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
EMOUT_ARTICLE_NAME=fig1 \
python fig1.py

# Replay: restore from recorded slices instead of the original large HDF5 files
EMOUT_ARTICLE_MODE=replay \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
EMOUT_ARTICLE_NAME=fig1 \
python fig1.py
```

`EMOUT_ARTICLE_NAME` is optional. When omitted, records are saved under
`default`, so a notebook or one script can collect all figures into a
single bundle. Recreating `Emout()` with the same `article_name` appends
only slices that are not already recorded.

When a script opens multiple simulation outputs, records are separated per
source under `article-records/datasets/<source>/default/`. Replay on another
machine first matches sources by directory basename. If multiple outputs have
the same basename, pass a stable `article_source_name` in the normal script.

```python
data = [
    emout.Emout("case_a/output", article_source_name="case_a"),
    emout.Emout("case_b/output", article_source_name="case_b"),
]
```

Recorded `data.h5` files use HDF5 gzip compression. To package a whole
bundle as `.tar.gz` or `.zip` for publication, set `EMOUT_ARTICLE_ARCHIVE=1`
(`.tar.gz`) / `EMOUT_ARTICLE_ARCHIVE=zip`, or pass `article_archive=True` /
`article_archive="zip"`. Replay automatically extracts the matching archive
when the extracted directory is not present.

The same settings can be passed as arguments.

```python
data = emout.Emout(
    "output_dir",
    article_mode="record",
    article_records_path="article-records",
    article_name="fig1",
    article_archive="zip",
)
```

Replay mode raises an exception when a script asks for an unrecorded slice.
This makes it clear whether the public data bundle contains everything
needed to reproduce a figure.
`plasma.inp`, `plasma.toml`, and small diagnostic files (`icur`, `pbody`)
are saved as well, so visualizations that depend on input parameters and
boundary meshes also replay, including `data.inp`, `data.toml`,
`data.boundaries.plot()`, and `data.phisp[-1].plot_surfaces(data.boundaries)`.

### Remote Execution (Dask) — Experimental

Offload data processing to HPC compute nodes; only plot images are returned to your login node.
For new code, prefer the explicit `Emout.remote()` workflow.
The older “auto-remote when a server is running” behavior is still kept as a backward-compatible compatibility mode.

```bash
# Start the server once in a terminal
emout server start --partition gr20001a --memory 60G
```

`emout server` configures TLS authentication automatically and, by
default, allows one active server per user. If you intentionally need an
additional session, use
`emout server start --allow-multiple --name <session>`.

To prevent accidental local materialization of field arrays on a login
node, use `emout.disable_local_data_access()` or set
`EMOUT_LOCAL_DATA_POLICY=remote_required`. Pass `Emout(...,
local_data_policy="allow")` when a small dataset may read fields locally.

```python
import matplotlib.pyplot as plt
import emout
from emout.distributed import remote_figure, remote_scope

data = emout.Emout("output_dir").remote()

# Recommended: keep remote objects explicitly
with remote_scope():
    ymid = int(data.inp.ny // 2)

    with remote_figure():
        plt.figure(figsize=(18, 16))
        data.phisp[-1, 180:400, ymid, :].plot()
        (-data.exz[-1, 180:400, ymid, :]).plot()
        plt.title("remote expression example")

# Compatibility mode: existing plot() code still works unchanged
local_data = emout.Emout("output_dir")
local_data.phisp[-1, :, 100, :].plot()    # only 2D slice transferred
plt.xlabel("x [m]")                       # local matplotlib annotation

# Run everything on the server (only PNG comes back)
with remote_figure():
    local_data.phisp[-1, :, 100, :].plot()
    plt.axhline(y=50, color="red")
    plt.title("Custom title")

# Save directly to a file in CLI / batch jobs
with remote_figure(savefilepath="figures/phisp.png"):
    local_data.phisp[-1, :, 100, :].plot()
    plt.title("saved remotely")

# The output format is inferred from the extension
with remote_figure(savefilepath="figures/phisp.svg"):
    local_data.phisp[-1, :, 100, :].plot()

# open/close style — easy to retrofit existing code
from emout.distributed import RemoteFigure

rf = RemoteFigure()
rf.open()
local_data.phisp[-1, :, 100, :].plot()
rf.close()

# Jupyter cell magic — just add to the top of a cell
# %load_ext emout.distributed.remote_figure
# %%remote_figure
# local_data.phisp[-1, :, 100, :].plot()
```

When `savefilepath` is provided, the rendered bytes are written to disk.
PNG/JPEG output is still shown inline in IPython, while CLI / batch
usage skips local display.

Heavy backtrace computations run on the server and stay in worker memory;
re-render with different parameters without recomputation.
Both `data.backtrace.get_probabilities(...)` and
`data.remote().backtrace.get_probabilities(...)` return dedicated
backtrace proxies — see the
[backtrace guide](https://nkzono99.github.io/emout/guide/backtrace.html)
for the API itself.

**Cross-simulation comparison** is also supported:

```python
data_a = emout.Emout("/sim_a")
data_b = emout.Emout("/sim_b")
result_a = data_a.backtrace.get_probabilities(...)
result_b = data_b.backtrace.get_probabilities(...)

with remote_figure(figsize=(12, 5)):
    plt.subplot(1, 2, 1)
    result_a.vxvz.plot()
    plt.subplot(1, 2, 2)
    result_b.vxvz.plot()
```

→ [Remote Execution Guide](https://nkzono99.github.io/emout/guide/distributed.html)

<details>
<summary>Experimental features (Poisson solver / backtrace)</summary>

```python
# Poisson solver
from emout.utils import poisson
phi = poisson(rho, dx=dx, btypes=btypes, epsilon_0=cn.epsilon_0)

# Backtrace (requires vdist-solver-fortran)
result = data.backtrace.get_probabilities(x, y, z, vx, vy, vz, ispec=0)
result.vxvz.plot()
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
