# emout — AI Agent User Guide

This document is a compact, structured reference for AI agents that generate code using the `emout` library.
Import this file via `@docs/agent-user-guide.md` to get the full API context.

## What emout does

`emout` reads EMSES PIC simulation outputs (HDF5 grid data, particle data, `plasma.inp`/`plasma.toml` parameter files) and provides:

- One-line 1D/2D/3D plotting with automatic SI unit conversion
- GIF/HTML animation creation from time-series data
- Parameter file access as a dictionary-like object
- Bidirectional EMSES↔SI unit conversion (30+ quantities)
- Particle data grouping, phase-space plots, and pandas integration
- Boundary mesh rendering and 3D field overlay
- VTK export for external visualization
- Remote execution via Dask (offload to HPC compute nodes)

## Installation

```bash
pip install emout                  # Core (requires Python >=3.9; Dask included on 3.10+)
pip install "emout[pyvista]"       # + 3D visualization
```

---

## Core API

### Loading data

```python
import emout

data = emout.Emout("output_dir")
# data = emout.Emout("output_dir", ad="auto")                          # auto-detect appended dirs
# data = emout.Emout("output_dir", append_directories=["dir2", "dir3"]) # manual append

# Separate input file and output directory
# data = emout.Emout(input_path="/path/to/plasma.toml", output_directory="output_dir")
```

**Constructor parameters:**

| Parameter | Description | Default |
| --- | --- | --- |
| `directory` | Base directory (input + output when others unset) | `"./"` |
| `input_path` | Full path to input file (e.g. `/path/to/plasma.toml`). Overrides `directory`/`inpfilename` for input. | `None` |
| `output_directory` | Directory for output files (h5, icur, pbody). Defaults to `directory`. | `None` |
| `append_directories` / `ad` | Additional output directories or `"auto"` | `None` |
| `inpfilename` | Input filename (ignored when `input_path` is set) | `"plasma.inp"` |

### Axis order

All grid data follows the axis order `(t, z, y, x)`. A 3D volume at one timestep is `(z, y, x)`.

### Accessing variables

Variables are resolved dynamically from EMSES filenames via `data.<name>`:

| Pattern | Example | Returns |
| --- | --- | --- |
| Scalar field | `data.phisp` | `GridDataSeries` (time series) |
| Species density | `data.nd1p`, `data.nd2p` | `GridDataSeries` |
| Current density | `data.j1x`, `data.j1y` | `GridDataSeries` |
| Electric/Magnetic | `data.ex`, `data.bz` | `GridDataSeries` |
| Relocated field | `data.rex`, `data.rby` | `GridDataSeries` |
| 2D vector | `data.j1xy` | `VectorData2d` (auto-combines x+y) |
| 3D vector | `data.j1xyz` | `VectorData3d` (auto-combines x+y+z) |
| Text output | `data.icur`, `data.pbody` | `pandas.DataFrame` |
| Particle species | `data.p4` | `ParticlesSeries` (species 4) |

Indexing a time series by timestep returns a NumPy-subclass array (`Data1d`, `Data2d`, `Data3d`):

```python
len(data.phisp)        # number of timesteps
data.phisp[0].shape    # (nz, ny, nx)
data.phisp[-1]         # last timestep → Data3d
```

### Slicing

```python
data.phisp[-1, :, ny//2, :]   # → Data2d (xz-plane)
data.phisp[-1, 100, :, :]     # → Data2d (xy-plane at z=100)
data.phisp[-1, :, 32, 32]     # → Data1d (z-profile at x=32, y=32)
data.phisp[:, 100, :, :]      # → all timesteps at z=100 (for gifplot)
```

### Data manipulation (chainable)

These methods return transformed copies — the original is not modified.

```python
data.phisp[-1, :, ny//2, :].negate()      # flip sign: -data
data.phisp[-1].scale(1e3)                  # multiply by factor
data.phisp[-1, :, ny//2, :].flip('z')     # reverse along axis ('x'/'y'/'z'/'t' or int)
data.phisp[-1, :, ny//2, :].mirror('z')   # append reflected copy (len → 2n-1)
data.phisp[-1, :, ny//2, :].tile(1, 'z')  # tile periodic copies along axis

# Chaining
data.ex[-1].negate().mirror('z').tile(1, 'z').plot()
```

### Data masking

```python
data.phisp[-1].masked(lambda phi: phi < phi.mean())          # returns copy with NaN
data.phisp[-1].masked(lambda phi: phi < phi.mean()).plot()    # plot masked data
```

---

## plot()

Call `.plot()` on any sliced data. Visualization is auto-selected by dimensionality.

### Data2d.plot()

```python
data.phisp[-1, :, ny//2, :].plot()
```

**Signature:**

```python
.plot(
    axes="auto",           # "xy", "zx", etc. or "auto"
    show=False,            # call plt.show()
    use_si=True,           # SI unit labels/values
    offsets=None,          # (x_off, y_off, z_off): "left"|"center"|"right"|float
    mode="cm",             # "cm" (colormap), "cont" (contour), "cm+cont", "surf" (3D surface)
    # kwargs passed to underlying plot function:
    savefilename=None,     # save to file
    cmap=None,             # matplotlib colormap
    vmin=None, vmax=None,  # colorbar range
    figsize=None,          # (width, height)
    xlabel=None, ylabel=None, title=None,
    norm=None,             # "log" for log scale
    interpolation="bilinear",
    dpi=10,
    colorbar_label="",
)
# Returns: AxesImage (or None if show/save)
```

### Data1d.plot()

```python
data.phisp[-1, :, 32, 32].plot()
```

**Signature:**

```python
.plot(
    show=False,
    use_si=True,
    offsets=None,          # (x_off, y_off)
    savefilename=None,
    vmin=None, vmax=None,
    figsize=None,
    xlabel=None, ylabel=None, label=None, title=None,
)
# Returns: Line2D (or None if show/save)
```

### Vector data plot

2D vector data plots as streamlines by default:

```python
data.j1xy[-1, 100, :, :].plot()                # streamlines
data.j1xy[-1, 100, :, :].plot(mode="quiver")   # quiver arrows
```

### Common patterns

```python
# Save to file
data.phisp[-1, 100, :, :].plot(savefilename="phisp.png")

# Log scale for density
data.nd1p[-1, 100, :, :].plot(norm="log", vmin=1e-3, vmax=20)

# Contour overlay
data.phisp[-1, 100, :, :].plot(mode="cm+cont")

# EMSES raw units instead of SI
data.phisp[-1, 100, :, :].plot(use_si=False)
```

### matplotlib integration

emout uses matplotlib internally. `plot()` returns standard matplotlib artists (`AxesImage`, `Line2D`, etc.), so you can customize anything after the call using the normal matplotlib API:

```python
import matplotlib.pyplot as plt

# plot() returns an AxesImage — grab the axes and customize
im = data.phisp[-1, 100, :, :].plot()
ax = im.axes
ax.set_title("Custom Title")
ax.set_xlabel("x [m]")
ax.axhline(y=50, color="red", linestyle="--")
plt.savefig("custom.png", dpi=150, bbox_inches="tight")

# Or set up your own figure/axes first
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plt.sca(axes[0])
data.phisp[-1, 100, :, :].plot()
plt.sca(axes[1])
data.nd1p[-1, 100, :, :].plot(norm="log", vmin=1e-3, vmax=20)
plt.sca(axes[2])
data.j1xy[-1, 100, :, :].plot()
fig.suptitle("Overview")
fig.tight_layout()
plt.savefig("multi.png")

# Use raw numpy data for fully custom plots
phi = data.phisp[-1, :, 32, :].val_si   # plain numpy array in SI
```

Key points:
- `plot()` draws on `plt.gca()` — use `plt.sca(ax)` or `plt.subplot()` to target a specific axes before calling
- Returned artists can be modified (`im.set_clim(...)`, `line.set_color(...)`, etc.)
- `plt.rcParams` changes (font, dpi, style) affect emout plots
- For full control, extract `.val_si` arrays and plot with pure matplotlib

---

## gifplot()

Call `.gifplot()` on data that includes the time axis (first axis).

```python
data.phisp[:, 100, :, :].gifplot()
```

**Signature:**

```python
.gifplot(
    fig=None,              # plt.Figure or None
    axis=0,                # animation axis
    mode=None,             # "cmap", "cont", "stream", or None (auto)
    action="to_html",      # "to_html" | "show" | "save" | "return" | "frames"
    filename=None,         # for action="save"
    interval=200,          # frame interval [ms]
    repeat=True,
    title=None,
    notitle=False,
    offsets=None,
    use_si=True,
    vmin=None, vmax=None,
    **kwargs,              # norm="log", etc.
)
```

**Actions:**

| action | Returns | Use case |
| --- | --- | --- |
| `"to_html"` | HTML string | Jupyter inline display |
| `"show"` | None | matplotlib window |
| `"save"` | None | save to `filename` (e.g. "out.gif") |
| `"return"` | `(fig, animation)` | manual control |
| `"frames"` | `FrameUpdater` | multi-panel layout |

### Multi-panel animation

```python
u0 = data.phisp[:, 100, :, :].gifplot(action="frames", mode="cmap")
u1 = data.nd1p[:, 100, :, :].build_frame_updater(mode="cmap", norm="log", vmin=1e-3, vmax=20)
u2 = data.j1xy[:, 100, :, :].build_frame_updater(mode="stream")

# layout: [row][col][overlay]
layout = [[ [u0], [u1], [u2] ]]
animator = u0.to_animator(layout=layout)
animator.plot(action="to_html")  # or "save", "show"
```

---

## data.boundaries — Boundary meshes

Access MPIEMSES finbound/legacy boundaries as Python objects. Requires `boundary_type` / `boundary_types` in `plasma.inp`.

### Basic access

```python
data.boundaries                    # BoundaryCollection (iterable, indexable)
len(data.boundaries)               # number of boundaries
data.boundaries.types              # list of boundary_types strings
data.boundaries.skipped            # [(index, type_name, reason), ...]

data.boundaries[0]                 # Boundary subclass (SphereBoundary, etc.)
data.boundaries[0].btype           # e.g. "sphere", "cylinderz"
data.boundaries[0].mesh()          # → MeshSurface3D (SI units by default)
data.boundaries[0].mesh(use_si=False)  # grid units
```

### Overlay on 3D field plot

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

data.phisp[-1].plot_surfaces(
    ax=ax,
    surfaces=data.boundaries,      # auto-wrapped into RenderItems
    use_si=True,
)
plt.show()
```

### Composite mesh and per-boundary styling

```python
# All boundaries as one mesh
composite = data.boundaries.mesh()     # → CompositeMeshSurface
V, F = composite.mesh()               # raw (vertices, faces) arrays

# Combine individual boundaries
combined = data.boundaries[0] + data.boundaries[1]

# Override resolution
data.boundaries[0].mesh(ntheta=64)
data.boundaries.mesh(per={0: dict(ntheta=64)})
```

### Supported boundary types

| Category | Type names |
| --- | --- |
| Closed solids | `sphere`, `cuboid` |
| Cylinders | `cylinderx/y/z`, `open-cylinderx/y/z` |
| Flat panels | `rectangle`, `circlex/y/z`, `diskx/y/z`, `plane-with-circlex/y/z` |
| Legacy single-body | `flat-surface`, `rectangle-hole`, `cylinder-hole` |

Unregistered types are recorded in `data.boundaries.skipped` instead of raising an error.

---

## data.toml — Parameter file (recommended for plasma.toml)

When `plasma.toml` is present, emout runs the `toml2inp` command to generate `plasma.inp`, then loads it. **Use `data.toml` for native TOML structure access** via the `TomlData` wrapper.

```python
data.toml                        # TomlData object (None if plasma.inp only)
data.toml.tmgrid.nx              # attribute access
data.toml["tmgrid"]["nx"]        # dict-style access
data.toml.species[0].wp          # nested structures (V2 format)
data.toml.meta.unit_conversion.dx  # unit conversion key
```

> **Requirement:** `toml2inp` must be on PATH (bundled with [MPIEMSES3D](https://github.com/Nkzono99/MPIEMSES3D)). If missing, a warning is logged and only an existing `plasma.inp` is loaded.

### TomlData API

`TomlData` wraps the raw TOML dictionary. Nested dicts become `TomlData`, lists of dicts become lists of `TomlData`.

```python
data.toml.tmgrid.nx              # attribute access → value
data.toml["tmgrid"]["nx"]        # dict-style access → value
data.toml.tmgrid.keys()          # dict-like: keys(), values(), items(), get()
data.toml.tmgrid.to_dict()       # unwrap to plain dict
"tmgrid" in data.toml            # containment check
```

### When to use data.toml vs data.inp

| | `data.toml` | `data.inp` |
| --- | --- | --- |
| Available when | `plasma.toml` exists | Always (`toml2inp` generates it from .toml) |
| Structure | Native TOML (nested, lists) | Flat namelist (group → key → value) |
| V2 species access | `data.toml.species[0].wp` | `data.inp.wp[0]` (flat list) |
| **Recommendation** | **Preferred when plasma.toml is used** | Fallback for plasma.inp-only projects |

### data.inp — Legacy / fallback

`data.inp` is always available. When `plasma.toml` exists, `toml2inp` auto-generates `plasma.inp`:

```python
data.inp                     # InpFile object (dict-like)
data.inp["tmgrid"]["nx"]     # group + key
data.inp["nx"]               # key only (if unambiguous)
data.inp.nx                  # attribute access
data.inp.tmgrid.nx           # group.key attribute access
```

### Key parameters (via data.inp)

```python
# Grid
data.inp.nx, data.inp.ny, data.inp.nz   # grid dimensions
data.inp.dx                              # grid spacing (EMSES units)

# Time
data.inp.dt                  # time step
data.inp.ifdiag              # output interval (steps)
data.inp.nstep               # total steps

# Plasma
data.inp.nspec               # number of species
data.inp.wp                  # plasma frequency per species (list)
data.inp.qm                  # charge-to-mass ratio per species (list)
data.inp.path                # thermal velocity per species (list)

# Boundary
data.inp.mtd_vbnd            # boundary type per axis [x,y,z]: 0=periodic, 1=Dirichlet, 2=Neumann
```

---

## data.unit — Unit conversion

Requires `!!key dx=[...],to_c=[...]` in the first line of `plasma.inp` (or `[meta.unit_conversion]` in `plasma.toml`).
If absent, `data.unit` is `None`.

### UnitTranslator API

```python
data.unit.v.trans(1e5)       # SI → EMSES: 1e5 m/s → EMSES velocity
data.unit.v.reverse(4.107)   # EMSES → SI: 4.107 → m/s
data.unit.phi                # potential translator
data.unit.E                  # electric field translator
data.unit.B                  # magnetic flux density translator
data.unit.J                  # current density translator
data.unit.n                  # number density translator
data.unit.t                  # time translator
data.unit.length             # length translator
```

### .val_si property

```python
data.phisp[-1].val_si              # full 3D array in SI [V]
data.phisp[-1, :, 32, :].val_si   # sliced 2D in SI
data.j1z[-1].val_si               # [A/m^2]
data.nd1p[-1].val_si              # [/m^3]
```

### Available translators

`phi`(V), `E`(V/m), `B`(T), `J`(A/m²), `n`(/m³), `rho`(C/m³), `v`(m/s), `t`(s), `f`(Hz), `length`(m), `q`(C), `m`(kg), `W`(J), `w`(J/m³), `P`(W), `T`(K), `F`(N), `a`(m/s²), `i`(A), `N`(/m²s), `c`(m/s), `eps`(F/m), `mu`(H/m), `C`(F), `L`(H), `G`(S), `q_m`(C/kg), `qe`(C), `qe_me`(C/kg), `kB`(J/K), `e0`(F/m), `m0`(N/A²)

---

## Particle data

### Time series access

```python
p4 = data.p4                # ParticlesSeries for species 4
p4.x, p4.y, p4.z            # position time series
p4.vx, p4.vy, p4.vz         # velocity time series
p4.tid                       # trace ID time series

p4.x[0]                     # ParticleData at timestep 0
p4.vx[0].val_si             # SI velocity array
p4.vx[0].to_series()        # → pandas.Series
```

### Snapshots and phase-space plots

Indexing a `ParticlesSeries` by timestep returns a `ParticleSnapshot` bundling all components:

```python
snap = data.p4[0]            # ParticleSnapshot at timestep 0

# Access components
snap.x, snap.vx, snap.tid   # ParticleData for each component
snap.keys()                  # available component names
snap.to_dataframe()          # → pandas.DataFrame

# Phase-space plots (shorthand attribute syntax)
snap.xvx()                   # scatter plot of x vs vx
snap.yvz()                   # scatter plot of y vs vz
snap.zvz()                   # etc. — any pair of (x,y,z,vx,vy,vz)

# Explicit call with options
snap.plot_phase_space(
    "x", "vx",
    kind="scatter",          # "scatter" or "hist2d"
    use_si=True,
    bins=64,                 # for hist2d
    ax=None,                 # target matplotlib axes
)
```

---

## 3D plotting (requires `emout[pyvista]`)

```python
data.phisp[-1, :, :, :].plot3d(mode="box", show=True)       # volume
data.phisp[-1, 100, :, :].plot3d(show=True)                  # 2D slice in 3D space
data.j1xyz[-1].plot3d(mode="stream", show=True)              # 3D streamlines
data.j1xyz[-1].plot3d(mode="quiver", show=True)              # 3D quiver
```

### 3D mesh surface rendering

Overlay boundary meshes on 3D scalar field:

```python
import matplotlib.pyplot as plt
from emout.plot.surface_cut import BoxMeshSurface, CylinderMeshSurface, RenderItem, plot_surfaces

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Using data.boundaries (simplest)
data.phisp[-1].plot_surfaces(ax=ax, surfaces=data.boundaries)

# Using explicit mesh surfaces with RenderItem for style control
plot_surfaces(
    ax,
    field=field3d,
    surfaces=[
        RenderItem(BoxMeshSurface(0, 10, 0, 6, 0, 4, faces=("zmax",)), style="field"),
        RenderItem(CylinderMeshSurface(center=(5, 3, 2), axis="z", radius=1.5, length=4.0),
                   style="solid", solid_color="0.7", alpha=0.5),
    ],
)
```

### VTK export

```python
data.phisp[-1].to_vtk("output.vti", use_si=True)
# → writes VTK ImageData (.vti) file for ParaView / PyVista
```

---

## Remote execution (Dask) — experimental

Offload heavy data processing to HPC compute nodes. Only plot images or small slices are returned.
Automatically available on Python ≥ 3.10 (Dask is included in core dependencies).

### Server management (CLI)

```bash
emout server start --partition gr20001a --memory 60G --walltime 03:00:00
emout server stop
emout server status
```

`emout server` now enables TLS authentication automatically. By default
it keeps one active server session per user; if you intentionally need
more than one, use `emout server start --allow-multiple --name <session>`
and connect explicitly with `connect(name="<session>")`.

### Three usage modes

| Mode | Entry point | What goes to the client | When to use |
| --- | --- | --- | --- |
| Compat (data-transfer) | bare `data.phisp[...].plot()` | Sliced numpy array (KB–MB) | Least disruptive; existing scripts work unchanged when a server is running |
| Image | `with remote_figure()` | PNG/JPEG bytes or saved image file | Minimal local memory; full matplotlib runs on worker |
| Explicit proxy (recommended) | `data.remote()` + `remote_scope()` | `RemoteRef` proxies (a few bytes each) | Heavy workflows, repeated visualisation, backtrace, gifplot |

### Compat mode (data-transfer)

```python
# Worker extracts slice, matplotlib runs locally
data.phisp[-1, :, 100, :].plot()
plt.xlabel("x [m]")               # ← local matplotlib, full customization
```

### Image mode (`remote_figure`)

```python
from emout.distributed import remote_figure

with remote_figure():
    data.phisp[-1, :, 100, :].plot()
    plt.axhline(y=50, color="red")
    plt.title("Custom title")
# ← PNG displayed in Jupyter here
```

For CLI / batch workflows, save the rendered image directly:

```python
with remote_figure(savefilepath="figures/phisp.png"):
    data.phisp[-1, :, 100, :].plot()
    plt.title("Saved remotely")

# Format is inferred from the extension when fmt is omitted
with remote_figure(savefilepath="figures/phisp.svg"):
    data.phisp[-1, :, 100, :].plot()
```

When `savefilepath` is provided, the rendered bytes are written to disk.
PNG/JPEG output is still displayed inline in IPython, while CLI / batch
usage skips local display.

`remote_figure` also supports explicit open/close and options:

```python
from emout.distributed import RemoteFigure

rf = RemoteFigure(dpi=300, savefilepath="figures/phisp.png")
rf.open()
data.phisp[-1, :, 100, :].plot()
rf.close()                         # ← replays commands on server, saves the image
```

### Explicit proxy mode — `Emout.remote()` + `remote_scope()`

`Emout.remote()` returns a `RemoteEmout` proxy; attribute access and slicing return `RemoteRef`
proxies whose real objects stay in worker memory. Expressions like `-ref`, `ref1 + ref2`,
`np.abs(ref)`, `int(ref)` all stay remote until you fetch explicitly.

`remote_scope()` tracks every ref created inside it and calls `drop()` on them when the scope
exits — worker memory is released in one go.

```python
import matplotlib.pyplot as plt
import numpy as np
import emout
from emout.distributed import remote_figure, remote_scope

rdata = emout.Emout("output_dir").remote()

with remote_scope():
    ymid = int(rdata.inp.ny // 2)            # int() coerces a RemoteRef → local int

    phi  = rdata.phisp[-1, :, ymid, :]       # RemoteRef (no data transferred)
    ez   = -rdata.exz[-1, :, ymid, :]        # still remote
    peak = np.abs(ez).max()                  # numpy ufuncs dispatch through the ref

    with remote_figure():
        plt.figure(figsize=(12, 6))
        phi.plot()
        ez.plot()
        plt.title(f"peak |Ez| = {float(peak):.2e}")
```

### `RemoteRef` — what you can do with it

| Operation | Remote? | Notes |
| --- | --- | --- |
| `ref.plot()`, `ref.gifplot()`, `ref.plot_surfaces(...)` | yes | Renders on worker inside `remote_figure()` |
| `-ref`, `ref + other`, `ref * 2` | yes | Returns a new `RemoteRef` |
| `np.abs(ref)`, `np.sqrt(ref)`, other ufuncs | yes | Dispatched via `__array_ufunc__` |
| `int(ref)`, `float(ref)` | fetch-then-coerce | Triggers a small transfer |
| `ref.fetch()` | **transfer** | Returns a local array / `Data*` object |
| `ref.drop()` | yes | Frees worker memory immediately |

### Scope control — `open()` / `close()` / `clear()` and nesting

`remote_scope()` can be used with `with`, or driven explicitly for Jupyter cells that span
multiple blocks.

```python
from emout.distributed import remote_scope

scope = remote_scope()
scope.open()                         # enter scope (idempotent)

rdata = data.remote()
ref = rdata.phisp[-1, :, 100, :]
ref.plot()
# ... continue in later cells ...

scope.close()                        # drops every ref registered to this scope
```

- `scope.clear()` — drops every registered ref but **keeps the scope open**. Use in long
  loops where worker memory would otherwise grow every iteration.
- `scope.close()` is idempotent; calling it twice is a no-op.
- Scopes nest: refs are always registered to the **innermost** active scope. You can mix
  `open()` and `with remote_scope():` as long as each is a **distinct** `remote_scope()`
  instance.

```python
scope1 = remote_scope(); scope1.open()

with remote_scope() as scope2:
    ref_inner = rdata.phisp[-1, :, 100, :]   # tracked by scope2
# scope2 auto-drops here; scope1 is still open

ref_outer = rdata.exz[-1]                     # tracked by scope1
scope1.close()
```

> **Footgun:** never hand the same instance to both `open()` and `with` — `__exit__` runs
> once and the instance is then closed forever. Always create a new `remote_scope()` for
> the inner `with`.

### Remote gifplot

`RemoteRef.gifplot()` runs frame generation **and** encoding on the worker; only the HTML
(or GIF bytes) comes back.

```python
rdata = emout.Emout("output_dir").remote()

with remote_scope():
    # Inline HTML for Jupyter (default)
    rdata.phisp[:, 100, :, :].gifplot()

    # Save a GIF to a path visible to the worker (shared FS)
    rdata.phisp[:, 100, :, :].gifplot(action="save", filename="phisp.gif")

    # Raw GIF bytes in client memory
    gif_bytes = rdata.phisp[:, 100, :, :].gifplot(action="bytes")
```

Actions `"show"`, `"return"`, `"frames"` are **not** supported over remote — they require
objects that cannot be sent across the wire.

### Remote backtrace

Heavy particle-backtrace computations run once on the worker and stay there. Re-render with
different visualisation parameters without recomputing.

```python
rdata = emout.Emout("output_dir").remote()

with remote_scope():
    result = rdata.backtrace.get_probabilities(
        x, y, z, vx_range, vy_center, vz_range, ispec=0,
    )

    with remote_figure():
        result.vxvz.plot(cmap="viridis")

    with remote_figure():
        result.plot_energy_spectrum(scale="log")

    heatmap = result.vxvz.fetch()     # ← small local copy for custom annotation
```

Both `data.backtrace...` and `data.remote().backtrace...` return the same
`RemoteProbabilityResult` / `RemoteBacktraceResult` proxies.

### Jupyter cell magic

```python
%load_ext emout.distributed.remote_figure

%%remote_figure --dpi 300 --fmt svg --figsize 12,6
data.phisp[-1, :, 100, :].plot()

%%remote_figure --savefilepath figures/phisp.png
data.phisp[-1, :, 100, :].plot()
```

---

## Poisson solver (experimental)

```python
from emout.utils import poisson
import scipy.constants as cn

rho = data.rho[-1].val_si
dx = data.inp.dx
btypes = ["pdn"[i] for i in data.inp.mtd_vbnd]
phi = poisson(rho, dx=dx, btypes=btypes, epsilon_0=cn.epsilon_0)
```

---

## Backtrace (experimental, requires `vdist-solver-fortran`)

```python
result = data.backtrace.get_probabilities(
    ix, iy, iz,
    (vx_min, vx_max, nvx),
    vy_center,
    (vz_min, vz_max, nvz),
    ispec=0,
)
result.vxvz.plot()

bt = data.backtrace.get_backtraces_from_particles(result.particles, ispec=0)
bt.xz.plot(color="black", alpha=alpha_values)
```

---

## Typical workflow template

### With plasma.toml (recommended)

```python
import emout

data = emout.Emout("output_dir")

# Parameter access via TOML
nx = data.toml.tmgrid.nx
ny = data.toml.tmgrid.ny
electron_wp = data.toml.species[0].wp

# 2D visualization
data.phisp[-1, :, ny // 2, :].plot()

# Time animation
data.phisp[:, data.toml.tmgrid.nz // 2, :, :].gifplot(action="save", filename="phisp.gif")

# SI values for analysis
phi_si = data.phisp[-1].val_si  # [V]
v_emses = data.unit.v.trans(1e5)  # 1e5 m/s → EMSES

# Boundary overlay
data.phisp[-1].plot_surfaces(surfaces=data.boundaries)

# Particle phase-space
data.p4[0].xvx()
```

### With plasma.inp (legacy)

```python
import emout

data = emout.Emout("output_dir")
nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz

data.phisp[-1, :, ny // 2, :].plot()
```

---

## Gotchas for agents

1. **Axis order is (t, z, y, x)**, not (t, x, y, z). When slicing `data.phisp[-1, :, ny//2, :]`, the `:` positions correspond to z and x.
2. **Prefer `data.toml` over `data.inp`** when `plasma.toml` is present. `data.toml` preserves the native TOML structure (nested dicts, species lists), while `data.inp` flattens everything into a namelist.
3. **`data.toml` is `None`** when only `plasma.inp` exists. Guard with `if data.toml is not None:` before using it.
4. **`data.unit` can be `None`** if `plasma.inp` lacks the `!!key` header (or `plasma.toml` lacks `[meta.unit_conversion]`). Always guard: `if data.unit is not None:`.
5. **`data.boundaries` needs `&ptcond`** in the parameter file. If no boundaries are defined, `data.boundaries` is empty (not `None`).
6. **`plot()` returns an artist object** unless `show=True` or `savefilename` is set. In scripts, add `show=True` or call `plt.show()`.
7. **`gifplot()` defaults to `action="to_html"`** which only works in Jupyter. For scripts, use `action="save"` or `action="show"`.
8. **Variable names are EMSES conventions**: `phisp` (potential), `nd{i}p` (species-i density), `j{i}x/y/z` (species-i current), `ex/ey/ez` (E-field), `bx/by/bz` (B-field).
9. **Vector auto-combination**: `data.j1xy` combines j1x+j1y. `data.j1xyz` combines j1x+j1y+j1z. Don't manually concatenate.
10. **`data.inp["nx"]`** works only if `nx` is unambiguous across all groups. If ambiguous, use `data.inp["tmgrid"]["nx"]`.
11. **`val_si` is a property**, not a method. Use `data.phisp[-1].val_si`, not `data.phisp[-1].val_si()`.
12. **Data manipulation methods (`flip`, `mirror`, `tile`, `negate`, `scale`) return copies**. They do not modify the original array.
13. **Phase-space shorthand** (`snap.xvx()`) works for any pair of `(x, y, z, vx, vy, vz)`. The first variable is the horizontal axis.
14. **Remote execution is transparent in compat mode**. When an emout server is running, existing `data.phisp[...].plot()` code automatically uses it (worker extracts slice, client renders). For new code, prefer the explicit `data.remote()` + `remote_scope()` workflow — it keeps big arrays on the worker as `RemoteRef` proxies and frees them in one go at scope exit.
15. **`remote_scope()` auto-drops refs** created inside it when the `with` block exits (or `close()` is called). Use `scope.clear()` inside long loops to release per-iteration refs without exiting the scope. Scopes nest: refs always attach to the innermost active scope.
16. **`RemoteRef.gifplot()` only supports `action="to_html" | "save" | "bytes"`**. `"show"`, `"return"`, `"frames"` are compat-mode only — they cannot cross the worker boundary.
