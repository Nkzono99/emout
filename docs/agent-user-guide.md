# emout — AI Agent User Guide

This document is a compact, structured reference for AI agents that generate code using the `emout` library.
Import this file via `@docs/agent-user-guide.md` to get the full API context.

## What emout does

`emout` reads EMSES PIC simulation outputs (HDF5 grid data, particle data, `plasma.inp`/`plasma.toml` parameter files) and provides:

- One-line 1D/2D/3D plotting with automatic SI unit conversion
- GIF/HTML animation creation from time-series data
- Parameter file access as a dictionary-like object
- Bidirectional EMSES↔SI unit conversion
- Particle data grouping and pandas integration

## Installation

```bash
pip install emout              # Core
pip install "emout[pyvista]"   # + 3D visualization
```

---

## Core API

### Loading data

```python
import emout

data = emout.Emout("output_dir")
# data = emout.Emout("output_dir", ad="auto")                          # auto-detect appended dirs
# data = emout.Emout("output_dir", append_directories=["dir2", "dir3"]) # manual append
```

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
    mode="cm",             # "cm" (colormap), "cont" (contour), "cm+cont"
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
data.j1xy[-1, 100, :, :].plot()          # streamlines
data.j1xy[-1, 100, :, :].plot(mode="quiver")  # quiver arrows
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

# Global style changes apply to emout plots too
plt.rcParams.update({"font.size": 14, "figure.dpi": 120})
data.phisp[-1, 100, :, :].plot()

# Use raw numpy data for fully custom plots
phi = data.phisp[-1, :, 32, :].val_si   # plain numpy array
x = data.phisp[-1, :, 32, :].x_si       # SI x-axis
z = data.phisp[-1, :, 32, :].z_si       # SI z-axis
plt.contourf(x, z, phi, levels=30, cmap="RdBu_r")
plt.colorbar(label="Potential [V]")
```

Key points:
- `plot()` draws on `plt.gca()` — use `plt.sca(ax)` or `plt.subplot()` to target a specific axes before calling
- Returned artists can be modified (`im.set_clim(...)`, `line.set_color(...)`, etc.)
- `plt.rcParams` changes (font, dpi, style) affect emout plots
- For full control, extract `.val_si` / `.x_si` / `.z_si` arrays and plot with pure matplotlib

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

## data.toml — Parameter file (recommended for plasma.toml)

When `plasma.toml` is present, **use `data.toml` instead of `data.inp`**. It provides the native TOML structure with attribute access via the `TomlData` wrapper.

```python
data.toml                        # TomlData object (None if plasma.inp only)
data.toml.tmgrid.nx              # attribute access
data.toml["tmgrid"]["nx"]        # dict-style access
data.toml.plasma.species[0].wp   # nested structures (V2 format)
data.toml.meta.unit_conversion.dx  # unit conversion key
```

### TomlData API

`TomlData` wraps the raw TOML dictionary. Nested dicts become `TomlData`, lists of dicts become lists of `TomlData`.

```python
data.toml.tmgrid.nx              # attribute access → value
data.toml["tmgrid"]["nx"]        # dict-style access → value
data.toml.tmgrid.keys()          # dict-like: keys(), values(), items(), get()
data.toml.tmgrid.to_dict()       # unwrap to plain dict
"tmgrid" in data.toml            # containment check
```

### TOML structure example (V2 format)

```toml
[meta.unit_conversion]
dx = 0.5
to_c = 10000.0

[tmgrid]
nx = 256
ny = 256
nz = 512
dt = 0.5
ifdiag = 100
nstep = 50000

[plasma]
nspec = 2

[[plasma.species]]
wp = 1.0
qm = -1.0
path = [0.1, 0.1, 0.1]

[[plasma.species]]
wp = 0.05
qm = 0.001
path = [0.01, 0.01, 0.01]
```

Access:

```python
data.toml.plasma.species[0].wp    # 1.0 (electron)
data.toml.plasma.species[1].qm    # 0.001 (ion)
data.toml.tmgrid.nx               # 256
data.toml.meta.unit_conversion.dx  # 0.5
```

### When to use data.toml vs data.inp

| | `data.toml` | `data.inp` |
| --- | --- | --- |
| Available when | `plasma.toml` exists | Always (auto-converted from .toml too) |
| Structure | Native TOML (nested, lists) | Flat namelist (group → key → value) |
| V2 species access | `data.toml.plasma.species[0].wp` | `data.inp.wp[0]` (flat list) |
| **Recommendation** | **Preferred when plasma.toml is used** | Fallback for plasma.inp-only projects |

### data.inp — Legacy / fallback

`data.inp` is always available regardless of file format. It flattens TOML into namelist-style access:

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

`phi`(V), `E`(V/m), `B`(T), `J`(A/m^2), `n`(/m^3), `rho`(C/m^3), `v`(m/s), `t`(s), `f`(Hz), `length`(m), `q`(C), `m`(kg), `W`(J), `w`(J/m^3), `P`(W), `T`(K), `F`(N), `a`(m/s^2), `i`(A), `N`(/m^2s), `c`(m/s), `eps`(F/m), `mu`(H/m), `C`(F), `L`(H), `G`(S), `q_m`(C/kg), `qe`(C), `qe_me`(C/kg), `kB`(J/K), `e0`(F/m), `m0`(N/A^2)

---

## Particle data

```python
p4 = data.p4                # species 4
p4.x[0]                     # x-positions at timestep 0 → ParticleData1d
p4.vx[0].val_si             # SI velocity array
p4.vx[0].to_series()        # → pandas.Series
p4.vx[0].val_si.to_series().hist(bins=200)
# Components: x, y, z, vx, vy, vz, tid
```

---

## Data masking

```python
data.phisp[-1].masked(lambda phi: phi < phi.mean())          # returns copy with NaN
data.phisp[-1].masked(lambda phi: phi < phi.mean()).plot()    # plot masked data
```

---

## 3D plotting (requires `emout[pyvista]`)

```python
data.phisp[-1, :, :, :].plot3d(mode="box", show=True)       # volume
data.phisp[-1, 100, :, :].plot3d(show=True)                  # 2D slice in 3D space
data.j1xyz[-1].plot3d(mode="stream", show=True)              # 3D streamlines
data.j1xyz[-1].plot3d(mode="quiver", show=True)              # 3D quiver
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

# Use data.toml for parameter access
nx = data.toml.tmgrid.nx
ny = data.toml.tmgrid.ny
nz = data.toml.tmgrid.nz

# V2 species access
electron_wp = data.toml.plasma.species[0].wp
ion_qm = data.toml.plasma.species[1].qm

# Quick 2D visualization
data.phisp[-1, :, ny // 2, :].plot()

# Time animation
data.phisp[:, nz // 2, :, :].gifplot(action="save", filename="phisp.gif")

# SI values for analysis
phi_si = data.phisp[-1].val_si  # [V]

# Unit conversion
v_emses = data.unit.v.trans(1e5)  # 1e5 m/s → EMSES
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
5. **`plot()` returns an artist object** unless `show=True` or `savefilename` is set. In scripts, add `show=True` or call `plt.show()`.
6. **`gifplot()` defaults to `action="to_html"`** which only works in Jupyter. For scripts, use `action="save"` or `action="show"`.
7. **Variable names are EMSES conventions**: `phisp` (potential), `nd{i}p` (species-i density), `j{i}x/y/z` (species-i current), `ex/ey/ez` (E-field), `bx/by/bz` (B-field).
8. **Vector auto-combination**: `data.j1xy` combines j1x+j1y. `data.j1xyz` combines j1x+j1y+j1z. Don't manually concatenate.
9. **`data.inp["nx"]`** works only if `nx` is unambiguous across all groups. If ambiguous, use `data.inp["tmgrid"]["nx"]` or prefer `data.toml.tmgrid.nx`.
10. **`val_si` is a property**, not a method. Use `data.phisp[-1].val_si`, not `data.phisp[-1].val_si()`.
