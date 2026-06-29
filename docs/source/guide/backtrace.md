# Backtrace (`data.trace` / `data.backtrace`) — Experimental

`data.trace` is the high-level workflow API for combining arrival
probabilities, backward traces, and forward traces. The older
`data.backtrace` entry point remains available as the lower-level API for
single-particle traces and existing code.

Calling `data.trace.forward(..., get_trace=True)` builds particles from a
6-D phase-space grid, computes arrival probabilities when requested, and
returns one result object that can plot probability-weighted trajectories.

> **Requirements:** backtrace relies on the external
> [`vdist-solver-fortran`](https://github.com/Nkzono99/vdist-solver-fortran)
> package (`vdsolverf`). Install it with `pip install vdist-solver-fortran`.
> Without it, calls to `data.trace.*` / `data.backtrace.*` raise
> `ImportError`.

## Input Unit Contract

The `position`, `velocity`, and `dt` values passed to `data.trace` /
`data.backtrace`, and the phase-space-grid `x` / `y` / `z` / `vx` / `vy` /
`vz` axes, are **all EMSES simulation units**. emout does not convert
these inputs from SI; it forwards them to `vdsolverf` unchanged.

If you want to specify SI values, convert them to EMSES units with
`data.unit` before calling the backtrace APIs:

```python
position = (
    data.unit.length.trans(0.20),  # m -> EMSES length
    data.unit.length.trans(0.32),
    data.unit.length.trans(0.40),
)
velocity = (
    data.unit.v.trans(1.0e5),      # m/s -> EMSES velocity
    0.0,
    data.unit.v.trans(-2.0e5),
)
vx_scan = (
    data.unit.v.trans(-3.0e5),
    data.unit.v.trans(3.0e5),
    64,
)
```

`data.unit` is available only when `plasma.inp` contains a
`!!key dx=...,to_c=...` header, or `plasma.toml` contains
`[meta.unit_conversion]`. If unit-conversion metadata is absent, pass
values that are already in EMSES units.

Arrays such as `bt.positions`, `bt.velocities`, and `result.phases` also
remain in EMSES units. Plot helpers such as `bt.xz.plot()` and
`result.vxvz.plot()` convert displayed axes to SI by default when unit
metadata is available (`use_si=False` keeps EMSES-unit display).

The default `dt` is `data.inp.dt`. To integrate in the opposite direction
from the usual backtrace, pass the opposite sign, for example
`dt=-data.inp.dt`.

## When to use it

- You want the **phase-space distribution** of particles that arrive at a
  given observation point.
- You want to **trace back the trajectory** of a particle of interest to
  see where it came from.
- You want to draw an **energy spectrum** of arriving particles.

Backtrace integrates an ODE backwards using `data.inp.dt` and the saved
EMSES fields, so a large `max_step` can become expensive. If you want to
push the work to an HPC node, combine it with the remote-execution
backend (see below).

## Quick start

```python
import emout

data = emout.Emout("output_dir")

# Single particle
position = (20.0, 32.0, 40.0)
velocity = (
    data.unit.v.trans(1.0e5),
    0.0,
    data.unit.v.trans(-2.0e5),
)
bt = data.backtrace.get_backtrace(position, velocity, ispec=0)

bt.tx.plot()      # t vs x trajectory
bt.xvz.plot()     # x vs vz phase space

# Many particles in one call
import numpy as np
positions = np.array([[20, 32, 40], [21, 32, 40], [22, 32, 40]], dtype=float)
velocities = np.zeros_like(positions)
velocities[:, 0] = data.unit.v.trans(1.0e5)
many = data.backtrace.get_backtraces(positions, velocities, ispec=0)

many.xz.plot(alpha=0.5)    # overlay all trajectories

# Arrival probability over a 6-D phase-space grid
vx_scan = (data.unit.v.trans(-3e5), data.unit.v.trans(3e5), 64)
vz_scan = (data.unit.v.trans(-3e5), data.unit.v.trans(3e5), 64)
result = data.backtrace.get_probabilities(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    ispec=0,
)

result.vxvz.plot(cmap="viridis")      # heatmap in the vx-vz plane
result.plot_energy_spectrum(scale="log")

# High-level workflow: probability + forward trace in one result
trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    ispec=0,
    get_trace=True,
)

trace.plot("vx", "vz", cmap="viridis")      # ProbabilityResult projection
trace.plot_traces("x", "z")                 # probability-weighted trajectories
```

## High-level workflow: `data.trace`

`data.trace.backward()` / `data.trace.forward()` / `data.trace.both()`
always return a :class:`TraceResult`. Payloads that were not requested are
stored as `None`.

```python
trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    get_trace=True,
    get_probabilities=True,
)

trace.probabilities        # ProbabilityResult
trace.forward_traces       # MultiBacktraceResult
trace.backward_traces      # None
trace.alpha                # np.clip(trace.probabilities.probabilities, 0, 1)

trace.plot("vx", "vz")     # arrival-probability heatmap
trace.plot_traces("x", "z")
```

Set `get_probabilities=False` to skip the probability solve, create only
particles from the phase-space grid, and return trajectories. In that
case `trace.probabilities` and `trace.alpha` are `None`, and
`plot_traces()` uses a uniform alpha unless you pass one explicitly.

```python
trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    get_trace=True,
    get_probabilities=False,
)

trace.forward_traces.xz.plot(alpha=0.3)
trace.plot_traces("x", "z", alpha=0.3)
```

`both()` computes backward and forward trajectories from the same
phase-space grid. If probabilities are requested, the probability solve
runs only once.

```python
trace = data.trace.both(..., get_trace=True)
trace.backward_traces.xz.plot(alpha=trace.alpha)
trace.forward_traces.xz.plot(alpha=trace.alpha)
```

For 3-D views, `plot3d()` returns a PyVista plotter. Pass an existing
plotter to overlay traces on a field or boundary view.

```python
plotter = data.phisp[-1].plot3d(mode="slice", show=False)
trace.plot3d(plotter=plotter, direction="forward", tube_radius=0.05, show=True)
```

## Single particle: `get_backtrace`

`get_backtrace(position, velocity, ispec=0, ...)` integrates one
trajectory and returns a :class:`BacktraceResult`. The result also
supports tuple unpacking.

```python
bt = data.backtrace.get_backtrace(position, velocity, ispec=0, max_step=50000)

ts, prob, positions, velocities = bt   # tuple unpacking
print(bt)                                # <BacktraceResult: n_steps=...>
```

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `bt.ts` | `(N,)` | time (EMSES units) |
| `bt.probability` | `(N,)` | arrival probability per step |
| `bt.positions` | `(N, 3)` | `[x, y, z]` |
| `bt.velocities` | `(N, 3)` | `[vx, vy, vz]` |

### Forward tracing instead of backtracing

`dt` is forwarded to `vdsolverf` unchanged. To follow the same particle in
the opposite direction from the usual backtrace, flip the sign of
`data.inp.dt`.

```python
ft = data.backtrace.get_backtrace(
    position,
    velocity,
    ispec=0,
    dt=-data.inp.dt,
)
```

### Shorthand plotting

`bt.pair(var1, var2)` selects two variables and returns an
:class:`XYData`. `var1` and `var2` can each be `t`, `x`, `y`, `z`, `vx`,
`vy`, or `vz`. The concatenated form (`bt.tx`, `bt.xvz`, `bt.yz`, ...)
is shorthand for the same call.

```python
bt.tx.plot()                 # = bt.pair("t", "x")
bt.xvz.plot()                # = bt.pair("x", "vz")
bt.yz.plot(color="black")    # xy projection of the trajectory
```

`XYData.plot()` converts to SI units by default and auto-generates axis
labels (`use_si=False` keeps EMSES units). Passing `gap=...` inserts NaN
breaks where consecutive points are too far apart, which is handy to
avoid spurious lines across periodic-boundary jumps.

## Many particles: `get_backtraces`

`get_backtraces(positions, velocities, ispec=0, n_threads=4, ...)`
returns a :class:`MultiBacktraceResult`. `positions` and `velocities`
must be `(N, 3)` arrays.

```python
ts, probs, pos_list, vel_list, last = many
many.xz.plot(alpha=np.clip(probs, 0, 1))    # alpha weighted by probability
many.sample(50, random_state=0).tvx.plot()   # random 50 trajectories
many.sample(slice(0, 10)).tx.plot()         # first 10 trajectories
```

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `ts_list` | `(N_traj, N_steps)` | |
| `probabilities` | `(N_traj,)` | final arrival probability |
| `positions_list` | `(N_traj, N_steps, 3)` | |
| `velocities_list` | `(N_traj, N_steps, 3)` | |
| `last_indexes` | `(N_traj,)` | end of valid data for each trajectory (padding) |

`many.pair("t", "x")` returns a :class:`MultiXYData`; `.plot()` overlays
every trajectory. `alpha` accepts either a scalar or an array of length
`N_traj`, which is convenient for probability weighting.

### Feeding raw Particle objects

If you already have `vdsolverf.core.Particle` instances (for example from
`ProbabilityResult.particles`), use
`get_backtraces_from_particles(particles, ...)`:

```python
from vdsolverf.core import Particle

particles = [Particle(p, v) for p, v in zip(positions, velocities)]
many = data.backtrace.get_backtraces_from_particles(particles, ispec=0)
```

A common pattern is to chain this with `get_probabilities` — compute the
probability grid, then trace back only the particles you care about:

```python
result = data.backtrace.get_probabilities(...)
bt = data.backtrace.get_backtraces_from_particles(result.particles, ispec=0)
bt.xz.plot(alpha=np.clip(result.probabilities, 0, 1))
```

## Arrival probability: `get_probabilities`

`get_probabilities(x, y, z, vx, vy, vz, ispec=0, ...)` builds a 6-D
phase-space grid, backtraces a particle from every grid point, and
returns the arrival probabilities as a :class:`ProbabilityResult`.

All input axes are EMSES units. To scan a velocity range given in SI,
convert the endpoints with `data.unit.v.trans(...)` before building the
`(start, stop, n)` tuple.

Each axis accepts:

- a tuple `(start, stop, n)` — equally-spaced grid
- an explicit array or list — arbitrary values
- a scalar — a size-1 axis (automatically squeezed when you call `pair()`)

```python
vx_scan = (data.unit.v.trans(-3e5), data.unit.v.trans(3e5), 64)
vz_scan = (data.unit.v.trans(-3e5), data.unit.v.trans(3e5), 64)

result = data.backtrace.get_probabilities(
    x=20.0, y=32.0, z=40.0,     # fixed position
    vx=vx_scan,                 # scan vx over 64 points
    vy=0.0,
    vz=vz_scan,                 # scan vz over 64 points
    ispec=0,
    max_step=10000,
    n_threads=8,
)
```

### MPI backend

The default backend is unchanged and uses the threaded `vdsolverf.emses`
functions.  When `vdist-solver-fortran[mpi]` is installed, you can opt in to
particle-parallel MPI without changing the result object:

```python
# Use when the script itself is launched with MPI, e.g.
# srun -n 8 python script.py
vx_scan = (data.unit.v.trans(-3e5), data.unit.v.trans(3e5), 64)
vz_scan = (data.unit.v.trans(-3e5), data.unit.v.trans(3e5), 64)

result = data.backtrace.get_probabilities(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    max_step=10000,
    parallel="mpi",
    n_threads=2,
)

# Or launch Slurm from the current Python process.
result = data.backtrace.get_probabilities(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    max_step=10000,
    parallel="srun",
    ntasks=8,
    n_threads=2,
    cpus_per_task=2,
)
```

### 2-D heatmap projections

`result.pair(var1, var2)` integrates out the four unselected axes
(trapezoidal rule) and returns a :class:`HeatmapData`. The shorthand
attribute form works the same way as for `BacktraceResult`.

```python
result.vxvz.plot(cmap="viridis")   # = result.pair("vx", "vz")
result.xvx.plot()                  # x-vx plane
result.yz.plot(cmap="plasma")      # y-z plane
```

`HeatmapData.plot()` draws a `pcolormesh` with a colour bar and SI-unit
labels (`use_si=False` keeps grid units). Extra keyword arguments are
forwarded straight to `pcolormesh`, so you can use `vmin` / `vmax` or
`norm=LogNorm(...)` to control the colour scale. Pass
`offsets=("center", 0)` to centre an axis or apply a numeric shift.

### Energy spectrum

`plot_energy_spectrum(energy_bins=None, scale="log")` renders an
energy-flux histogram of the arriving particles. `energy_bins` accepts
either an integer (number of bins) or an array of bin edges.

```python
result.plot_energy_spectrum(scale="log", energy_bins=80)
```

Internally it reads `wp` (or the photoelectron settings `path` / `curf`
when `nflag_emit == 2`) from `plasma.inp` to compute a reference number
density `n0`, weights each phase-space point by its probability, and
integrates.

### Raw histogram arrays

`result.energy_spectrum(energy_bins=...)` returns the `(hist, bin_edges)`
tuple directly so you can feed it to custom post-processing or another
library (e.g. `ax.step`, seaborn).

## Remote execution integration

`data.backtrace` shares the `Emout` facade's `remote_open_kwargs`, so if
an emout server is running the computation automatically runs on the
worker and you get back a `RemoteProbabilityResult` /
`RemoteBacktraceResult` proxy. Because the result is cached on the
worker, changing visualisation parameters does **not** trigger
recomputation.

```python
from emout.distributed import remote_figure

result = data.backtrace.get_probabilities(...)   # computed once on the worker

with remote_figure():
    result.vxvz.plot(cmap="viridis")

with remote_figure():
    result.plot_energy_spectrum(scale="log")

result.drop()   # free worker memory when done
```

If you prefer the explicit remote style, switch to `data.remote().backtrace...`
— it returns the same dedicated proxies:

```python
from emout.distributed import remote_scope, remote_figure

with remote_scope():
    rdata = data.remote()
    bt = rdata.backtrace.get_backtrace(position, velocity, ispec=0)
    result = rdata.backtrace.get_probabilities(...)

    with remote_figure():
        bt.tx.plot()
        result.vxvz.plot()
```

For the remote-execution mechanics, environment variables, and server
management, see the [remote execution guide](distributed.md).

### `fetch()` for local customisation

When you want full matplotlib control (custom annotations, shared colour
bars, dropping the heatmap into your own subplot grid), use `fetch()` to
pull the small result arrays back to the client:

```python
heatmap = result.vxvz.fetch()      # local HeatmapData
fig, ax = plt.subplots()
heatmap.plot(ax=ax, cmap="plasma")
ax.axhline(y=0, color="red", linestyle="--")
```

## Related classes

See the API reference (the `emout.core.backtrace` package) for full
signatures.

- `BacktraceWrapper` — the `data.backtrace` object itself
- `BacktraceResult` / `MultiBacktraceResult` — trajectory containers
- `ProbabilityResult` — 6-D probability grid and heatmap projections
- `XYData` / `MultiXYData` / `HeatmapData` — lightweight visualisation containers
