# Backtrace (`data.trace`) — Experimental

`data.trace` is the workflow API for building particles from a 6-D phase-space grid and combining arrival probabilities, backward traces, and forward traces. For new analysis code, prefer `data.trace.forward()` / `data.trace.backward()` / `data.trace.both()`.

The older `data.backtrace` entry point remains available as a lower-level API for existing code that needs explicit single-particle traces or raw `Particle` arrays. This page keeps `data.trace` as the main path and folds the `data.backtrace` route into a details section.

> **Requirements:** backtrace relies on the external [`vdist-solver-fortran`](https://github.com/Nkzono99/vdist-solver-fortran) package (`vdsolverf`). Install it with `pip install vdist-solver-fortran`. Without it, calls to `data.trace.*` raise `ImportError`.

## Input Unit Contract

The `x` / `y` / `z` / `vx` / `vy` / `vz`, `dt`, and `probability_dt` values passed to `data.trace` are **all EMSES simulation units**. emout does not convert these inputs from SI.

If you want to specify SI values, convert them to EMSES units with `data.unit` before calling:

```python
position = (
    data.unit.length.trans(0.20),  # m -> EMSES length
    data.unit.length.trans(0.32),
    data.unit.length.trans(0.40),
)
vx_scan = (
    data.unit.v.trans(-3.0e5),     # m/s -> EMSES velocity
    data.unit.v.trans(3.0e5),
    64,
)
vz_scan = (
    data.unit.v.trans(-3.0e5),
    data.unit.v.trans(3.0e5),
    64,
)
```

`data.unit` is available only when `plasma.inp` contains a `!!key dx=...,to_c=...` header, or `plasma.toml` contains `[meta.unit_conversion]`. If unit-conversion metadata is absent, pass values that are already in EMSES units.

Arrays such as `TraceResult.phases`, `TraceResult.particles`, and `trace.traces.positions_list` also remain in EMSES units. Plot helpers such as `trace.plot()`, `trace.plot_traces()`, and `trace.traces.xz.plot()` convert displayed axes to SI by default when unit metadata is available (`use_si=False` keeps EMSES-unit display).

`dt` and `probability_dt` are non-negative step widths in EMSES time units. When they are `None`, emout uses `abs(data.inp.dt)`. `data.trace.backward()` passes `+dt` to the solver, while `data.trace.forward()` flips the sign internally and passes `-dt`. `data.trace.both()` uses the same `dt` magnitude as `+dt` for backward traces and `-dt` for forward traces. Arrival-probability solves use `probability_dt` with the backward-sign convention. Negative values raise `ValueError`.

## When to Use `data.trace`

- You want the **phase-space distribution** of particles that arrive at an observation point.
- You want to follow arriving particles as **backward / forward trajectories**.
- You want to draw an **energy spectrum** of arriving particles.
- You want probabilities and trajectories from the same particle set, with probability-derived alpha.

Backtrace integrates ODEs through saved field output, so large `max_step` values or fine phase-space grids can become expensive. If you want to push the work to an HPC node, combine it with remote execution (see below).

## Quick Start

```python
import emout

data = emout.Emout("output_dir")

vx_scan = (data.unit.v.trans(-3e5), data.unit.v.trans(3e5), 64)
vz_scan = (data.unit.v.trans(-3e5), data.unit.v.trans(3e5), 64)

trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    ispec=0,
    get_trace=True,
    get_probabilities=True,
    max_step=10000,
    n_threads=8,
)

trace.plot("vx", "vz", cmap="viridis")       # arrival-probability heatmap
trace.plot_traces("x", "z")                  # probability-weighted trajectories
trace.probabilities.plot_energy_spectrum(scale="log")
```

Scalar values are treated as size-1 phase-space axes. You can also write a single trajectory without a probability solve through `data.trace`:

```python
single = data.trace.backward(
    x=20.0, y=32.0, z=40.0,
    vx=data.unit.v.trans(1.0e5),
    vy=0.0,
    vz=data.unit.v.trans(-2.0e5),
    ispec=0,
    get_trace=True,
    get_probabilities=False,
)

single.plot_traces("t", "x")
single.traces.xvz.plot()
```

## Workflow API: `data.trace`

`data.trace.backward()` / `data.trace.forward()` / `data.trace.both()` always return a `TraceResult`. Payloads that were not requested are stored as `None`.

### Get Probabilities and Traces Together

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

`trace.plot()` dispatches to `trace.probabilities.pair(...).plot()` when probabilities are available, otherwise to `trace.plot_traces()`. Pass `kind="probability"` / `kind="trace"` when you want to be explicit.

### Get Traces Only

Set `get_probabilities=False` to skip the probability solve, create only particles from the phase-space grid, and return trajectories. In that case `trace.probabilities` and `trace.alpha` are `None`, and `plot_traces()` uses a uniform alpha unless you pass one explicitly.

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

### Get Backward and Forward Together

`both()` computes backward and forward trajectories from the same phase-space grid. If probabilities are requested, the probability solve runs only once.

```python
trace = data.trace.both(..., get_trace=True)
trace.backward_traces.xz.plot(alpha=trace.alpha)
trace.forward_traces.xz.plot(alpha=trace.alpha)
trace.plot_traces("x", "z", direction="backward")
trace.plot_traces("x", "z", direction="forward")
```

### Overlay in 3D

`plot3d()` returns a PyVista plotter. Pass an existing plotter to overlay traces on a field or boundary view.

```python
plotter = data.phisp[-1].plot3d(mode="slice", show=False)
trace.plot3d(plotter=plotter, direction="forward", tube_radius=0.05, show=True)
```

## Plotting Results

### 2-D Heatmap Projections

`trace.probabilities.pair(var1, var2)` integrates out the four unselected axes with the trapezoidal rule and returns `HeatmapData`. `trace.plot(var1, var2)` is the shorter entry point for the same operation.

```python
trace.plot("vx", "vz", cmap="viridis")
trace.probabilities.xvx.plot()
trace.probabilities.yz.plot(cmap="plasma")
```

`HeatmapData.plot()` draws a `pcolormesh` with a colour bar and SI-unit labels (`use_si=False` keeps grid units). Extra keyword arguments are forwarded straight to `pcolormesh`, so you can use `vmin` / `vmax` or `norm=LogNorm(...)` to control the colour scale.

### Energy Spectrum

Energy spectra are plotted from the probability payload's `ProbabilityResult`.

```python
trace.probabilities.plot_energy_spectrum(scale="log", energy_bins=80)
hist, bin_edges = trace.probabilities.energy_spectrum(energy_bins=80)
```

Internally it reads `wp` (or the photoelectron settings `path` / `curf` when `nflag_emit == 2`) from `plasma.inp` to compute a reference number density `n0`, weights each phase-space point by its probability, and integrates.

## Remote Execution Integration

`data.trace` shares the `Emout` facade's `remote_open_kwargs`, so if an emout server is running the computation runs on the worker by default and you get back a `RemoteTraceResult` proxy. Because the result is cached on the worker, changing visualisation parameters does **not** trigger recomputation.

```python
from emout.distributed import remote_figure

trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    get_trace=True,
)

with remote_figure():
    trace.plot("vx", "vz", cmap="viridis")

with remote_figure():
    trace.plot_traces("x", "z")

trace.drop()   # free worker memory when done
```

If you prefer the explicit remote style, switch to `data.remote().trace...`:

```python
from emout.distributed import remote_scope, remote_figure

with remote_scope():
    rdata = data.remote()
    trace = rdata.trace.forward(
        x=20.0, y=32.0, z=40.0,
        vx=vx_scan,
        vy=0.0,
        vz=vz_scan,
        get_trace=True,
    )

    with remote_figure():
        trace.plot("vx", "vz")
        trace.plot_traces("x", "z")
```

For the remote-execution mechanics, environment variables, and server management, see the [remote execution guide](distributed.md).

### `fetch()` for Local Customisation

When you want full matplotlib control (custom annotations, shared colour bars, dropping the heatmap into your own subplot grid), use `fetch()` to pull the small result arrays back to the client:

```python
local_trace = trace.fetch()
heatmap = local_trace.probabilities.vxvz
fig, ax = plt.subplots()
heatmap.plot(ax=ax, cmap="plasma")
ax.axhline(y=0, color="red", linestyle="--")
```

<details>
<summary>For existing code: show the lower-level `data.backtrace` API</summary>

## Low-Level API: `data.backtrace`

`data.backtrace` is the `BacktraceWrapper` used internally by `data.trace`. Use it when you need one trajectory from explicit `position` / `velocity` inputs, when you want to pass `vdsolverf.core.Particle` objects directly, or when you need direct control over the signed `dt` passed to the solver. Prefer `data.trace` for new phase-space workflows.

### Single Particle: `get_backtrace`

```python
position = (20.0, 32.0, 40.0)
velocity = (
    data.unit.v.trans(1.0e5),
    0.0,
    data.unit.v.trans(-2.0e5),
)

bt = data.backtrace.get_backtrace(position, velocity, ispec=0, max_step=50000)

bt.tx.plot()                 # = bt.pair("t", "x")
bt.xvz.plot()                # = bt.pair("x", "vz")
bt.yz.plot(color="black")    # yz projection of the trajectory
```

`bt.ts`, `bt.probability`, `bt.positions`, and `bt.velocities` are EMSES-unit arrays. `XYData.plot()` converts to SI units by default and auto-generates axis labels (`use_si=False` keeps EMSES units).

### Many Particles: `get_backtraces`

```python
import numpy as np

positions = np.array([[20, 32, 40], [21, 32, 40], [22, 32, 40]], dtype=float)
velocities = np.zeros_like(positions)
velocities[:, 0] = data.unit.v.trans(1.0e5)

many = data.backtrace.get_backtraces(positions, velocities, ispec=0)
many.xz.plot(alpha=0.5)
many.sample(50, random_state=0).tvx.plot()
```

`positions` and `velocities` are paired `(N, 3)` arrays. Use `data.trace` when you want the Cartesian product of a phase-space grid.

### Feeding Raw Particle Objects

```python
from vdsolverf.core import Particle

particles = [Particle(p, v) for p, v in zip(positions, velocities)]
many = data.backtrace.get_backtraces_from_particles(particles, ispec=0)
```

A common pattern is to chain this with `ProbabilityResult` particles:

```python
result = data.backtrace.get_probabilities(...)
bt = data.backtrace.get_backtraces_from_particles(result.particles, ispec=0)
bt.xz.plot(alpha=np.clip(result.probabilities, 0, 1))
```

### Arrival Probability: `get_probabilities`

`data.trace` internally calls `data.backtrace.get_probabilities(...)` and stores the `ProbabilityResult` as `trace.probabilities`. If you call the lower-level API directly, use this form:

```python
result = data.backtrace.get_probabilities(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    ispec=0,
    max_step=10000,
    n_threads=8,
)

result.vxvz.plot(cmap="viridis")
result.plot_energy_spectrum(scale="log")
```

### MPI Backend

`parallel="mpi"` / `parallel="srun"` are lower-level `get_probabilities()` backend options. You can also pass them through `data.trace` as `**kwargs`.

```python
trace = data.trace.forward(
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

</details>

## Related Classes

See the API reference (the `emout.core.backtrace` package) for full signatures.

- `TraceWrapper` — the `data.trace` object itself
- `TraceResult` — workflow result containing probability and trace payloads
- `BacktraceWrapper` — the `data.backtrace` object itself (lower-level API)
- `BacktraceResult` / `MultiBacktraceResult` — trajectory containers
- `ProbabilityResult` — 6-D probability grid and heatmap projections
- `XYData` / `MultiXYData` / `HeatmapData` — lightweight visualisation containers
