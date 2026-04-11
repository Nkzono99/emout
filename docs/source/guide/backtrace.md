# Backtrace (`data.backtrace`) — Experimental

`data.backtrace` is the entry point for integrating particle trajectories
**backwards in time** through EMSES fields. You can compute arrival
probabilities (`get_probabilities`) or trajectories for individual
particles (`get_backtrace` / `get_backtraces`). Results come back wrapped
in dedicated containers that let you chain straight into visualization
with shorthand like `.vxvz.plot()`.

> **Requirements:** backtrace relies on the external
> [`vdist-solver-fortran`](https://github.com/Nkzono99/vdist-solver-fortran)
> package (`vdsolverf`). Install it with `pip install vdist-solver-fortran`.
> Without it, calls to `data.backtrace.*` raise `ImportError`.

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
velocity = (1.0e5, 0.0, -2.0e5)
bt = data.backtrace.get_backtrace(position, velocity, ispec=0)

bt.tx.plot()      # t vs x trajectory
bt.xvz.plot()     # x vs vz phase space

# Many particles in one call
import numpy as np
positions = np.array([[20, 32, 40], [21, 32, 40], [22, 32, 40]], dtype=float)
velocities = np.zeros_like(positions)
velocities[:, 0] = 1.0e5
many = data.backtrace.get_backtraces(positions, velocities, ispec=0)

many.xz.plot(alpha=0.5)    # overlay all trajectories

# Arrival probability over a 6-D phase-space grid
result = data.backtrace.get_probabilities(
    x=20.0, y=32.0, z=40.0,
    vx=(-3e5, 3e5, 64),
    vy=0.0,
    vz=(-3e5, 3e5, 64),
    ispec=0,
)

result.vxvz.plot(cmap="viridis")      # heatmap in the vx-vz plane
result.plot_energy_spectrum(scale="log")
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

Each axis accepts:

- a tuple `(start, stop, n)` — equally-spaced grid
- an explicit array or list — arbitrary values
- a scalar — a size-1 axis (automatically squeezed when you call `pair()`)

```python
result = data.backtrace.get_probabilities(
    x=20.0, y=32.0, z=40.0,     # fixed position
    vx=(-3e5, 3e5, 64),         # scan vx over 64 points
    vy=0.0,
    vz=(-3e5, 3e5, 64),         # scan vz over 64 points
    ispec=0,
    max_step=10000,
    n_threads=8,
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
