# Remote Execution (Dask) — Experimental

A remote execution framework that processes large simulation outputs on
HPC compute nodes and returns only plot images to the login node (your Jupyter).

## How It Works

```
Login node (Jupyter)                 Compute node (SLURM worker)

emout server start              →    Scheduler + Worker start
                                     ↕ InfiniBand high-speed comm
data = emout.Emout("dir")
data.phisp[-1,:,100,:].plot()   →    HDF5 load → 2D slice → transfer
                                ←    small array (few KB)
plt.xlabel("custom")                 ← local matplotlib rendering

with remote_figure():
    data.phisp[-1,:,100,:].plot()  → all operations on server
    plt.xlabel("custom")           → (recorded as commands)
                                   ← only PNG bytes (~50 KB)

with remote_figure(savefilepath="figure.png"):
    data.phisp[-1,:,100,:].plot()  → render on server
                                   ← save the image to a file
```

### Shared Session Architecture

A single `RemoteSession` Dask Actor manages all Emout instances on one
worker.  When you access data from different simulations, the session
lazily loads each `Emout` instance on first use and caches it for
subsequent calls.

This means **results from different simulations can be freely mixed** in
the same `remote_figure()` block:

```python
data_a = emout.Emout("/path/to/sim_a")
data_b = emout.Emout("/path/to/sim_b")

result_a = data_a.backtrace.get_probabilities(...)
result_b = data_b.backtrace.get_probabilities(...)

with remote_figure(figsize=(12, 5)):
    plt.subplot(1, 2, 1)
    data_a.phisp[-1, :, 100, :].plot()
    plt.title("Sim A: potential")

    plt.subplot(1, 2, 2)
    result_b.vxvz.plot(cmap="plasma")
    plt.title("Sim B: backtrace")
```

All commands are replayed on the same worker — no data is transferred to the client.
If `savefilepath` is provided, the rendered image can be saved directly in
CLI / batch workflows. When the path has an extension, the output format
is inferred from it.

## Setup

On Python 3.10+, `pip install emout` automatically includes Dask and the
TLS dependency used by `emout server`. No extra setup is needed.

### 1. Start the server (once, in a terminal)

```bash
emout server start --partition gr20001a --memory 60G --walltime 03:00:00
```

The InfiniBand IP is auto-detected. `emout` also generates per-user TLS
credentials automatically, stores them with user-only permissions, and
mirrors the active session to `~/.emout/server.json`.

```
Session: default
Scheduler running at tls://10.10.64.2:8786
Detected IP: 10.10.64.2
Workers: 1
```

By default, only one active server session is allowed per user. To run
an additional session intentionally, give it a name:

```bash
emout server start --allow-multiple --name batch2 --memory 120G
emout server status --all
emout server stop --name batch2
```

### 2. Use from scripts

With the active session saved, existing code still works through the
compatibility mode. The compat mode always follows the active/default
session. For new code, prefer the explicit `Emout.remote()` workflow:

```python
import emout
from emout.distributed import remote_figure, remote_scope

data = emout.Emout("output_dir").remote()

with remote_scope():
    ymid = int(data.inp.ny // 2)
    with remote_figure():
        data.phisp[-1, :, ymid, :].plot()
```

### 3. Stop the server

```bash
emout server stop
```

Additional named sessions can be stopped with `emout server stop --name <session>`
or all at once with `emout server stop --all`.

If a worker job is cancelled with `scancel` or disappears after walltime
timeout, the next `emout server start` / auto-connect treats that session
as stale and clears the saved state automatically. Remote execution fails
fast instead of waiting forever: compatibility mode falls back to local
execution, while explicit remote usage asks you to restart the server.

## Usage Modes

### Recommended mode (`Emout.remote()`)

This keeps worker-side objects alive as `RemoteRef` proxies while letting
you write code close to normal `emout` / `numpy` style. Expressions such as
`-ref`, `ref1 + ref2`, `np.abs(ref)`, and `int(ref)` stay remote until you
explicitly fetch them.

```python
import matplotlib.pyplot as plt
import emout
from emout.distributed import remote_figure, remote_scope

rdata = emout.Emout("output_dir").remote()

with remote_scope():
    ymid = int(rdata.inp.ny // 2)

    with remote_figure():
        plt.figure(figsize=(18, 16))
        rdata.phisp[-1, 180:400, ymid, :].plot()
        (-rdata.exz[-1, 180:400, ymid, :]).plot()
        plt.title("remote expression example")
```

Objects created inside `remote_scope()` are automatically `drop()`-ed when
the context exits, so you can reuse intermediate remote results many times
within the block without having to manage worker-side cleanup yourself.

#### `open()` / `close()` — explicit form for Jupyter

If you do not want to indent a whole cell under `with`, call `open()`
and `close()` directly. The scope survives across cells, so you can
keep `rdata` and every registered ref alive for as long as you need:

```python
from emout.distributed import remote_scope

scope = remote_scope()
scope.open()

rdata = data.remote()
ref = rdata.phisp[-1, :, 100, :]
ref.plot()

# ...continue working in other cells with rdata / ref...

scope.close()   # drops every registered ref in one go
```

`close()` is idempotent, so you rarely need a `try/finally` — calling
it twice is a no-op.

#### `clear()` — manual GC while the scope stays open

In loops that create many intermediate refs, `clear()` drops every
registered ref **without** leaving the scope:

```python
scope = remote_scope()
scope.open()
rdata = data.remote()

for t in range(100):
    ref = rdata.phisp[t, :, 100, :]
    arr = ref.fetch()
    # ... work with arr ...
    scope.clear()   # release this iteration's refs, keep the scope

scope.close()
```

After `clear()` the scope is still active, so refs created afterwards
continue to be tracked by the same scope. This is the tool of choice
for long-running sessions where you would otherwise see worker memory
grow monotonically.

#### Nesting scopes

`remote_scope` behaves like a stack. You can open an outer scope and
create another one inside it; every new ref is registered to **the
innermost active scope**, so closing the inner scope drops only its
refs and leaves the outer scope running:

```python
# open/open/close/close
scope1 = remote_scope()
scope1.open()

scope2 = remote_scope()
scope2.open()

ref_inner = rdata.phisp[-1, :, 100, :]   # tracked by scope2
scope2.close()                             # drops ref_inner only

ref_outer = rdata.exz[-1]                  # tracked by scope1
scope1.close()                             # drops ref_outer
```

Mixing explicit `open()` with a ``with`` block nests cleanly too:

```python
scope1 = remote_scope()
scope1.open()

with remote_scope() as scope2:
    ref_inner = rdata.phisp[-1, :, 100, :]   # tracked by scope2
# scope2 auto-drops here; scope1 is still open

ref_outer = rdata.exz[-1]                     # tracked by scope1
scope1.close()
```

> **Foot-gun: never use the same scope instance with both ``open()`` and
> a ``with`` block.** The snippet below looks fine but breaks — ``with
> scope:`` calls ``__exit__`` on the instance, so ``scope`` is already
> closed by the time the block returns. Subsequent refs are tracked by
> **nothing**, and ``scope.close()`` becomes a no-op:
>
> ```python
> scope = remote_scope()
> scope.open()
> with scope:                    # ← do NOT hand the same scope to `with`
>     ref = rdata.phisp[-1]
> # scope is already closed here
> leaked = rdata.phisp[-2]      # ← not tracked by any scope!
> scope.close()                  # ← no-op
> ```
>
> When you need both styles, open **a new** ``remote_scope()`` inside
> the outer one (see the example above with ``with remote_scope() as scope2:``).

### Data-transfer mode (compatibility mode)

This is the compatibility mode for existing `plot()`-centric code.
The worker extracts the slice and transfers it locally; matplotlib runs on the client.
**`plt.axhline()` and other customizations work freely.**

```python
data.phisp[-1, :, 100, :].plot()
plt.axhline(y=50, color="red")    # ← local matplotlib
plt.xlabel("x [m]")
plt.title("Custom title")
plt.savefig("output.png")
```

Only a 2D slice (KB–MB) is transferred; the full 3D array stays on the worker.

### Image mode (`remote_figure`)

**All matplotlib operations run on the server**; only PNG bytes come back.
Use when you want minimal local memory usage.

```python
from emout.distributed import remote_figure

with remote_figure():
    data.phisp[-1, :, 100, :].plot()
    plt.axhline(y=50, color="red")    # ← runs on server
    plt.xlabel("x [m]")
    plt.title("Custom title")
# ← PNG displayed in Jupyter here
```

#### `open()` / `close()` style

When adding `with` blocks to existing code is cumbersome, use `RemoteFigure`
with explicit `open()` / `close()`:

```python
from emout.distributed import RemoteFigure

rf = RemoteFigure()
rf.open()
data.phisp[-1, :, 100, :].plot()
plt.xlabel("x [m]")
rf.close()   # ← commands replayed on server, PNG displayed
```

`RemoteFigure` also works as a context manager (`with RemoteFigure() as rf: ...`).

> **Note:** If you forget to call `close()`, matplotlib stays monkey-patched
> and a `ResourceWarning` is emitted at garbage collection.

#### Jupyter cell magic (`%%remote_figure`)

Register the magic once per session, then use `%%remote_figure` at the top
of any cell:

```python
# Register (once)
%load_ext emout.distributed.remote_figure
# or: from emout.distributed import register_magics; register_magics()
```

```python
%%remote_figure
data.phisp[-1, :, 100, :].plot()
plt.xlabel("x [m]")
```

Options can be passed on the magic line:

```python
%%remote_figure --dpi 300 --fmt svg --figsize 12,6
data.phisp[-1, :, 100, :].plot()
```

| Option | Short | Description | Default |
| --- | --- | --- | --- |
| `--dpi` | `-d` | Output resolution | `150` |
| `--fmt` | `-f` | Image format (`png`, `svg`, …) | `png` |
| `--figsize` | | `width,height` | matplotlib default |
| `--emout-dir` | | Emout directory for session lookup | auto |

### Backtrace integration

Heavy particle-backtrace computations run once on the server; the result
stays in worker memory. Re-render with different visualisation parameters
without recomputing.

```python
# Computation (runs on server, result cached in worker memory)
result = data.backtrace.get_probabilities(
    x, y, z, vx_range, vy_center, vz_range, ispec=0,
)

# Visualise repeatedly using the same result (no recomputation)
with remote_figure():
    result.vxvz.plot(cmap="viridis")
    plt.title("Velocity distribution (vx-vz)")

with remote_figure():
    result.plot_energy_spectrum(scale="log")
    plt.xlabel("Energy [eV]")

# Free worker memory when done
result.drop()
```

Both `data.backtrace...` and `data.remote().backtrace...` return the same
dedicated proxies (`RemoteProbabilityResult` / `RemoteBacktraceResult`).
Use the former when you want to keep existing code almost unchanged, and
the latter when you want one explicit-remote workflow across fields,
boundaries, and backtrace results:

```python
with remote_scope():
    rdata = data.remote()

    bt = rdata.backtrace.get_backtrace(position, velocity, ispec=0)
    result = rdata.backtrace.get_probabilities(
        x, y, z, vx_range, vy_center, vz_range, ispec=0,
    )

    with remote_figure():
        bt.tx.plot()
        result.vxvz.plot(cmap="viridis")
```

For the backtrace API itself (`BacktraceResult` / `MultiBacktraceResult` /
`ProbabilityResult`, shorthand attribute access, axis lists), see the
dedicated [backtrace guide](backtrace.md).

#### Local customisation with fetch()

If you need full matplotlib control (e.g. custom annotations, shared colour bars),
use `fetch()` to pull the small result arrays back to the client:

```python
heatmap = result.vxvz.fetch()   # → local HeatmapData
fig, ax = plt.subplots()
heatmap.plot(ax=ax, cmap="plasma")
ax.axhline(y=0, color="red", linestyle="--")
ax.set_title("Custom annotation")
```

### Boundary meshes

```python
# Boundary shapes only (lightweight, always local)
data.boundaries.plot()

# Overlay on field (3D array slice-transferred from server)
data.phisp[-1].plot_surfaces(ax=ax, surfaces=data.boundaries)
ax.set_xlabel("x [m]")
```

### Animations (`gifplot`)

`gifplot()` runs end-to-end on the worker as well: frame generation and
encoding stay on the worker, and only the inline HTML or GIF bytes come
back to the client.

```python
rdata = emout.Emout("output_dir").remote()

with remote_scope():
    rdata.phisp[:, 100, :, :].gifplot()                                 # inline HTML
    rdata.phisp[:, 100, :, :].gifplot(action="save", filename="out.gif")  # shared FS path
    gif = rdata.phisp[:, 100, :, :].gifplot(action="bytes")             # raw bytes
```

See the "Remote execution" section of the
[animations guide](animation.md) for the full options.

## Explicit connection

To connect manually instead of auto-connecting:

```python
from emout.distributed import connect
client = connect()                                          # active/default session
client = connect(name="batch2")                             # additional named session
client = connect("tls://10.10.64.2:8786", name="batch2")    # explicit address + saved credentials
```

## Environment variables

| Variable | Description | Default |
| --- | --- | --- |
| `EMOUT_DASK_SCHED_IP` | Scheduler IP (overrides auto-detection) | InfiniBand auto |
| `EMOUT_DASK_SCHED_PORT` | Scheduler port | `10000 + (UID % 50000)` |
| `EMOUT_DASK_PARTITION` | SLURM partition | `gr20001a` |
| `EMOUT_DASK_CORES` | Worker cores | `60` |
| `EMOUT_DASK_MEMORY` | Worker memory | `60G` |
| `EMOUT_DASK_WALLTIME` | Job wall time | `03:00:00` |

### Port selection

The scheduler port defaults to `10000 + (UID % 50000)`, so each user on
the same login node gets a different port automatically (e.g. UID 36291
→ port 46291).  If that port is already in use, up to 20 consecutive
ports are probed until a free one is found.  Set
`EMOUT_DASK_SCHED_PORT` to override.

## Limitations

- Python >= 3.10 with `dask` and `distributed` installed.
- All simulation directories must be accessible from the worker node
  (shared filesystem required).
- Worker memory grows with each loaded Emout instance.  For very large
  campaigns, call `result.drop()` to free cached computation results.
