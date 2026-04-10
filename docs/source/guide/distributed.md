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

## Setup

On Python 3.10+, `pip install emout` automatically includes Dask. No extra install step is needed.

### 1. Start the server (once, in a terminal)

```bash
emout server start --partition gr20001a --memory 60G --walltime 03:00:00
```

The InfiniBand IP is auto-detected and saved to `~/.emout/server.json`.

```
Scheduler running at tcp://10.10.64.2:8786
Detected IP: 10.10.64.2
Workers: 1
```

### 2. Use from scripts

With `server.json` present, **no code changes are needed**:

```python
import emout

data = emout.Emout("output_dir")

# ↓ automatically remote if server is running; local otherwise
data.phisp[-1, :, 100, :].plot()
```

### 3. Stop the server

```bash
emout server stop
```

## Usage Modes

### Data-transfer mode (default)

The default when not using `with remote_figure()`.
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

Heavy computation runs once on the server; results stay in server memory.
Re-render with different visualization parameters without recomputation.

```python
# Computation (runs on server, result cached in server memory)
result = data.backtrace.get_probabilities(
    x, y, z, vx_range, vy_center, vz_range, ispec=0,
)

# Visualize repeatedly (no recomputation)
with remote_figure():
    result.vxvz.plot(cmap="viridis")
    plt.title("Velocity distribution (vx-vz)")

with remote_figure():
    result.vxvy.plot(cmap="plasma")

with remote_figure():
    result.plot_energy_spectrum(scale="log")
    plt.xlabel("Energy [eV]")

# Free server memory when done
result.drop()
```

#### Local customisation with fetch()

If you need full matplotlib control (e.g. custom annotations, shared colour bars),
use `fetch()` to transfer the small result arrays to the client:

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

## Explicit connection

To connect manually instead of auto-connecting:

```python
from emout.distributed import connect
client = connect()                         # auto-detect from ~/.emout/server.json
client = connect("tcp://10.10.64.2:8786")  # explicit address
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

### Limitations

- Python >= 3.10 with `dask` and `distributed` installed.
- All simulation directories must be accessible from the worker node
  (shared filesystem required).
- Worker memory grows with each loaded Emout instance.  For very large
  campaigns, call `result.drop()` to free cached computation results.
