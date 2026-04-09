Lang: [English](distributed.md) | [日本語](distributed.ja.md)

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

## Setup

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
| `EMOUT_DASK_SCHED_PORT` | Scheduler port | `8786` |
| `EMOUT_DASK_PARTITION` | SLURM partition | `gr20001a` |
| `EMOUT_DASK_CORES` | Worker cores | `60` |
| `EMOUT_DASK_MEMORY` | Worker memory | `60G` |
| `EMOUT_DASK_WALLTIME` | Job wall time | `03:00:00` |

## Limitations

- Requires Python ≥ 3.10 (`dask` / `distributed` dependency)
- Inside `remote_figure()`, matplotlib return values (`AxesImage` etc.) are not available
- `plot3d()` (PyVista) remote execution is not yet supported
