# Remote Execution (Dask) — Experimental

Remote execution lets emout read and render large EMSES simulation outputs on an HPC compute node while returning only small results to a login node or local Jupyter session. For new code, prefer the explicit `Emout.remote()` + `remote_scope()` + `remote_figure()` workflow.

## Minimal Workflow

### 1. Start the Server

Start the emout server once from a terminal. Choose the SLURM partition, memory, and walltime for your environment.

```bash
emout server start --partition gr20001a --memory 60G --walltime 03:00:00
```

The InfiniBand IP address and TLS credentials are configured automatically, and the active session is stored in `~/.emout/server.json`.

```text
Session: default
Scheduler running at tls://10.10.64.2:8786
Detected IP: 10.10.64.2
Workers: 1
```

### 2. Use It from Scripts

Call `Emout.remote()` to work with the worker-side `Emout`, and use `remote_scope()` to manage the lifetime of worker-side objects. Matplotlib commands inside `remote_figure()` are replayed on the worker, and only the PNG image is returned locally.

```python
import matplotlib.pyplot as plt
import emout
from emout.distributed import remote_figure, remote_scope

rdata = emout.Emout("output_dir").remote()

with remote_scope():
    ymid = int(rdata.inp.ny // 2)
    with remote_figure():
        rdata.phisp[-1, :, ymid, :].plot()
        plt.xlabel("x [m]")
        plt.title("Potential")
```

### 3. Stop the Server

Stop the server when you are done.

```bash
emout server stop
```

Named sessions can be stopped with `emout server stop --name <session>`. Use `emout server stop --all` to stop every saved session.

## Choosing a Mode

| Goal | Recommended mode | What returns locally |
| --- | --- | --- |
| View figures without reading large fields locally | `Emout.remote()` + `remote_scope()` + `remote_figure()` | PNG/SVG image |
| Reuse worker-side objects repeatedly | `Emout.remote()` + `remote_scope()` | `RemoteRef` / dedicated proxy |
| Keep existing `plot()` code with minimal edits | compatibility mode | Small arrays such as 2-D slices |
| Customise freely with local matplotlib | `fetch()` small results | Local arrays / lightweight data objects |

### Recommended Mode (`Emout.remote()` + `remote_scope()`)

Worker-side objects are held as `RemoteRef` instances while you write code close to local `emout` / `numpy` style. Expressions such as `-ref`, `ref1 + ref2`, `np.abs(ref)`, and `int(ref)` stay remote until you explicitly fetch a result.

```python
import numpy as np
from emout.distributed import remote_scope

rdata = emout.Emout("output_dir").remote()

with remote_scope():
    phi = rdata.phisp[-1, :, 100, :]
    ex = -rdata.exz[-1, :, 100, :]
    peak = np.abs(ex).max()
    small = phi.fetch()  # pull data local only when needed
```

Remote objects created inside `remote_scope()` are dropped automatically when the `with` block exits. You can reuse intermediate results inside the block while leaving worker memory management to the scope.

### Image Mode (`remote_figure`)

`remote_figure()` replays matplotlib commands on the worker and returns only an image. This is the default choice when you do not want large field arrays on a login node.

```python
from emout.distributed import remote_figure

with remote_figure(figsize=(8, 5), dpi=200):
    rdata.phisp[-1, :, 100, :].plot()
    plt.axhline(y=50, color="red")
    plt.title("remote rendered figure")
```

Pass `savefilepath` to save the generated figure directly from CLI or batch runs. The extension is used to infer the output format when possible.

```python
with remote_figure(savefilepath="figure.png"):
    rdata.phisp[-1, :, 100, :].plot()
```

### Data-Transfer Mode (Compatibility Mode)

When an active/default session is saved, existing `plot()` code can still run in compatibility mode. The worker extracts a small slice and sends it back, while matplotlib runs locally.

```python
data = emout.Emout("output_dir")
data.phisp[-1, :, 100, :].plot()
plt.title("local matplotlib")
plt.savefig("output.png")
```

Only sliced arrays are transferred. For new code, prefer `Emout.remote()` / `remote_scope()` because worker-side object lifetime is explicit.

## Feature Integrations

### Backtrace / `data.trace`

Heavy particle backtrace workflows can run once on the server and remain cached in worker memory. For new analysis code, prefer the high-level `data.trace...` / `data.remote().trace...` workflow.

```python
with remote_scope():
    rdata = data.remote()

    trace = rdata.trace.forward(
        x=20.0, y=32.0, z=40.0,
        vx=vx_scan,
        vy=0.0,
        vz=vz_scan,
        ispec=0,
        get_trace=True,
        get_probabilities=True,
    )

    with remote_figure():
        trace.plot("vx", "vz")
        trace.plot_traces("x", "z")

    trace.drop()
```

The lower-level `data.backtrace...` / `data.remote().backtrace...` route remains available for existing code. For the backtrace API itself (`TraceResult`, `ProbabilityResult`, `BacktraceResult`, and related containers), see the [backtrace guide](backtrace.md).

#### Local Customisation with `fetch()`

When you want full matplotlib control (custom annotations, shared colour bars, and similar), use `fetch()` to pull the small result object back to the client.

```python
local_trace = trace.fetch()
heatmap = local_trace.probabilities.vxvz
fig, ax = plt.subplots()
heatmap.plot(ax=ax, cmap="plasma")
ax.axhline(y=0, color="red", linestyle="--")
```

### Boundary Meshes

Boundary shapes are lightweight, so `data.boundaries.plot()` can run locally. When you overlay them on a 3-D field, combine them with remote execution so the 3-D field is not read on the login node.

```python
data.boundaries.plot()

with remote_scope():
    rdata = data.remote()
    with remote_figure():
        rdata.phisp[-1].plot_surfaces(surfaces=data.boundaries)
```

See the [boundary meshes guide](boundaries.md) for details.

### Animations (`gifplot`)

`gifplot()` can also run on the worker. Frame generation and encoding happen on the worker, while the client receives inline HTML, GIF bytes, or a saved file.

```python
with remote_scope():
    rdata = emout.Emout("output_dir").remote()
    rdata.phisp[:, 100, :, :].gifplot()                                  # inline HTML
    rdata.phisp[:, 100, :, :].gifplot(action="save", filename="out.gif")  # save to shared FS
    gif = rdata.phisp[:, 100, :, :].gifplot(action="bytes")               # receive bytes
```

See the remote execution section of the [animations guide](animation.md) for details.

## Safety and Lifecycle Details

### Disable Local Field Reads

If you want to avoid accidentally materialising large field arrays on a login node, disable local field data access.

```python
import emout

emout.disable_local_data_access()

data = emout.Emout("output_dir")
data.phisp[-1].materialize()  # LocalDataAccessDisabledError
```

This setting also affects existing `Emout` instances. Instances created with `Emout(..., local_data_policy="allow")` are explicitly allowed to read local data.

```python
small = emout.Emout("small_output", local_data_policy="allow")
```

Use the context manager when you only need a temporary exception.

```python
with emout.local_data_policy("allow"):
    small_slice = data.phisp[-1, :10, :10, :10]
```

Use an environment variable to enforce the policy for a whole shell.

```bash
export EMOUT_LOCAL_DATA_POLICY=remote_required
```

Metadata such as `data.phisp.shape` and `data.phisp.grid_shape` is still available under `remote_required`. Analyses that read actual field data should run through `data.remote()` / `RemoteRef`.

### `remote_scope()` Lifecycle

`remote_scope()` groups worker-side objects so they can be dropped together. For short workflows, `with remote_scope():` is the normal form.

#### `open()` / `close()` — Explicit Jupyter Form

When you do not want to indent an entire Jupyter cell, call `open()` / `close()` directly.

```python
from emout.distributed import remote_scope

scope = remote_scope()
scope.open()

rdata = data.remote()
ref = rdata.phisp[-1, :, 100, :]
ref.plot()

# ...continue using rdata / ref in another cell...

scope.close()   # drop every remote object registered to the scope
```

`close()` is safe to call multiple times; later calls are no-ops.

#### `clear()` — Manual GC without Closing the Scope

When a loop creates many intermediate refs, call `clear()` to release accumulated refs without closing the scope.

```python
scope = remote_scope()
scope.open()
rdata = data.remote()

for t in range(100):
    ref = rdata.phisp[t, :, 100, :]
    arr = ref.fetch()
    # ... work ...
    scope.clear()

scope.close()
```

#### Nested Scopes

`remote_scope` behaves like a stack. Newly created refs are always registered to the innermost scope, so closing an inner scope drops only those refs while the outer scope stays active.

```python
scope1 = remote_scope()
scope1.open()

with remote_scope():
    ref_inner = rdata.phisp[-1, :, 100, :]

ref_outer = rdata.exz[-1]
scope1.close()
```

> **Pitfall:** Do not use the same scope instance with both `open()` and `with scope:`. Exiting the `with` block closes that scope, so later refs may be left untracked. If you need nesting, create a new `remote_scope()` for the inner block.

### `remote_figure()` Variants

#### Receiving a FigureProxy via `as fig`

`remote_figure(...)` yields the worker-side `Figure` as a `FigureProxy`, so you can receive it with `as fig` and call methods such as `fig.add_axes(...)`.

```python
with remote_figure(figsize=(13, 6), dpi=300) as fig:
    ax = fig.add_axes([0.13, 0.11, 0.57, 0.78], projection="3d")
    cax = fig.add_axes([0.74, 0.12, 0.025, 0.76])
    rdata.phisp[-1].plot_surfaces(ax=ax, surfaces=data.boundaries)
    ax.view_init(elev=36, azim=-110)
    plt.colorbar(cax=cax, label=r"$\phi$ (V)")
```

#### `open()` / `close()` Style

If adding a `with` block to existing code is awkward, use `RemoteFigure.open()` / `close()`.

```python
from emout.distributed import RemoteFigure

rf = RemoteFigure()
rf.open()
rdata.phisp[-1, :, 100, :].plot()
plt.xlabel("x [m]")
rf.close()
```

Forgetting `close()` leaves matplotlib monkey-patched, so prefer `with remote_figure():` when possible.

#### Jupyter Cell Magic (`%%remote_figure`)

Register the magic once per session, then put `%%remote_figure` at the top of a cell.

```python
%load_ext emout.distributed.remote_figure
# or: from emout.distributed import register_magics; register_magics()
```

```python
%%remote_figure --dpi 300 --fmt svg --figsize 12,6
rdata.phisp[-1, :, 100, :].plot()
plt.xlabel("x [m]")
```

| Option | Short | Meaning | Default |
| --- | --- | --- | --- |
| `--dpi` | `-d` | Output resolution | `150` |
| `--fmt` | `-f` | Image format (`png`, `svg`, ...) | `png` |
| `--figsize` | | `width,height` | Matplotlib default |
| `--emout-dir` | | Directory used to find the session | automatic |

## How It Works

```text
Login node (Jupyter)              Compute node (SLURM worker)

emout server start          ->    Scheduler + Worker starts
                                  <-> InfiniBand communication
rdata = emout.Emout("dir").remote()
with remote_scope():
    with remote_figure():
        rdata.phisp[-1,:,100,:].plot()  ->  HDF5 read + rendering on worker
                                    <-  PNG bytes only
```

A normal local `plot()` reads a slice from HDF5 into the local Python process and then draws it with matplotlib. `remote_figure()` keeps both the HDF5 read and rendering on the worker, returning only an image.

### Shared Session Architecture

One `RemoteSession` Dask Actor manages all Emout instances on one worker. When you access a different simulation, the session lazily loads that `Emout` on first use and caches it for later calls.

This means results from different simulations can be freely mixed inside the same `remote_figure()` block.

```python
data_a = emout.Emout("/path/to/sim_a").remote()
data_b = emout.Emout("/path/to/sim_b").remote()

with remote_scope():
    with remote_figure(figsize=(12, 5)):
        plt.subplot(1, 2, 1)
        data_a.phisp[-1, :, 100, :].plot()
        plt.title("Sim A")

        plt.subplot(1, 2, 2)
        data_b.phisp[-1, :, 100, :].plot()
        plt.title("Sim B")
```

If the worker job is cancelled with `scancel` or expires at walltime, the next `emout server start` or auto-connect treats it as a stale session and cleans up saved state. Compatibility mode falls back to local execution, while explicit remote execution raises an error asking you to restart the server.

## Explicit Connection

Use `connect()` when you want manual control instead of auto-connect.

```python
from emout.distributed import connect

client = connect()                                         # active/default session
client = connect(name="batch2")                            # named session
client = connect("tls://10.10.64.2:8786", name="batch2")   # address + saved credentials
```

Start additional named sessions when you need them.

```bash
emout server start --allow-multiple --name batch2 --memory 120G
emout server status --all
emout server stop --name batch2
```

## Environment Variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `EMOUT_DASK_SCHED_IP` | Scheduler IP override | InfiniBand auto-detection |
| `EMOUT_DASK_SCHED_PORT` | Scheduler port | `10000 + (UID % 50000)` |
| `EMOUT_DASK_PARTITION` | SLURM partition | `gr20001a` |
| `EMOUT_DASK_CORES` | Worker core count | `60` |
| `EMOUT_DASK_MEMORY` | Worker memory | `60G` |
| `EMOUT_DASK_WALLTIME` | Job walltime | `03:00:00` |

### Port Selection

The default scheduler port is `10000 + (UID % 50000)`, which gives each user a different port on the same login node (for example, UID 36291 -> port 46291). If that port is already in use, emout scans up to 20 consecutive ports. Set `EMOUT_DASK_SCHED_PORT` to override it manually.

## Limitations

- Python >= 3.10 must have `dask` and `distributed` installed.
- Every simulation directory must be visible from the worker node, usually through a shared file system.
- Worker memory increases with each loaded Emout instance. Use `drop()` and `remote_scope()` to release cached results in large campaigns.
- Interactive worker-side PyVista scenes are out of scope. See [PyVista visualization](pyvista.md) for PyVista-specific guidance.
