# Animations (`gifplot`)

`gifplot()` creates time-series animations from multi-dimensional data. It is the second most frequently used feature after `plot()`.

## Basic Usage

```python
import emout

data = emout.Emout("output_dir")

# Inline display in Jupyter Notebook (default action='to_html')
data.phisp[:, 100, :, :].gifplot()
```

The slice `[:, 100, :, :]` selects all timesteps at z=100, producing an animation of the xy-plane over time.

## Output Actions

The `action` parameter controls what happens after frames are generated:

| Action | Description |
| --- | --- |
| `'to_html'` | Return HTML for inline Jupyter display (default) |
| `'show'` | Show in a matplotlib window |
| `'save'` | Save to a file (requires `filename`) |
| `'return'` | Return `(fig, animation)` for manual control |
| `'frames'` | Return a `FrameUpdater` for multi-panel layouts |

```python
# Save as GIF
data.phisp[:, 100, :, :].gifplot(action="save", filename="phisp.gif")

# Show in matplotlib window
data.phisp[:, 100, :, :].gifplot(action="show")
```

## Common Options

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| `action` | `str` | Output mode (see table above) | `'to_html'` |
| `filename` | `str` | Save path for `action='save'` | `None` |
| `axis` | `int` | Axis to animate over | `0` |
| `interval` | `int` | Frame interval in milliseconds | `200` |
| `repeat` | `bool` | Loop the animation | `True` |
| `use_si` | `bool` | Use SI unit labels | `True` |
| `vmin` | `float` | Minimum colorbar value | auto |
| `vmax` | `float` | Maximum colorbar value | auto |
| `norm` | `str` | `'log'` for logarithmic scale | `None` |
| `mode` | `str` | Plot mode (`'cmap'`, `'cont'`, `'stream'`) | auto |

## Multi-Panel Animations

Combine multiple data sources into a single animation with a grid layout:

```python
# Step 1: Create frame updaters
updater0 = data.phisp[:, 100, :, :].gifplot(action="frames", mode="cmap")
updater1 = data.phisp[:, 100, :, :].build_frame_updater(mode="cont")
updater2 = data.nd1p[:, 100, :, :].build_frame_updater(
    mode="cmap", vmin=1e-3, vmax=20, norm="log"
)
updater3 = data.nd2p[:, 100, :, :].build_frame_updater(
    mode="cmap", vmin=1e-3, vmax=20, norm="log"
)
updater4 = data.j2xy[:, 100, :, :].build_frame_updater(mode="stream")

# Step 2: Define layout as a triple-nested list [row][col][overlay]
layout = [
    [
        [updater0, updater1],   # Row 0, Col 0: colormap + contour overlay
        [updater2],             # Row 0, Col 1: density (log scale)
        [updater3, updater4],   # Row 0, Col 2: density + streamlines overlay
    ]
]

# Step 3: Create animator and display
animator = updater0.to_animator(layout=layout)
animator.plot(action="to_html")
```

### Layout Structure

The layout is a **3-level nested list**:

- **Level 1 (outer):** Rows
- **Level 2:** Columns within a row
- **Level 3 (inner):** Overlaid updaters sharing the same subplot

Each updater in the innermost list draws on the same axes, allowing overlay of different visualization modes (e.g., colormap + contour, density + streamlines).

### `build_frame_updater` vs `gifplot(action='frames')`

Both create a `FrameUpdater` object. The difference:

- `gifplot(action='frames')` is a convenience shorthand
- `build_frame_updater()` gives you explicit control over `mode`, `vmin`, `vmax`, etc.

Use `build_frame_updater()` when you need per-panel customization.

## Remote execution (`Emout.remote()`)

`gifplot()` works through `data.remote()` as well. The whole animation is
rendered on the worker and only the inline HTML or GIF bytes are
shipped back to the client.

```python
import emout
from emout.distributed import remote_scope

rdata = emout.Emout("output_dir").remote()

with remote_scope():
    # Inline display in Jupyter (the worker produces the HTML)
    rdata.phisp[:, 100, :, :].gifplot()

    # Save the GIF directly to a shared-filesystem path
    rdata.phisp[:, 100, :, :].gifplot(action="save", filename="/scratch/you/phisp.gif")

    # If the worker's filesystem is not shared with the client, grab the bytes
    gif_bytes = rdata.phisp[:, 100, :, :].gifplot(action="bytes")
    from pathlib import Path
    Path("phisp.gif").write_bytes(gif_bytes)
```

Remote mode only supports `action="to_html"`, `"save"`, and `"bytes"`.
`"show"`, `"return"`, and `"frames"` are not meaningful on a headless
worker (or are not picklable), so they raise `ValueError`. Long or
high-resolution animations can produce HTML strings of tens of
megabytes (`ani.to_jshtml()` base64-encodes every frame), so prefer
`action="bytes"` or `action="save"` with a shared-filesystem path for
those.
