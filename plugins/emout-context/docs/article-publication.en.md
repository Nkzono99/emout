Lang: [English](article-publication.en.md) | [日本語](article-publication.md)

# Article Data Publication Workflow

This document summarizes the standard guidance for emout article record/replay workflows used for paper data publication, supplemental material, and reproducible figure bundles.

## Basic Policy

- Keep the normal visualization script mostly unchanged, and switch `normal` / `record` / `replay` with `EMOUT_ARTICLE_MODE` or `Emout(..., article_mode=...)`.
- Publish the minimum grid data used by the figure, not the original full HDF5 output.
- For 2D plots, slice first, such as `data.phisp[-1, :, ymid, :]`.
- For 3D `plot_surfaces()`, pass `bounds` so the recorded data is limited to the plotted ROI.
- For time averages, use `data.phisp[-20:].mean()`. emout records the averaged data, not every source timestep used in the average.

## Recommended Code

```python
import emout

data = emout.Emout("output_dir", article_records_path="article-records")

# 2D slice: only this 2D slice is recorded
ymid = data.inp.ny // 2
data.phisp[-1, :, ymid, :].plot()

# Time-mean 3D surface: read only the bounded ROI from each timestep,
# then record only the averaged ROI
field = data.phisp[-20:].mean()
field.plot_surfaces(data.boundaries, bounds=bounds, mode="cmap")
```

The field returned by `mean()` exposes `field.inp`, `field.unit`, and `field.boundaries`, so helper functions that read `data.inp` or `data.boundaries` can accept the averaged field.

```python
def build_items_and_geom(data):
    zs = data.unit.length.reverse(data.inp.zssurf)
    boundaries = data.boundaries.mesh(theta_range=[0, np.pi])
    ...

field = data.phisp[-20:].mean()
items, geom = build_items_and_geom(field)
field.plot_surfaces(items, bounds=bounds, mode="cmap")
```

## Record / Replay With Environment Variables

Environment variables are the simplest way to switch modes. The visualization script can keep using normal code such as `emout.Emout("output_dir")`.

```bash
# Normal run: read the original data
python figure.py

# Record: save only the data used for visualization under article-records/
EMOUT_ARTICLE_MODE=record \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
python figure.py

# Replay: read from article-records/ instead of the original large HDF5 output
EMOUT_ARTICLE_MODE=replay \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
python figure.py
```

Common environment variables:

| Environment variable | Meaning |
| --- | --- |
| `EMOUT_ARTICLE_MODE=record` | Record mode. Saves consumed slices, averaged data, and input metadata |
| `EMOUT_ARTICLE_MODE=replay` | Replay mode. Re-runs the same script from the saved bundle |
| `EMOUT_ARTICLE_RECORDS_PATH=article-records` | Root directory for article bundles |
| `EMOUT_ARTICLE_NAME=fig1` | Bundle name for a figure or notebook. Defaults to `default` |
| `EMOUT_ARTICLE_SOURCE_NAME=case_a` | Stable source name for multiple simulations or replay on another machine |
| `EMOUT_ARTICLE_ARCHIVE=zip` | Write a `.zip` archive after recording. `tar.gz` is also supported |

When one notebook or script creates all publication figures, omit `EMOUT_ARTICLE_NAME` to collect them under `default`.
Set `EMOUT_ARTICLE_NAME=fig1` only when the figures should be split into separate bundles.

For multiple simulations, or when paths will change after publication, use `article_source_name`.

```python
data = [
    emout.Emout("case_a/output", article_source_name="case_a"),
    emout.Emout("case_b/output", article_source_name="case_b"),
]
```

If upload size limits matter, enable archives.

```bash
EMOUT_ARTICLE_ARCHIVE=zip python figure.py
```

## Review Checklist

- Check that `plot_surfaces()` receives `bounds`; otherwise a full 3D field may be recorded.
- `data.phisp[-20:].mean()` defaults to a time-axis mean. If the user intends a spatial average, prefer explicit axes such as `mean(axis="x")`.
- Prefer `data.phisp[...].to_numpy()` over `np.asarray(data.phisp[...])` when the intent is to record a consumed array.
- Article replay does not provide particles, backtrace, or remote execution itself. Record the grid data needed for visualization.
