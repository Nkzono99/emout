# Article Data Recording And Replay (`EMOUT_ARTICLE_MODE`)

The article feature records only the minimum grid data that a figure script
actually uses, then regenerates the same figures without the original large
EMSES simulation output. It is designed for paper data publication,
supplemental material, and reproducible bundles shared with collaborators.

Keep the normal analysis script unchanged, and switch between `normal`,
`record`, and `replay` with environment variables or `Emout()` arguments.

## When to use it

- You want to publish only the slices consumed by `plot()` or `to_numpy()`.
- You want one bundle for all figures created by a Jupyter notebook or one script.
- You want plots comparing multiple simulation outputs to replay on another machine.
- You want `plasma.inp` / `plasma.toml` / boundary meshes / small diagnostics included.

Particle data, backtrace, and remote execution itself are not included in an
article replay bundle. Replay is focused on regenerating figures from recorded
grid slices and input metadata.

## Quick start

Write the figure script normally. Axis order is `(t, z, y, x)`.

```python
import emout

data = emout.Emout("output_dir")
ymid = data.inp.ny // 2

data.phisp[-1, :, ymid, :].plot(cmap="viridis")
arr = data.ex[-1, :, ymid, :].to_numpy()
```

Run the same script in record or replay mode with environment variables.

```bash
# Normal run
python figure.py

# Record article data
EMOUT_ARTICLE_MODE=record \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
python figure.py

# Replay from recorded article data
EMOUT_ARTICLE_MODE=replay \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
python figure.py
```

The same settings can be passed explicitly.

```python
data = emout.Emout(
    "output_dir",
    article_mode="record",
    article_records_path="article-records",
)
```

## What is saved

In record mode, emout saves only the data materialized by `plot()` and
`to_numpy()`. For example, `data.phisp[-1, :, ymid, :].plot()` stores only
that 2D slice in `data.h5`. Reusing the same field and selector does not
write duplicate data.

| File | Contents | Purpose |
| --- | --- | --- |
| `manifest.json` | Recorded field, selector, shape, slice axes, unit metadata | Matches requested replay slices to saved slices |
| `data.h5` | Recorded NumPy arrays | Rebuilds `Data1d` / `Data2d` / `Data3d` objects |
| `source.json` | Original simulation path, basename, recorded file hashes | Matches sources on another machine and detects tampering |
| `plasma.inp` | Input file | Replays `data.inp`, unit conversion, and boundary meshes |
| `plasma.toml` | TOML input file | Replays `data.toml` |
| `icur`, `pbody` | Small diagnostics, when present | Replays `data.icur` / `data.pbody` |

Datasets inside `data.h5` are written with HDF5 gzip compression. HDF5
decompresses them transparently during replay, so normal `plot()` /
`to_numpy()` usage does not change.

## Directory layout

The basic layout is `records-path/datasets/<source>/<article-name>/`.
When `article_name` is omitted, it defaults to `default`.

```text
article-records/
└── datasets/
    └── output_dir-012345abcd/
        ├── source.json
        └── default/
            ├── manifest.json
            ├── data.h5
            ├── plasma.inp
            ├── plasma.toml
            ├── icur
            └── pbody
```

`<source>` is normally built from the source directory basename plus an
absolute-path hash. Because absolute paths change on another machine, replay
first tries the direct `<source>` match, then falls back to the basename in
`source.json`.

## Multiple figures and multiple simulations

Set `EMOUT_ARTICLE_NAME` to split bundles by figure, such as `fig1` and
`fig2`. If it is omitted, everything goes into `default`, which is convenient
for collecting all figures from a notebook or one script.

```bash
EMOUT_ARTICLE_MODE=record \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
EMOUT_ARTICLE_NAME=fig1 \
python figure.py
```

Recreating `Emout()` with the same `article_name` appends only slices that
are not already recorded. This supports Jupyter cell reruns and scripts that
open `Emout()` inside helper functions.

When a script opens multiple simulation outputs, each source gets its own
record directory. If multiple outputs share the same basename, pass
`article_source_name` so the same source can be selected after publication.

```python
data = [
    emout.Emout("case_a/output", article_source_name="case_a"),
    emout.Emout("case_b/output", article_source_name="case_b"),
]
```

Using the same `article_source_name` in record and replay mode gives a stable
directory such as `article-records/datasets/case_a/default/`, even when
absolute paths change.

## Archives and publication size

Enable `article_archive` to write an archive for each bundle automatically.

```python
data = emout.Emout(
    "output_dir",
    article_mode="record",
    article_records_path="article-records",
    article_archive="zip",
)
```

```bash
EMOUT_ARTICLE_MODE=record \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
EMOUT_ARTICLE_ARCHIVE=zip \
python figure.py
```

| Setting | Archive created |
| --- | --- |
| `article_archive=True` / `EMOUT_ARTICLE_ARCHIVE=1` | `<article-name>.tar.gz` |
| `article_archive="tar.gz"` / `EMOUT_ARTICLE_ARCHIVE=tar.gz` | `<article-name>.tar.gz` |
| `article_archive="zip"` / `EMOUT_ARTICLE_ARCHIVE=zip` | `<article-name>.zip` |

During replay, emout automatically extracts the matching `.tar.gz` or `.zip`
when the extracted directory is not present. Zip is useful when an upload
service rejects `.tar.gz` or when readers prefer a format that is easy to
open on Windows.

## What replay can do

In replay mode, `emout.Emout()` returns a proxy that reads the recorded bundle
instead of the original HDF5 output. Recorded slices support the usual `plot()`
and `to_numpy()` calls.

```python
data = emout.Emout(
    "output_dir",
    article_mode="replay",
    article_records_path="article-records",
)

data.phisp[-1, :, ymid, :].plot()
arr = data.ex[-1, :, ymid, :].to_numpy()
```

Vector aliases also work when the required components are recorded.

```python
data.exz[-1, :, ymid, :].plot()
```

Input metadata and boundaries are replayable too.

```python
data.boundaries.plot()
data.phisp[-1].plot_surfaces(data.boundaries)
icur = data.icur
pbody = data.pbody
```

Accessing an unrecorded slice raises an exception. This is intentional: it
checks that the public bundle contains the data required to reproduce the
figure.

## Configuration reference

| Argument | Environment variable | Default | Meaning |
| --- | --- | --- | --- |
| `article_mode` | `EMOUT_ARTICLE_MODE` | `normal` | Switches `normal` / `record` / `replay` |
| `article_records_path` / `records_path` | `EMOUT_ARTICLE_RECORDS_PATH` / `EMOUT_RECORDS_PATH` | none | Root directory for bundles |
| `article_name` | `EMOUT_ARTICLE_NAME` | `default` | Bundle name for a figure or notebook |
| `article_source_name` | `EMOUT_ARTICLE_SOURCE_NAME` | none | Stable source name for replaying multiple sources elsewhere |
| `article_archive` | `EMOUT_ARTICLE_ARCHIVE` | none | Writes a `tar.gz` or `zip` archive |

## Common gotchas

- `record` / `replay` requires `article_records_path`. Set it with an environment variable or argument.
- If multiple sources share a basename, such as `case_a/output` and `case_b/output`, pass `article_source_name`.
- Prefer `data.phisp[...].to_numpy()` over `np.asarray(data.phisp[...])` so the recording intent is explicit.
- Replay does not provide particles, backtrace, or remote execution. Run those from the original data, then record the grid slices used for visualization.
- If the original script depends on random styling or external files, the article bundle alone cannot fully reproduce the figure. Publish the figure script as well.

## Related classes

See the API reference (`emout.article` and `emout.core.facade`) for full
signatures.

- `emout.Emout` — public entry point for `article_mode` / `article_records_path` / `article_name`
- `ArticleRecorder` — internal class that saves slices, metadata, and archives in record mode
- `ArticleReplayEmout` — replay-mode proxy that reads recorded bundles
