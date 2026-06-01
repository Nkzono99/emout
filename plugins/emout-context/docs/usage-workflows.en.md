Lang: [English](usage-workflows.en.md) | [日本語](usage-workflows.md)

# emout Usage Workflows

This document summarizes standard workflows plugin skills can present to users. In actual responses, shorten and adapt them to the user's data path, physical quantity, and environment.

## Load an Output Directory

```python
import emout

data = emout.Emout("output_dir")
```

Then access the target physical quantity through attributes derived from EMSES filenames.

```python
data.phisp
data.nd1p
data.j1xy
```

Grid data slicing axis order is `(t, z, y, x)`. When creating a plane, explicitly identify the target plane and fixed axis.

## Visualize a Plane

```python
ymid = data.inp.ny // 2
data.phisp[-1, :, ymid, :].plot()
```

For large outputs, slice with a form such as `[-1, :, ymid, :]` before reading the full array through `val_si`. For quantities with a wide dynamic range, such as density or particle distributions, consider `norm="log"`.

## Check Units

```python
data.phisp[-1, :, ymid, :].val_si
data.unit.v.reverse(1.0)
```

If SI conversion is not as expected, check the input file's `!!key dx=...,to_c=...` header or the `[meta.unit_conversion]` table in `plasma.toml`.

## Overlay Boundaries

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
data.phisp[-1, :, ymid, :].plot(ax=ax)
data.phisp[-1, :, ymid, :].plot_surfaces(ax=ax, surfaces=data.boundaries)
```

If the boundary collection is empty, check whether finbound is configured in the input file and whether the shape is supported by emout.

## Use Remote Execution

```bash
emout server start --partition gr20001a --memory 60G
```

```python
from emout.distributed import remote_figure, remote_scope

data = emout.Emout("output_dir").remote()

with remote_scope():
    ymid = int(data.inp.ny // 2)
    with remote_figure(savefilepath="figures/phisp.png"):
        data.phisp[-1, :, ymid, :].plot()
```

Remote execution is available in Python 3.10+ environments. Check server state with `emout server status`.

`RemoteSession` is the internal Actor that shares `Emout` instances and intermediate results on the worker. Scripts normally should not construct it directly; use `Emout.remote()`, `remote_scope()`, `remote_figure()`, or `RemoteFigure` for wrapping existing code.

## Record And Replay Article Publication Data

For paper data publication and reproducible figure bundles, keep the visualization script unchanged and switch record / replay with environment variables.

```bash
# Normal run
python figure.py

# Record
EMOUT_ARTICLE_MODE=record \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
python figure.py

# Replay
EMOUT_ARTICLE_MODE=replay \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
python figure.py
```

Omit `EMOUT_ARTICLE_NAME` to collect multiple figures into one bundle. Set `EMOUT_ARTICLE_NAME=fig1` when figures should be split.
Use `EMOUT_ARTICLE_SOURCE_NAME=case_a` for multiple simulations or replay on another machine. Use `EMOUT_ARTICLE_ARCHIVE=zip` or `tar.gz` when upload size limits matter.

For 3D surfaces, pass `bounds` so publication data is limited to the plotted ROI. Time averages can be recorded as averaged data with `mean()`.

```python
field = data.phisp[-20:].mean()
field.plot_surfaces(data.boundaries, bounds=bounds, mode="cmap")
```

## Create a Visualization Script

When creating a script from a natural-language request, first decide the target quantities, plane, timestep, and save path. For large outputs, do not start the server inside the script by default; document `emout server start ...` as a setup step before running it.

```python
import argparse

import emout
from emout.distributed import remote_figure, remote_scope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--output", default="phisp.png")
    args = parser.parse_args()

    rdata = emout.Emout(args.output_dir).remote()
    with remote_scope():
        ymid = int(rdata.inp.ny // 2)
        with remote_figure(savefilepath=args.output):
            rdata.phisp[-1, :, ymid, :].plot()


if __name__ == "__main__":
    main()
```

## Triage a Problem

1. Prepare a minimal script and traceback.
2. Check `python -m pip show emout`, installation method, and Python version.
3. Summarize the output directory file listing.
4. Check grid size, unit conversion, and boundary settings in the input file.
5. Verify axis order `(t, z, y, x)` and whether the script loads full arrays.
