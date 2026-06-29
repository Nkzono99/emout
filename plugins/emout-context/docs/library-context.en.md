Lang: [English](library-context.en.md) | [日本語](library-context.md)

# emout Library Context

This document summarizes the shared assumptions plugin skills should use when helping emout users. Prefer the detailed guides bundled in `references/` for API details.

## Basic Model

- The public entry point is `emout.Emout`.
- Typical initialization is `data = emout.Emout("output_dir")`.
- When the input file and output directory are separate, use `emout.Emout(input_path="/path/to/plasma.toml", output_directory="output_dir")`.
- To combine appended outputs, use `emout.Emout("output_dir", ad="auto")`.
- Grid data slicing axis order is `(t, z, y, x)`.

## Common Attributes

| Attribute | Meaning |
| --- | --- |
| `data.phisp` | Time-series grid data for potential |
| `data.nd1p` | Species-1 number density |
| `data.j1x`, `data.j1y`, `data.j1z` | Species-1 current density components |
| `data.j1xy`, `data.j1xyz` | Auto-combined 2D / 3D vector data |
| `data.icur`, `data.pbody` | Text diagnostics loaded as `pandas.DataFrame` |
| `data.inp` | Input parameters from `plasma.inp` / `plasma.toml` |
| `data.toml` | Structured parameters when TOML input is available |
| `data.unit` | EMSES and SI unit conversion |
| `data.boundaries` | Python objects for finbound boundaries |

## Unit Conversion

SI conversion is available when the input file contains unit conversion metadata.

```python
data.unit.v.trans(1.0)       # SI -> EMSES
data.unit.v.reverse(1.0)     # EMSES -> SI
data.phisp[-1].val_si        # ndarray in SI units
```

For outputs without unit conversion metadata, do not assume `val_si` is valid. First check the input file's `!!key` header or `[meta.unit_conversion]` table.

## Visualization Model

- Use `plot()`, `cmap()`, and `contour()` for 1D/2D views.
- Use `gifplot()` for GIF/HTML time-series output.
- Use `plot3d()` for 3D views. PyVista is a regular dependency in emout 2.20.0+, so import errors usually indicate an old environment or stale editable install.
- Build boundary meshes from `data.boundaries` and pass them to APIs such as `plot_surfaces`.
- On HPC systems, `emout server start` plus `Emout.remote()` / `remote_figure()` can offload work to compute nodes.
- For large visualization scripts, use `Emout.remote()`, `remote_scope()`, `remote_figure()`, or `RemoteFigure` instead of constructing `RemoteSession` directly. Explain `RemoteSession` as the internal shared Dask Actor.

## Support Boundary

emout is a library for reading, analyzing, and visualizing generated EMSES outputs. Improving existing analysis scripts and creating visualization scripts from natural-language requests are in scope for the emout plugin. MPIEMSES3D input design and simulator run failures belong to the MPIEMSES3D context plugin.
