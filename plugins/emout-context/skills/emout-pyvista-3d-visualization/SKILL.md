---
name: emout-pyvista-3d-visualization
description: Design and troubleshoot emout PyVista 3D visualization workflows, including plot3d/plot_pyvista for scalar and vector fields, boundary mesh overlays, trace/backtrace path overlays, streamline seed modes, screenshots/HTML export, unit/axis handling, and HPC-safe execution.
---

Use this skill when a user asks for 3D visualization with emout and PyVista, especially `plot3d()`, `plot_pyvista()`, scalar volume/slice/contour views, vector streamlines/quiver, `surfaces=data.boundaries`, boundary `plot3d()`, trace/backtrace trajectory overlays, PyVista screenshots/HTML export, or how to run PyVista rendering on HPC systems.

## Response Language

- Respond in the user's language.
- Keep code identifiers, filenames, commands, EMSES variable names, and PyVista option names unchanged.
- If language is unclear, default to Japanese for this repository.

## Context Sources

- Primary bundled references: `../../references/pyvista.ja.md`, `../../references/pyvista.md`.
- Related references: `../../references/plotting.ja.md`, `../../references/plotting.md`, `../../references/boundaries.ja.md`, `../../references/boundaries.md`, `../../references/backtrace.ja.md`, `../../references/backtrace.md`, `../../references/distributed.ja.md`, `../../references/distributed.md`, `../../references/quickstart.ja.md`, `../../references/quickstart.md`.
- Bundled docs: `../../docs/library-context.md`, `../../docs/library-context.en.md`, `../../docs/analysis-pitfalls.md`, `../../docs/analysis-pitfalls.en.md`.
- Repo root docs only when the full checkout is available and may be newer, especially `docs/source/guide/pyvista.ja.md` and `docs/source/guide/pyvista.md`.
- User-provided output path, target variables, desired layers, output format, screenshots, traceback, and execution environment.

## Workflow

- Classify the requested scene:
  - 2D scalar slice in 3D: `data.phisp[-1, z_or_y_or_x, :, :].plot3d(...)`
  - 3D scalar field: `data.phisp[-1].plot3d(mode="box" | "volume" | "slice" | "contour")`
  - 3D vector field: `data.j1xyz[-1].plot3d(mode="stream" | "quiver")`
  - Boundaries: `data.boundaries.plot3d(plotter=...)` or `surfaces=data.boundaries`
  - Trace paths: `trace.plot3d(plotter=...)` from `data.trace.forward/backward/both(..., get_trace=True)`
- State grid axis order whenever indexing appears: grid data is `(t, z, y, x)`, and a single 3D field is `(z, y, x)`. PyVista receives coordinates in `(x, y, z)`.
- Treat PyVista as a regular emout dependency in emout 2.20.0+. For `ModuleNotFoundError: pyvista`, suggest updating/reinstalling emout or the editable environment.
- Build overlays by reusing one `plotter`:
  - first layer with `show=False`
  - subsequent layers with `plotter=plotter`
  - final layer or explicit call uses `show=True`, `screenshot()`, or `filename=...`
- For boundary overlays, prefer `surfaces=data.boundaries` when plotting field/vector data. Use `data.boundaries.plot3d()` when drawing boundary geometry by itself or adding it manually to an existing plotter.
- For trace overlays, mention that `get_probabilities=False` means probability-derived `alpha="auto"` is unavailable; use explicit `alpha` when needed.
- For streamline control, use `seed_mode`, `seed_plane`, `seed_position`, `n_points`, `source_center`, `source_radius`, `tube_radius`, and `tube_radius="magnitude"` as appropriate.
- Keep `use_si` and `offsets` consistent across all layers. Default `use_si=True` displays SI coordinates/values when unit metadata exists.
- For saving, prefer `filename=` / `savefilename=` on emout plot helpers when available, or `plotter.screenshot("figure.png")` followed by `plotter.close()`. Use `.html` only when the user's PyVista/Jupyter/trame environment supports it.
- For HPC/login-node environments, do not recommend heavy PyVista rendering on a login node. Suggest compute-node or supported visualization workflows, `show=False`, screenshot output, and batch-friendly scripts.
- Distinguish PyVista from emout remote Matplotlib rendering: `remote_figure()` is for Matplotlib images and does not keep a PyVista scene on the worker.
- For complete runnable scripts, provide concise scripts with `argparse` when asked; otherwise give focused snippets and the sequence of layers.

## Output

Use the response language and translate headings when appropriate:

```text
## 可視化方針
...

## Python 例
...

## 実行・保存
...

## 注意点
...
```

Keep examples small enough to adapt. Do not invent unavailable variables; ask for `ls` output or `data` attribute names when the requested quantity is ambiguous.
