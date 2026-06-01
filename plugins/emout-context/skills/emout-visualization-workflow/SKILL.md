---
name: emout-visualization-workflow
description: Design emout plots, animations, 3D PyVista views, boundary overlays, and remote rendering workflows for EMSES outputs.
---

Use this skill when a user asks how to plot or animate EMSES output with emout, compare fields, draw vector data, overlay boundaries, create 3D views, save figures, or render plots on an HPC compute node. If the user asks for a complete runnable script, prefer the `emout-visualization-script` skill.

## Response Language

- Respond in the user's language.
- Keep code identifiers, filenames, commands, EMSES variable names, and plot option names unchanged.
- If language is unclear, default to Japanese for this repository.

## Context Sources

- Bundled references: `../../references/plotting.ja.md`, `../../references/plotting.md`, `../../references/animation.ja.md`, `../../references/animation.md`, `../../references/article.ja.md`, `../../references/article.md`, `../../references/boundaries.ja.md`, `../../references/boundaries.md`, `../../references/distributed.ja.md`, `../../references/distributed.md`, `../../references/quickstart.ja.md`, `../../references/quickstart.md`.
- Bundled docs: `../../docs/library-context.md`, `../../docs/library-context.en.md`, `../../docs/usage-workflows.md`, `../../docs/usage-workflows.en.md`, `../../docs/analysis-pitfalls.md`, `../../docs/analysis-pitfalls.en.md`, `../../docs/article-publication.md`, `../../docs/article-publication.en.md`.
- Repo root docs only when the full checkout is available and may be newer.
- User-provided target quantity, slice plane, desired output format, and environment.

## Workflow

- Determine whether the requested view is 1D, 2D scalar, 2D vector, 3D scalar/vector, animation, or remote rendering.
- State axis order `(t, z, y, x)` and choose a slice that matches the requested physical plane.
- Prefer `plot()`, `cmap()`, and `contour()` for 1D/2D scalar data.
- Prefer combined vector attributes such as `j1xy` or `j1xyz` when plotting vectors.
- For dynamic ranges, consider `norm="log"` only when physically appropriate.
- For animations, use `gifplot()` and clarify whether output should be inline, GIF, HTML, or a saved file.
- For 3D views, mention that PyVista is optional and can be installed with `pip install "emout[pyvista]"`.
- For boundaries, pass `data.boundaries` or individual meshes to the plotting API.
- If the user mentions paper data publication, article record/replay, reproducible bundles, `EMOUT_ARTICLE_*`, or averaged public data, prefer the `emout-article-publication` skill or consult `article-publication.*.md`.
- For HPC workflows, use `emout server start`, `Emout.remote()`, `remote_scope()`, and `remote_figure()` when remote rendering is requested.
- Treat `RemoteSession` as internal architecture. For user-facing code, recommend `Emout.remote()`, `remote_scope()`, `remote_figure()`, or `RemoteFigure`.

## Output

Use the response language and translate headings when appropriate:

```text
## 可視化方針
...

## Python 例
...

## 注意点
...
```

Avoid long plotting scripts unless the user asks for a complete script.
