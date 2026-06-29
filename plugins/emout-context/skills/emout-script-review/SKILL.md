---
name: emout-script-review
description: Review Python analysis scripts that use emout for axis order, unit conversion, plotting APIs, memory use, optional dependencies, and remote execution.
---

Use this skill when a user asks whether an emout analysis script is correct, why a script produces a surprising plot, how to make a script safer for large outputs, or how to adapt a script for HPC remote execution.

## Response Language

- Respond in the user's language.
- Keep code identifiers, filenames, commands, EMSES variable names, and Python API names unchanged.
- If language is unclear, default to Japanese for this repository.

## Context Sources

- Bundled references: `../../references/quickstart.ja.md`, `../../references/quickstart.md`, `../../references/plotting.ja.md`, `../../references/plotting.md`, `../../references/animation.ja.md`, `../../references/animation.md`, `../../references/article.ja.md`, `../../references/article.md`, `../../references/units.ja.md`, `../../references/units.md`, `../../references/boundaries.ja.md`, `../../references/boundaries.md`, `../../references/distributed.ja.md`, `../../references/distributed.md`, `../../references/backtrace.ja.md`, `../../references/backtrace.md`.
- Bundled docs: `../../docs/library-context.md`, `../../docs/library-context.en.md`, `../../docs/analysis-pitfalls.md`, `../../docs/analysis-pitfalls.en.md`, `../../docs/article-publication.md`, `../../docs/article-publication.en.md`.
- Repo root docs only when the full checkout is available and may be newer.
- User-provided script, expected figure, output directory layout, and data size.

## Workflow

- Review for correctness before style: axis order, target plane, variable names, unit conversion, and time index.
- State that grid data axis order is `(t, z, y, x)` when commenting on indexing.
- Check whether `.val` or `.val_si` is applied before slicing large data.
- Check whether SI conversion metadata exists or is assumed without evidence.
- Check vector attributes and slice planes for component mismatch.
- If the script calls `data.backtrace`, check that `position`, `velocity`, `dt`, and probability-grid axes are EMSES-unit values, or are converted from SI with `data.unit.length.trans(...)`, `data.unit.v.trans(...)`, or `data.unit.t.trans(...)`.
- Check plotting code for unnecessary global matplotlib state when the user wants reusable scripts.
- For PyVista 3D plotting, check that overlays reuse a `plotter`, that `show=False` / screenshots are used for batch environments, and that emout 2.20.0+ installs PyVista as a regular dependency.
- If the script is intended for publication/article data, check that `plot_surfaces()` has `bounds`, time averages use `data.field[-N:].mean()`, and record/replay can be controlled with `EMOUT_ARTICLE_*` environment variables.
- For HPC use, suggest explicit `Emout.remote()`, `remote_scope()`, and `remote_figure()` when it reduces data transfer or login-node work.
- Treat `RemoteSession` as internal architecture. Recommend direct `RemoteSession` construction only for maintainers or tests.
- Provide minimal corrected snippets rather than rewriting the whole script unless requested.

## Output

Use code-review style, with findings first:

```text
## 指摘
- [High] ...
- [Medium] ...

## 修正例
...

## 残る確認点
...
```

If no issues are found, say so clearly and mention any remaining assumptions such as missing input metadata or unknown data size.
