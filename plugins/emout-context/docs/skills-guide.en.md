Lang: [English](skills-guide.en.md) | [日本語](skills-guide.md)

# emout Context Skill Guide

The skills in this plugin use the bundled files in `references/` as their primary source so they can help users who installed emout with pip and do not have the full repository available. In a development checkout, root docs with the same names may be newer and should be checked when relevant.

## Common Policy

- Respond in the user's language. Keep code identifiers, filenames, commands, and EMSES variable names in English.
- Treat `emout.Emout` as the public entry point, and always state that grid data slicing uses axis order `(t, z, y, x)`.
- SI conversion is available when `plasma.inp` has a `!!key dx=...,to_c=...` header or `plasma.toml` has `[meta.unit_conversion]`. Do not claim SI conversion is available without that basis.
- For large HDF5 data, prefer examples that slice by time, plane, and range before loading a full 4D array.
- Input files, logs, and output paths may include personal paths, hostnames, job IDs, or secrets. Summarize or mask them before preparing external issues.

## Skill List

| Skill | When to use it | Main inputs | Main outputs | Main references |
| --- | --- | --- | --- | --- |
| `emout-usage-guide` | Explain basic emout usage, variable access, unit conversion, and parameter inspection | Output directory, target quantity, input file format | Minimal code, axis order, unit conversion notes, next guide links | `README.en.md`, `quickstart.md`, `inp.md`, `units.md` |
| `emout-article-publication` | Guide article record/replay, environment variables, archives, and averaged data for paper/publication bundles | Visualization script, publication-data requirements, records path, multiple simulations, averaging window | Record/replay commands, environment variables, saved-data granularity, gotchas | `article-publication.en.md`, `usage-workflows.en.md` |
| `emout-visualization-workflow` | Design 1D/2D/3D plots, animations, and boundary overlays | Quantity, slice condition, view type, output path | Plotting steps, Python examples, dependencies, saving method | `plotting.md`, `animation.md`, `boundaries.md`, `distributed.md`, `backtrace.md` |
| `emout-visualization-script` | Create or improve visualization scripts from natural-language requests or existing scripts | Goal, output directory, quantities, existing script, HPC constraints | Runnable script, remote execution variant, run steps, assumptions | `quickstart.md`, `plotting.md`, `animation.md`, `distributed.md`, `backtrace.md` |
| `emout-output-diagnose` | Diagnose loading failures, plot errors, unit conversion, or remote execution issues | Traceback, output listing, input file, environment | Likely causes, check commands, minimal fixes, missing information | `quickstart.md`, `inp.md`, `units.md`, `backtrace.md`, `analysis-pitfalls.en.md` |
| `emout-script-review` | Review analysis scripts that use emout | Python script, goal, data size, sample output | Required fixes, recommended improvements, risks, corrected snippets | `library-context.en.md`, `analysis-pitfalls.en.md`, `plotting.md`, `distributed.md`, `backtrace.md` |
| `emout-feedback-report` | Turn bugs, improvement requests, documentation gaps, and analysis workflow friction into GitHub Issue drafts | Symptom, expectation, impact, script, traceback, environment, privacy constraints | Classification, issue draft, missing details, label suggestions | `analysis-pitfalls.en.md`, `library-context.en.md`, `README.en.md`, `backtrace.md` |
| `emout-issue-report` | Turn bugs, questions, or improvement requests into GitHub issues | Symptom, expected behavior, reproduction steps, environment, traceback | Issue title, body, label suggestions, pre-submit checklist | `analysis-pitfalls.en.md`, `README.en.md`, `quickstart.md`, `backtrace.md` |

## Typical Requests

```text
I want to plot phisp from output_dir on an xz-plane with emout.
Show how to record and replay `data.phisp[-20:].mean().plot_surfaces(..., bounds=...)` for paper data publication.
Create a script that saves phisp and nd1p from output_dir in two panels. Use remote_figure because the data is large.
Diagnose this emout loading failure from the traceback.
Review whether this analysis.py uses axis order and SI conversion correctly.
Turn my pain points with emout remote_figure into maintainer feedback.
Draft a GitHub issue for this emout problem.
```

For MPIEMSES3D input parameters or simulator run failures, use the MPIEMSES3D context plugin alongside this one. The emout plugin focuses on reading generated outputs, visualizing them, and validating analysis scripts. `RemoteSession` is the internal shared Dask Actor, so user-facing scripts normally use `Emout.remote()`, `remote_scope()`, `remote_figure()`, and `RemoteFigure`.
