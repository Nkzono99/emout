---
name: emout-output-diagnose
description: Diagnose emout loading, plotting, unit conversion, boundary, optional dependency, and remote execution problems for EMSES output directories.
---

Use this skill when a user provides an emout traceback, says an output directory cannot be loaded, sees a wrong plot, has missing units or variables, cannot render boundaries or 3D plots, or has trouble with `emout server`.

## Response Language

- Respond in the user's language.
- Keep code identifiers, filenames, commands, EMSES variable names, traceback symbols, and parameter names unchanged.
- If language is unclear, default to Japanese for this repository.

## Context Sources

- Bundled references: `../../references/quickstart.ja.md`, `../../references/quickstart.md`, `../../references/inp.ja.md`, `../../references/inp.md`, `../../references/units.ja.md`, `../../references/units.md`, `../../references/boundaries.ja.md`, `../../references/boundaries.md`, `../../references/distributed.ja.md`, `../../references/distributed.md`, `../../references/backtrace.ja.md`, `../../references/backtrace.md`.
- Bundled docs: `../../docs/analysis-pitfalls.md`, `../../docs/analysis-pitfalls.en.md`, `../../docs/library-context.md`, `../../docs/library-context.en.md`.
- Repo root docs only when the full checkout is available and may be newer.
- User-provided traceback, output listing, `plasma.inp` / `plasma.toml` excerpts, and analysis script.

## Workflow

- Classify the failure as import/install, path/layout, HDF5 variable discovery, input parameter parsing, unit conversion, plotting, optional dependency, boundary parsing, or remote execution.
- Ask for missing critical information only after extracting what can be inferred from the provided traceback or script.
- Check axis order `(t, z, y, x)` for wrong-slice or wrong-plane results.
- Check whether full arrays are loaded before slicing for memory or performance failures.
- Check `!!key` or `[meta.unit_conversion]` before diagnosing SI conversion.
- For backtrace problems, verify whether SI values were accidentally passed to APIs that require EMSES-unit inputs.
- For 3D plotting failures, check whether the installed emout environment includes PyVista; in emout 2.20.0+ it is a regular dependency, so stale editable installs may need reinstalling.
- For remote failures, check Python 3.10+, `emout server status`, session naming, and whether the server is already running.
- When the user says `remote_session`, map that to the internal shared `RemoteSession` architecture and check whether their script should instead use `Emout.remote()`, `remote_scope()`, `remote_figure()`, or `RemoteFigure`.
- For suspected emout bugs, reduce the reproduction to the smallest output listing, input snippet, script, and traceback.

## Output

Use the response language and translate headings when appropriate:

```text
## 診断
- ...

## まず試すこと
1. ...

## 追加で必要な情報
- ...
```

Separate confirmed facts from likely causes. Do not invent file names or parameter values that were not provided.
