---
name: emout-visualization-script
description: Create or improve complete emout visualization scripts from natural-language requests or existing scripts, including remote_scope, remote_figure, and RemoteFigure for large EMSES outputs.
---

Use this skill when a user asks Codex to create a Python visualization script without an existing script, to extend an existing emout analysis script, to add panels/quantities/animations, or to adapt plotting code for large HPC outputs with remote execution.

## Response Language

- Respond in the user's language.
- Keep code identifiers, filenames, commands, EMSES variable names, and Python API names unchanged.
- If language is unclear, default to Japanese for this repository.

## Context Sources

- Bundled references: `../../references/quickstart.ja.md`, `../../references/quickstart.md`, `../../references/plotting.ja.md`, `../../references/plotting.md`, `../../references/animation.ja.md`, `../../references/animation.md`, `../../references/units.ja.md`, `../../references/units.md`, `../../references/boundaries.ja.md`, `../../references/boundaries.md`, `../../references/distributed.ja.md`, `../../references/distributed.md`.
- Bundled docs: `../../docs/library-context.md`, `../../docs/library-context.en.md`, `../../docs/usage-workflows.md`, `../../docs/usage-workflows.en.md`, `../../docs/analysis-pitfalls.md`, `../../docs/analysis-pitfalls.en.md`.
- Repo root docs only when the full checkout is available and may be newer.
- User-provided natural-language plotting request, existing script, output directory, target variables, slice plane, time range, save path, and HPC constraints.

## Workflow

- Convert the user's physical request into concrete emout attributes, slice indices, plot type, and output files.
- State and apply grid axis order `(t, z, y, x)` when generating indexing code.
- For scripts without an existing base, generate a runnable script with `argparse`, clear input/output paths, and conservative defaults.
- For existing scripts, preserve the user's structure where practical and patch only the needed parts.
- Prefer slicing before `.val` / `.val_si` to avoid loading full 4D HDF5 data.
- Use `val_si` or SI-labeled plots only when unit conversion metadata is available or when the user's data is known to include it.
- For large visualization, prefer this structure:
  - `emout server start ...` in the setup note, not inside the script unless explicitly requested.
  - `rdata = emout.Emout(args.output_dir).remote()`.
  - `with remote_scope():` around remote refs and repeated plotting.
  - `with remote_figure(savefilepath=args.output):` around matplotlib/emout plotting commands.
- Explain that `RemoteSession` is the shared Dask Actor used internally. User scripts normally use `Emout.remote()`, `remote_scope()`, `remote_figure()`, or `RemoteFigure`, not direct `RemoteSession` construction.
- Use `RemoteFigure().open()` / `.close()` only when adapting existing notebook/script code where adding a `with remote_figure():` block would be disruptive.
- If the user requests animation for large data, prefer remote `gifplot()` patterns from the distributed and animation references.
- Include optional boundary overlays with `data.boundaries` only when the requested plot needs geometry context.

## Output

For a new script, use:

````text
## 可視化 script
```python
...
```

## 実行方法
...

## 前提
...
````

For an existing script review/modification, lead with the concrete changes and provide a patch-sized snippet or edited file guidance. Keep scripts complete enough to run, but do not invent unavailable variable names; ask for an output listing if the target variable is ambiguous.
