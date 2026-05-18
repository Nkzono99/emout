---
name: emout-usage-guide
description: Guide emout users through loading EMSES outputs, accessing variables, slicing grid data, reading parameters, and applying SI unit conversion.
---

Use this skill when a user asks how to use emout, how to load an output directory, how to access EMSES variables, how to slice data, how to inspect input parameters, or how to convert between EMSES and SI units.

## Response Language

- Respond in the user's language.
- If the request mixes languages, use the main natural language of the request.
- Keep code identifiers, filenames, commands, EMSES variable names, and parameter names unchanged.
- If language is unclear, default to Japanese for this repository.

## Context Sources

- Bundled references: `../../references/README.md`, `../../references/README.en.md`, `../../references/quickstart.ja.md`, `../../references/quickstart.md`, `../../references/inp.ja.md`, `../../references/inp.md`, `../../references/units.ja.md`, `../../references/units.md`.
- Bundled docs: `../../docs/library-context.md`, `../../docs/library-context.en.md`, `../../docs/usage-workflows.md`, `../../docs/usage-workflows.en.md`, `../../docs/skills-guide.md`, `../../docs/skills-guide.en.md`.
- Repo root docs only when the full checkout is available and may be newer.
- User-provided output path, input file, target quantity, and current analysis code.

## Workflow

- Identify whether the user has an output directory, an input file only, or both separately.
- Start from `import emout` and `data = emout.Emout("output_dir")` unless the user needs appended output or separated input/output paths.
- State that grid data slicing axis order is `(t, z, y, x)` whenever indexing appears.
- Map requested physical quantities to common attributes such as `phisp`, `nd1p`, `j1x`, `j1xy`, `j1xyz`, `icur`, and `pbody`.
- Check whether SI conversion is supported by input metadata before recommending `val_si`.
- Prefer small examples that slice before loading large arrays.
- Point to plotting, animation, boundary, backtrace, or distributed guides only when they are relevant to the request.

## Output

Use the response language and translate headings when appropriate:

```text
## 最小コード
...

## 軸順序と単位
...

## 次に確認すること
...
```

Keep examples short and adapt paths, variable names, and slice indices to the user's data.
