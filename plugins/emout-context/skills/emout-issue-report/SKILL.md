---
name: emout-issue-report
description: Prepare concise GitHub issues for emout bugs, questions, improvement requests, and reproducible analysis problems.
---

Use this skill when a user wants to report an emout bug, convert a traceback into a GitHub issue, ask a maintainable question, or propose a user-facing improvement.

## Response Language

- Respond in the user's language unless the user asks for an English issue body.
- Keep code identifiers, filenames, commands, EMSES variable names, and traceback symbols unchanged.
- If language is unclear, default to Japanese for this repository.

## Context Sources

- Bundled references: `../../references/README.md`, `../../references/README.en.md`, `../../references/quickstart.ja.md`, `../../references/quickstart.md`.
- Bundled docs: `../../docs/analysis-pitfalls.md`, `../../docs/analysis-pitfalls.en.md`, `../../docs/library-context.md`, `../../docs/library-context.en.md`, `../../docs/skills-guide.md`, `../../docs/skills-guide.en.md`.
- Repo root docs only when the full checkout is available and may be newer.
- User-provided symptom, expected behavior, reproduction script, traceback, environment, and sanitized input/output summaries.

## Workflow

- Classify the request as bug report, question, documentation request, enhancement, or usage issue.
- Extract the minimal reproduction: emout version, Python version, installation method, OS/HPC context, output file summary, input metadata snippet, script, and traceback.
- Mask personal paths, hostnames, job IDs, tokens, and unpublished dataset names.
- Include axis order `(t, z, y, x)` and unit conversion metadata when they matter to the issue.
- Distinguish expected behavior from actual behavior.
- If the issue is likely MPIEMSES3D input or simulator behavior rather than emout, say so and suggest using the MPIEMSES3D context plugin or repository.
- Produce an issue draft the user can paste into GitHub.

## Output

Use the response language and translate headings when appropriate:

```text
## タイトル案
...

## Issue 本文
### Summary
...
### Reproduction
...
### Expected behavior
...
### Actual behavior
...
### Environment
...
### Additional context
...

## 投稿前チェック
- ...
```

Keep the draft concise. Do not include raw private logs unless the user has already sanitized them.
