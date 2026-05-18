---
name: emout-feedback-report
description: Collect, classify, sanitize, and draft GitHub issues for emout user feedback about bugs, improvement requests, documentation gaps, usability friction, and analysis workflow pain points.
---

Use this skill when a user wants to send feedback to emout maintainers, report a rough bug without a polished reproduction yet, describe confusing behavior, request an improvement, or capture friction from analysis / visualization workflows. The default output should be a GitHub Issue draft unless the user explicitly asks for another format.

## Response Language

- Respond in the user's language unless the user asks for an English issue body.
- Keep code identifiers, filenames, commands, EMSES variable names, traceback symbols, and API names unchanged.
- If language is unclear, default to Japanese for this repository.

## Context Sources

- Bundled references: `../../references/README.md`, `../../references/README.en.md`, `../../references/quickstart.ja.md`, `../../references/quickstart.md`, `../../references/plotting.ja.md`, `../../references/plotting.md`, `../../references/distributed.ja.md`, `../../references/distributed.md`.
- Bundled docs: `../../docs/analysis-pitfalls.md`, `../../docs/analysis-pitfalls.en.md`, `../../docs/library-context.md`, `../../docs/library-context.en.md`, `../../docs/skills-guide.md`, `../../docs/skills-guide.en.md`.
- Repo root docs only when the full checkout is available and may be newer.
- User-provided symptom, expectation, script, traceback, output listing, environment, and privacy constraints.

## Workflow

- Classify feedback as one or more of: bug, usability issue, documentation gap, feature request, performance issue, remote execution issue, compatibility issue, or question.
- Separate confirmed facts from interpretation and proposed fixes.
- Ask for missing critical details only when the feedback cannot be made actionable without them.
- For bugs, collect the smallest useful reproduction: emout version, Python version, installation method, platform/HPC context, output file summary, input metadata snippet, script, and traceback.
- For remote execution feedback, include `emout server status`, Python 3.10+ status, session name, whether `remote_scope()` / `remote_figure()` / `RemoteFigure` is used, and whether the issue happens locally.
- For visualization/script-generation feedback, include target variable, intended plane, axis order `(t, z, y, x)`, expected figure, actual figure, and whether SI conversion metadata exists.
- Sanitize personal paths, hostnames, job IDs, tokens, unpublished dataset names, and large raw logs.
- Produce a GitHub Issue draft even when some details are missing; mark unknown fields as "未確認" / "Unknown" rather than blocking.
- Include a short "追加であると良い情報" / "Useful additional information" section after the draft when reproduction details are incomplete.
- If the request is already a polished GitHub issue, keep the issue format and only sanitize or tighten it.

## Output

Use the response language and translate headings when appropriate:

```text
## 分類
- ...

## GitHub Issue 下書き
### Title
...
### Summary
...
### Context
...
### Impact
...
### Evidence / Reproduction
...
### Suggested direction
...

## 追加であると良い情報
- ...
```

Keep the draft concise and actionable. Do not include private raw logs unless the user explicitly says they are sanitized.
