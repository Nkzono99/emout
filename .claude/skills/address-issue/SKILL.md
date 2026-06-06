---
name: address-issue
description: Handle incoming emout GitHub issues end to end with durable fixes. Use when the user asks Codex to respond to, investigate, implement, or close an emout issue, especially when they want long-term operation improvements rather than a short-term patch.
---

# address-issue

Use this skill to turn an incoming issue into a maintainable repo change. The goal is not only to make the symptom disappear, but to update the code, docs, plugin references, and harness context that would prevent the same confusion from recurring.

## Workflow

### 1. Resolve scope

- Identify the issue number or URL. If none is given, inspect open issues for this repo and pick the only clear candidate; otherwise ask.
- Read the issue body, labels, comments, and any linked PRs. Separate confirmed facts, user interpretation, impact, and suggested direction.
- Check `git status --short --branch` before editing. Work with unrelated user changes; do not revert them.
- Read `CLAUDE.md`, `AGENTS.md`, and relevant `.claude/rules/*` before implementation.

### 2. Find the source of truth

- Trace the issue to code first, then docs, tests, and plugin context. Do not rely on docstrings alone when behavior matters.
- For public API behavior, follow `.claude/rules/backward-compat.md`: prefer additive APIs or documentation over changing defaults.
- For docs, use the `docs` skill: author Japanese first, mirror English, and keep section structure in sync.
- For test strategy, use the `run-tests` skill. Add or update tests when behavior changes; docs-only fixes normally need docs validation instead.

### 3. Choose a durable fix

Classify the required change before editing:

- **Code bug:** fix implementation, add a focused regression test, then update docs if user-visible behavior changes.
- **Documentation gap:** update the canonical guide pair and any README or agent-facing guide that repeats the same convention.
- **Plugin context gap:** update `plugins/emout-context` source files, not the installed cache under `~/.codex/plugins/cache`. If a bundled reference mirrors root docs, sync it after editing the root doc.
- **Harness/process gap:** add or update a `.claude/skills/*` or `.claude/rules/*` entry only when the pattern is likely to recur. Keep `CLAUDE.md` and `AGENTS.md` identical if either changes.

When an issue suggests a convenience API, first decide whether the current behavior is a stable contract. If changing it risks existing users, document the contract now and leave a clearly scoped future enhancement for an opt-in API.

### 4. Implement in layers

- Start with the narrow source fix.
- Propagate the same convention to all user-facing surfaces that can generate the same confusion.
- Update plugin skill `Context Sources` when a skill could otherwise miss the relevant guide.
- Update plugin guide tables when reference coverage changes.
- Keep edits surgical; do not reformat unrelated files.

### 5. Validate

Run the checks that match the touched surface:

- Docs: `cd docs && sphinx-build -n -b html -q --keep-going source /tmp/emout-docs-build`
- Python: targeted pytest first, then the broader `run-tests` baseline when risk warrants it.
- Lint: `ruff check ...` and `ruff format --check ...` for touched Python code.
- Skills: `python3 /home/b/b36291/.codex/skills/.system/skill-creator/scripts/quick_validate.py <skill-dir>` for any new or updated skill.
- Harness: `diff -u CLAUDE.md AGENTS.md` when either file is touched.

If a check cannot run because an optional dependency is missing, report that explicitly and run the closest useful substitute.

## Gotchas

- Do not close or comment on the GitHub issue unless the user asked for that action. If posting a comment, include what changed, what was validated, and any remaining risk.
- Plugin references under `plugins/emout-context/references/` are bundled context for installed users; root docs may be newer and should remain the source when both exist.
- The installed plugin cache is an output, not the repo source. Treat cache edits as temporary debugging only.
- Keep issue summaries concise in the final response: issue number/title, files changed, validation, and any deferred follow-up.
