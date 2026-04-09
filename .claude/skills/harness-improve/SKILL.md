---
name: harness-improve
description: Audit and improve the emout repository's Claude Code harness — CLAUDE.md, AGENTS.md, .claude/rules/*, .claude/skills/*, .claude/agents/*, .claude/settings.json. Includes itself in scope. Use when harness instructions drift from project reality, when you notice repeated friction, or when the user passes a reference repo/URL to compare against.
---

# harness-improve

Make the repository's Claude Code harness keep pace with the project.

The harness spans:

- `CLAUDE.md` / `AGENTS.md` — slim project overview (~60 lines, identical copies)
- `.claude/rules/` — domain-specific rules, auto-loaded every session
- `.claude/skills/` — project-local skills (including **this one**)
- `.claude/agents/` — project-local subagents
- `.claude/settings.json` — shared permissions baseline

This skill explicitly includes **itself** as a target of improvement.

## Design principles

1. **CLAUDE.md is a landing page, not a manual.** Keep it under 80 lines. Domain rules go in `.claude/rules/`, recipes go in skills.
2. **`.claude/rules/` for always-on constraints.** Rules load every session. Each file should have a clear `<important if="...">` scope so models know when to apply it.
3. **Skills are triggers, not summaries.** The `description` field answers "when should I fire?", not "what do I do?". Include a Gotchas / Common mistakes section.
4. **Settings enforce; instructions advise.** Deterministic behavior (permissions, attribution) goes in `settings.json`, not CLAUDE.md.
5. **Don't write what the code already says.** File paths, class names, and API surface are derivable from the repo. Work logs belong in git history.

## Procedure

### Phase 1 — Inventory

Read every harness file. Also run `git log --oneline -20`. Do reads in parallel.

### Phase 2 — Truth check

Verify load-bearing claims: file paths, test baselines, class names, MPIEMSES external paths. Any mismatch is a harness bug.

### Phase 3 — Capture new patterns

If the session introduced a recurring pattern, decide where it belongs:

- **`.claude/rules/`** — always-on constraint (e.g. backward compat, doc pairs)
- **A skill** — repeatable recipe long enough that inlining would dilute CLAUDE.md
- **An agent** — heavyweight research deserving its own context window
- **`settings.json`** — deterministic, tool-mediated rule

Resist adding everything. If a pattern fired exactly once, wait for the second occurrence.

If the user supplied an external reference, treat it as a benchmark: adopt only patterns that match a real friction point in *this* repo.

### Phase 4 — Prune

Fix broken claims. Delete stale content. Merge duplicates. A stale skill that misleads is worse than no skill.

### Phase 5 — Self-improvement

Audit **this file**. Is any phase redundant? Missing? Is this file still under ~120 lines?

### Phase 6 — Report

Compact summary of changes, grouped by file. Call out anything deferred.

## Invariants

- `CLAUDE.md` and `AGENTS.md` are identical. Edit one, `cp` to the other.
- Rules files use `<important if="...">` tags for scoping.
- Skill `description` fields are triggers ("Use when ..."), not summaries.
- `settings.json` edits must pass `python -c "import json; json.load(open('.claude/settings.json'))"`.

## Non-goals

- Rewriting project code to match the harness.
- Adding CI or git hooks.
- Importing patterns wholesale from other repos.
