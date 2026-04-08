---
name: harness-improve
description: Audit and improve the emout repository's Claude Code harness — CLAUDE.md, AGENTS.md, .claude/skills/*, .claude/agents/*, and .claude/settings.local.json. Includes itself in scope. Use when harness instructions start drifting from actual project reality, when you notice repeated friction in a workflow, or at the end of a session to capture new learnings.
---

# harness-improve

Make the repository's Claude Code harness keep pace with the project.

The harness is everything under:

- `CLAUDE.md` (Claude Code–facing guide)
- `AGENTS.md` (generic LLM/Codex guide; the two files are deliberately kept aligned but independent)
- `.claude/skills/` (project-local skills, including **this one**)
- `.claude/agents/` (project-local subagents)
- `.claude/settings.local.json` (permission list)

`harness-improve` explicitly includes **itself** as a target of improvement. If you find the instructions in this file lead you astray, rewrite this file too.

## When to invoke

- At the end of a long session that touched new subsystems, to capture learnings before context is lost.
- When you notice you are repeating the same small correction to other harness files.
- When an instruction in `CLAUDE.md` / `AGENTS.md` / a skill / an agent **contradicts** what the code actually does (e.g. file paths, class names, test names).
- When you hit friction that a better skill or agent would remove.
- **Not** for routine minor edits — those belong in the normal edit flow. `harness-improve` is for deliberate, systematic passes.

## What "improvement" means here

In order of priority:

1. **Truthfulness.** Every claim in the harness must match the current repo state. Fix stale file paths, class names, test counts, and command snippets. Run the commands the harness recommends and update anything that does not still work.
2. **Load-bearing guidance.** Each line should earn its place. If you cannot imagine a concrete decision a future session would make differently because of a sentence, consider removing it.
3. **Self-consistency.** `CLAUDE.md` and `AGENTS.md` share a section outline. If one is updated the other should be re-checked so they do not silently diverge.
4. **Discoverability.** New skills and agents should be announced in `CLAUDE.md §12` so future sessions know they exist.
5. **Minimalism.** Do not add skills "in case they are useful". Only add a skill when you see a concrete pattern that was repeated or mistaken at least once.

## Procedure

Run the phases in order. Do not skip phases, but you can no-op a phase if nothing changed.

### Phase 1 — Inventory

Read every harness file end to end:

- `CLAUDE.md`
- `AGENTS.md`
- every `.claude/skills/*/SKILL.md` (including this file)
- every `.claude/agents/*.md`
- `.claude/settings.local.json`

Also run `git log --oneline -20` for context on what has changed recently. Do not rely on memory — the point of this skill is that memory is unreliable.

### Phase 2 — Truth check

For each load-bearing claim in the harness, verify it still holds. Specific things to check:

- File paths and module names. `Glob` or `Read` them.
- Test baseline counts and command flags. Actually run them (`run-tests` skill is fine for this).
- Class names and public API mentioned in skills and the harness docs. `Grep` them to confirm they still exist.
- The MPIEMSES3D external paths referenced from `AGENTS.md §3` and agents — those are absolute paths to a sibling repo and can drift.
- Current date references in AGENTS.md (test baseline section, etc.). Update to the real current date if they look stale.

Any mismatch is a bug in the harness.

### Phase 3 — Capture new patterns

If the session's work introduced a new recurring pattern — a new subsystem, a new common mistake, a new recipe — decide whether it belongs in:

- **The docs themselves** (`CLAUDE.md` / `AGENTS.md`): a short note, a pointer, a gotcha.
- **A new skill**: a repeatable recipe that is long enough that inlining it in the docs would dilute them, and where a future session would benefit from being told "use skill X" instead of re-discovering.
- **A new agent**: a research or investigation task that is heavyweight enough to deserve its own context window.

Resist the urge to add everything. A useful heuristic: if the pattern fired exactly once in this session, it is probably not a skill yet. Wait for the second occurrence.

### Phase 4 — Prune

For anything in the harness that failed Phase 2 or that you cannot justify under Phase 3:

- Fix it if it is broken.
- Delete it if it is no longer true or no longer useful.
- Merge duplicate guidance. Two skills saying the same thing is worse than one.

Deletion is acceptable for skills and agents you created, too. A stale skill that mislead is strictly worse than no skill.

### Phase 5 — Self-improvement

Audit **this file**. Specifically:

- Is the "When to invoke" list accurate given how you actually used the skill this session? Add or remove bullets.
- Is any phase above redundant? Collapse.
- Is any phase missing? This is the only place to record cross-session knowledge about how harness maintenance itself should work. Write it down.
- Is the skill still small enough that a future session will actually read it? If it grew beyond ~200 lines, something is wrong.

If you did nothing in Phase 5, that is fine — but at least confirm you looked.

### Phase 6 — Report

Output a compact summary of changes you made, grouped by file. Call out anything deferred (e.g. "considered adding an `add-test-fixture` skill but the pattern only fired once, postponed"). This lets the user decide whether to sign off.

## Invariants to preserve

- `CLAUDE.md` and `AGENTS.md` are independent regular files (not a symlink). Do not re-symlink them.
- `CLAUDE.md` and `AGENTS.md` keep the same top-level section outline so cross-referencing `§N` stays meaningful.
- Skills live at `.claude/skills/<name>/SKILL.md` with a `name` + `description` frontmatter only (no extra fields unless you know the harness reads them).
- Agent files live at `.claude/agents/<name>.md` with `name` + `description` + optional `tools` frontmatter, then the system prompt.
- Do not weaken `.claude/settings.local.json` permissions casually. If a command needs a new permission, add it narrowly.

## Non-goals

- Rewriting project code to match the harness. The harness describes the code, not the other way around.
- Adding CI or git hooks. Those are out of scope — this skill is strictly about the local Claude Code harness.
- Standardizing with other repositories. Each repository's harness stands alone.
