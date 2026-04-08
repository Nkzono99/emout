---
name: harness-improve
description: Audit and improve the emout repository's Claude Code harness — CLAUDE.md, AGENTS.md, .claude/skills/*, .claude/agents/*, .claude/settings.json, and .claude/settings.local.json. Includes itself in scope. Use when harness instructions start drifting from actual project reality, when you notice repeated friction in a workflow, when the user passes a reference repo/URL to compare against, or at the end of a session to capture new learnings.
---

# harness-improve

Make the repository's Claude Code harness keep pace with the project.

The harness is everything under:

- `CLAUDE.md` (Claude Code–facing guide)
- `AGENTS.md` (generic LLM/Codex guide; the two files are deliberately kept aligned but independent)
- `.claude/skills/` (project-local skills, including **this one**)
- `.claude/agents/` (project-local subagents)
- `.claude/settings.json` (shared, committed permissions baseline)
- `.claude/settings.local.json` (per-user, gitignored overrides)

`harness-improve` explicitly includes **itself** as a target of improvement. If you find the instructions in this file lead you astray, rewrite this file too.

## When to invoke

- At the end of a long session that touched new subsystems, to capture learnings before context is lost.
- When you notice you are repeating the same small correction to other harness files.
- When an instruction in `CLAUDE.md` / `AGENTS.md` / a skill / an agent **contradicts** what the code actually does (e.g. file paths, class names, test names).
- When you hit friction that a better skill or agent would remove.
- When the user invokes you with a reference URL or repo (e.g. `/harness-improve https://github.com/...`) — fetch it, treat it as an external benchmark in Phase 3, and adopt only patterns that pull weight here.
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
- `.claude/settings.json` and `.claude/settings.local.json` (latter may be absent)

Also run `git log --oneline -20` for context on what has changed recently. Do not rely on memory — the point of this skill is that memory is unreliable.

Run the inventory in **parallel**: a single message containing one Bash for `git log` plus the Reads/Glob/Grep is much cheaper than sequential round-trips. Files can also be modified between invocations (linters, the user, sibling sessions), so re-Read before editing — never trust your in-context copy.

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
- **Settings** (`.claude/settings.json`): if the rule is *deterministic and tool-mediated* (e.g. "always allow this Bash pattern", "deny that path"), put it in settings rather than as a behavioral instruction in CLAUDE.md. Settings are enforced; instructions are advisory.

Resist the urge to add everything. A useful heuristic: if the pattern fired exactly once in this session, it is probably not a skill yet. Wait for the second occurrence.

If the user supplied an external reference (URL, repo, doc) when invoking this skill, treat it as a benchmark: fetch it, list the patterns it advocates, and adopt only those that match a real friction point you observed in *this* repo. Do not import patterns wholesale — every adopted item must justify its presence by improving truthfulness or removing a recurring mistake here.

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
- Skill `description` fields are *triggers*: write them so a future model recognises when to invoke the skill, not as a prose summary of what it does. "Use when …" phrasing, not "This skill provides …".
- Agent files live at `.claude/agents/<name>.md` with `name` + `description` + optional `tools` frontmatter, then the system prompt.
- `.claude/settings.json` is the **shared, committed** permissions baseline. Keep it permissive (broad allow + narrow deny) so contributors do not get stuck on prompts. `.claude/settings.local.json` is per-user and gitignored — leave it alone unless the user asks.
- Whenever you edit `settings.json` / `settings.local.json`, validate JSON syntax (`python -c "import json; json.load(open(...))"`).

## Non-goals

- Rewriting project code to match the harness. The harness describes the code, not the other way around.
- Adding CI or git hooks. Those are out of scope — this skill is strictly about the local Claude Code harness.
- Standardizing with other repositories. Each repository's harness stands alone.
