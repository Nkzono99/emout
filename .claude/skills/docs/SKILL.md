---
name: docs
description: Update or create emout's bilingual (ja/en) documentation — README.{md,en.md}, docs/source/guide/*.{md,ja.md}, CLAUDE.md/AGENTS.md — keeping the pairs in sync, reflecting recent code or API changes, and verifying that readers can actually use the result. Use when the user asks to write, rewrite, translate, polish, extend, or reflect code changes in any doc, add a new guide page, or surface a new feature in the documentation.
---

# docs

Keep emout's bilingual documentation consistent and readable when adding
or updating pages.

This skill adapts the **Context Gathering → Refinement & Structure →
Reader Testing** loop from Anthropic's `doc-coauthoring` skill to
emout's fixed surface (Sphinx + MyST, ja/en pairs, 8 guide pages today)
and borrows MUSUBI `technical-writer`'s strict bilingual discipline. It
is procedural — do not ask the user questions unless a judgement call
genuinely needs their input.

## When to use this skill

- The user asks to write, rewrite, translate, polish, or reorganise any
  doc.
- You just shipped a code or API change that needs to be reflected in
  the guides.
- You're adding a new guide page or feature section and want to wire it
  into the toctree and README at the same time.
- You notice drift between a ja/en pair (different examples, different
  section counts, stale version numbers) and want to resync.

Invoke `run-tests` separately when you touch code in addition to docs.

## Canonical source per surface

`.claude/rules/doc-pairs.md` lists the pairs. **Which side is the source
of truth matters** because it decides where you type new prose first.

**Japanese is canonical across the whole repo.** emout's target audience
is Japanese-speaking EMSES users (see `CLAUDE.md`: やりとりの言語は
日本語), so every bilingual pair is authored Japanese-first and the
English version mirrors it.

| Surface | Canonical side | Mirror | Why |
| --- | --- | --- | --- |
| `docs/source/guide/*.ja.md` | **Japanese** | `docs/source/guide/*.md` | Terminology, section order, tone, and example selection are decided on the Japanese side. EN mirrors the structure but should read as native English prose (not a literal translation). |
| `README.md` (JA) | **Japanese** | `README.en.md` | Already Japanese-first. |
| `CLAUDE.md` | **Japanese** | `AGENTS.md` | `AGENTS.md` is a literal copy; `diff CLAUDE.md AGENTS.md` must be empty. |

**Rule of thumb:** when writing new content, draft the Japanese version
first, then produce the English mirror. When editing existing content,
touch both sides in the same commit — never ship a half-update.

**Note on existing history.** Commit `067f1fe Improve Japanese docs:
natural wording and sync with English` reflects an older rule where EN
was the reference. That rule is superseded — but the guides written
under it are still in sync structurally, so no historical rewrite is
needed. Only *new* content and *future* updates follow the JA-canonical
rule.

Pages covered today: `quickstart` / `plotting` / `animation` / `inp` /
`units` / `boundaries` / `backtrace` / `distributed`. If you add
another, update `.claude/rules/doc-pairs.md` in the same commit.

## Bilingual rules

These are load-bearing — follow them even when translating a single
paragraph.

1. **Structure parity.** Both language versions must have the same
   section count, same heading levels, and same section order. Add a
   section in one language → add it in the other in the same position.
   Quick check: `diff <(grep '^#' guide/foo.md) <(grep '^#' guide/foo.ja.md)`
   should be empty (modulo the heading text itself).
2. **Code blocks are identical, prose is translated.** Function calls,
   variable names, keyword arguments, shell commands, Python identifiers,
   EMSES parameter names (`phisp`, `nd1p`, `boundary_types`, `plasma.inp`,
   `remote_figure`, …) are always English. Comments inside code blocks
   *may* be translated, but keep them short so line-for-line diffing still
   works.
3. **Tables stay aligned.** Column order, column count, and row order
   must match between ja/en. Only the content of description cells
   changes.
4. **Cross-links resolve to the same-language counterpart.** If the
   English guide cross-links to another guide, the Japanese guide
   cross-links to the same guide's `.ja.md` file — not back to the
   English one.
5. **No half-translations.** Don't leave an English paragraph inside
   `*.ja.md` because "it's clearer in English". Translate it, or cut it
   from both sides.
6. **Version numbers, dates, file paths, numeric baselines are
   identical on both sides.** When you bump the pytest baseline in
   CLAUDE.md, bump it in AGENTS.md in the same commit.

When writing Japanese prose, prefer natural wording over a literal
translation. When writing English prose, write as a native English
technical writer, not as a translator. Both versions should read as if
each were the original.

## Workflow

### 1. Context gathering

**Goal:** close the gap between what the code/repo knows and what the
reader will need.

Before typing any prose:

- **For a code-driven update:** read the affected source (not the
  docstring — the actual function body, its defaults, its exceptions).
  Check `git log` for the last few commits in that area. If a public
  API changed, grep for every mention of it across
  `docs/source/guide/` and `README*.md` — missing a call site is the
  most common failure of this workflow.
- **For a polish pass:** read *both* language versions of every file
  you plan to touch before editing anything. Note existing terminology
  choices so you don't invent new variants mid-page.
- **For a translation sync:** structurally diff first
  (`diff <(grep '^#' A) <(grep '^#' B)`), then reconcile section by
  section.
- **Always:** scan sibling guides for consistent terminology. New
  phrasing should match what the rest of the docs already use.

**Ask the user only when:**

- A judgement call affects the public API surface (e.g. "should I
  deprecate this or remove it?").
- You're about to delete a substantial chunk of existing content.
- The feature is experimental and the stability story is unclear from
  the code.

**Exit condition:** you can list, in your head, (a) which files you
will touch, (b) the rough section structure of each change, and (c)
which sibling pages need cross-link updates. Do not start writing prose
until those three are clear.

### 2. Refinement & structure

**Goal:** produce content that reads well in both languages with
structural parity guaranteed by construction.

**Scaffold-first drafting.** Never type the first paragraph of a new
section before the section headers are in place:

1. **Draft the Japanese file first (`.ja.md`).** Terminology, section
   order, and tone are decided here. Use placeholder bodies like
   `<!-- TODO: このセクションの目的 -->` so structure is visible.
2. **For a new page, use the scaffold in
   `.claude/skills/docs/templates/new-guide.ja.md`** as the starting
   skeleton, and its English mirror `templates/new-guide.md` for the
   `.md` side. Both match the structure of the existing 8 guides so
   new pages don't drift.
3. **Fill one section at a time, in both languages, in lockstep.**
   Write the Japanese section body first, then produce the English
   mirror immediately. Do not finish the Japanese page and then
   translate the whole thing later — by the time you come back you
   will have forgotten the choices you made.
4. **Write code examples once**, paste into both files unchanged. Only
   the surrounding prose varies.
5. **Tables: headers and keys first, prose cells second.** Build the
   skeleton on the Japanese side, copy it to the English side, then
   fill both in lockstep.
6. **Add cross-references last.** They pay for themselves only after
   the structure is stable, and they are the #1 source of rebuild
   churn.

**Brainstorm-before-curate.** For any section covering more than one
point (a comparison table, a checklist, a tip block, a "common
mistakes" list), brainstorm 5–20 candidate entries before committing
any of them to prose. This is borrowed from `doc-coauthoring`: it
surfaces angles you would otherwise forget, and it keeps you from
writing filler to pad a section. After brainstorming, curate down to
what readers actually need.

**Idiomatic emout phrasing** (JA is canonical — pick the Japanese
wording first, then mirror naturally in English):

- "EMSES シミュレーション出力" → "EMSES simulation output" (not
  "EMSES output data")
- "グリッドデータ" → "grid data" (not "field data" — reserve that for
  specific contexts)
- "境界" → "boundary" for finbound entries; "境界メッシュ" → "boundary mesh"
  when talking about rendering
- "リモート実行" → "remote execution" for the Dask path
- Keep `plasma.inp`, `plasma.toml`, `&ptcond`, `boundary_types`, etc.
  in backtick-code font on both sides
- "スライスの軸順序は `(t, z, y, x)`" → "Axis order is `(t, z, y, x)`"
  — always mention this where indexing appears

**Editing discipline.** When modifying an existing page, use the `Edit`
tool for surgical diffs. Do not reprint the whole file with `Write`
unless the change is a near-total rewrite. This keeps review scopes
small and makes the doc history diffable.

**Exit condition:** both language files pass the structural diff check
(`diff <(grep '^#' A) <(grep '^#' B)`), all code blocks are
byte-identical, and every cross-link has been resolved at least once in
prose (even if anchor verification happens later).

### 3. Reader testing

**Goal:** confirm the docs actually work for someone who doesn't have
your context. Build-clean is necessary but not sufficient.

**3a. Build-clean.** The minimum bar — do not skip:

```bash
cd docs
sphinx-build -n -b html -q --keep-going source /tmp/emout-docs-build 2>&1 \
    | grep -iE "<file-you-touched>|myst\.xref_missing|local id not found"
```

Resolve every new `myst.xref_missing` or `local id not found` warning.
See **Cross-reference gotchas** below for the common traps.

**3b. Verify cross-links resolve in the built HTML:**

```bash
grep -oE 'href="[^"]*your-new-page\.html[^"]*"' \
    /tmp/emout-docs-build/guide/related-page.html | head
```

**3c. Reader testing with a fresh agent (recommended for new pages).**
This is `doc-coauthoring`'s core idea adapted to emout. Spawn an
`Explore` subagent that only sees the new/changed guide file(s) and
ask it to answer realistic questions. The subagent has no conversation
context, so it will trip over anything you explained-but-didn't-write.

```
Agent(
  subagent_type="Explore",
  description="Reader-test the new backtrace guide (JA)",
  prompt="""
    Read docs/source/guide/backtrace.ja.md only. Do not read any other file.
    日本語でガイドを読み、emout を初めて触る日本語話者の視点で次の質問に答えてください:

      1. 単一種の粒子について、ある観測点での到達確率分布を計算するにはどうすればよいか？
      2. result.vxvz.plot() は具体的に何を描画する？ result.plot_energy_spectrum() とは何が違う？
      3. Dask を使ったリモート実行でも同じように動く？ 何を変える必要がある？
      4. fetch() は plot() と比べて何を手に入れるためのもの？
      5. 初見のユーザーが最もハマりそうな点は？

    各質問ごとに:
      - 回答（1〜3 文）
      - 不明瞭だった点 / ガイドが前提としている知識（1 文）
      - 根拠になるガイド内の引用（1 行）

    全体で 400 語以内。日本語で答えてください。
  """
)
```

After iterating on the Japanese version, run the same subagent against
`backtrace.md` to confirm the English mirror is equally legible to a
non-Japanese reader.

Use the subagent's answers to spot blind spots: anything it got wrong,
any assumption-it-called-out, and any question it answered by inferring
from a code example rather than from prose. Those are the holes you
fix in another Refinement pass.

**Default question templates for emout guides:**

- *Quickstart / feature discovery:* "How do I X?", "What does the
  default do if I don't pass anything?", "What do I need installed
  first?"
- *API reference-ish guides (backtrace, boundaries, units):* "What
  does this function return?", "What are the required vs optional
  arguments?", "How does this interact with remote execution?"
- *Workflow guides (distributed, animation):* "What's the difference
  between the compat mode and the explicit `remote()` mode?",
  "When should I use action=X vs action=Y?", "What does this NOT
  support?"
- *Concept guides (plotting, inp, units):* "What's the one thing I'll
  get wrong on my first try?"

**Exit condition:** build is clean, the subagent can answer the golden
questions without "the doc doesn't say" or "I'm inferring from the
example", and the structural parity check still passes.

## Adding a new guide page

Follow this checklist in order.

1. **Copy `.claude/skills/docs/templates/new-guide.ja.md`** to
   `docs/source/guide/<name>.ja.md`. This is the canonical draft —
   edit the `<...>` placeholders first, don't write prose into a
   placeholder structure.
2. **Copy `.claude/skills/docs/templates/new-guide.md`** to
   `docs/source/guide/<name>.md` as the English mirror scaffold.
3. **Fill sections in lockstep**, Japanese first then English mirror,
   one section at a time.
4. **Register in `docs/source/index.rst`** under the `User Guide`
   toctree. Japanese entry first, mirroring existing order:
   ```rst
      guide/<name>.ja
      guide/<name>
   ```
5. **Add `<name>` to `.claude/rules/doc-pairs.md`** in the
   comma-separated list.
6. **Surface in both README feature tables.** URLs:
   `https://nkzono99.github.io/emout/guide/<name>.ja.html` and
   `https://nkzono99.github.io/emout/guide/<name>.html`.
7. **Cross-link from one or two related existing guides** (ja/en
   pair-to-pair). Don't scatter links everywhere — pick the pages a
   reader would naturally land on first.
8. **Run the reader-testing workflow** (§ 3 above). For a brand-new
   page, run the subagent step against the Japanese version first
   (that is what real readers will open) and then optionally against
   the English version.
9. **Run `pytest -q`** if the feature you documented also has runtime
   code changes; pure docs changes skip this.

## Reflecting code changes into docs

When a code change ships and the docs need to catch up:

1. **Find every affected guide.** Grep for the symbol/API name across
   `docs/source/guide/` and `README*.md`. Missing a file is the most
   common failure mode.
2. **Update in pairs.** Never commit an update to `foo.md` without the
   matching update to `foo.ja.md`. If you notice the pairs have already
   drifted, resync them first as a separate edit (easier to review).
3. **Update every spot where the old behaviour is described**, not
   just the first one. Use grep, not memory.
4. **Refresh code examples.** If a signature changed, the example must
   compile against the new signature. Copy-paste the updated example
   into both language files so they stay byte-identical.
5. **Bump obvious counters.** The `pytest` baseline in CLAUDE.md and
   AGENTS.md is the most common one.
6. **Reader-test at least the changed section** with a subagent if the
   API surface moved meaningfully (new kwarg, new return type, new
   error mode). Build-clean alone does not catch "my explanation
   doesn't match the new signature".

## Cross-reference gotchas

MyST + Sphinx has a few specific traps. Learning them once saves a
rebuild cycle each time.

- **Internal doc-to-doc links.** `[text](other.md)` works and resolves
  to `other.html` in the build. For the JA side use `[text](other.ja.md)`.
- **Anchor links across files require a Sphinx label, not a slugified
  heading.** `[text](other.md#my-section)` will usually emit
  `myst.xref_missing`. If you want deep linking:
  - add an explicit label above the heading and reference it with
    Sphinx's `{ref}` role (not standard Markdown), or
  - drop the anchor and link to the page top, describing the section
    by name in prose — "see the "Remote execution" section of [the
    animations guide](animation.md)". **This is what the repo currently
    uses** and what you should default to.
- **Japanese headings get ASCII-stripped slugs.** A heading like
  `## リモート実行（Emout.remote()）` becomes `#emout-remote` in HTML.
  Anchor linking against it is fragile; prefer prose references.
- **Do not link to `../api/*.rst` from a guide.** Those are Sphinx
  sources, not the built pages. Either describe the class in prose
  with backticks and let readers navigate via the API sidebar, or
  (rarely) use Sphinx's `{doc}` role — the repo does not use it today.
- **Code blocks with triple backticks and braces.** Literal `{}` inside
  a triple-backtick block needs no escaping. Only MyST role syntax
  (``{role}`text` ``) needs braces escaped.

## Common mistakes

- **Editing only the canonical side** because "I'll mirror it later."
  You won't — do both sides in the same commit.
- **Translating code identifiers.** `data.phisp` is not `data.電位`.
  APIs and parameter names are universal.
- **Mismatched section counts** because you added a subsection in one
  file and forgot the other. The structural diff check exists for this
  reason — run it.
- **Stale `doc-pairs.md`** after adding a new guide. The list is
  small, update it in the same commit.
- **Forgetting the README feature table.** A new guide that's not in
  the README table is effectively hidden — users won't find it.
- **Skipping reader testing for "obviously clear" changes.** The
  changes that feel obviously clear to you are exactly the ones where
  a fresh reader will ask the question you thought you answered.
- **Reprinting an entire file with `Write` for a one-paragraph
  change.** Use `Edit` — review scopes stay small and history stays
  diffable.
- **Committing `docs/build/`.** It's a build artifact; never stage it.
