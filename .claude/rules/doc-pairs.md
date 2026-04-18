<important if="editing documentation, README, or guide files">

# ドキュメント日英ペア

**日本語版を source of truth とする。** 新規・更新はいずれも日本語側を
先に書き、英語側は日本語版の構造（節の並び、表、コード例）をそのまま
ミラーする。英語はネイティブが読んで自然な文章にする（逐語訳ではない）。

公開 API や挙動を変えた場合、以下の日英ペアを同じ PR で両方更新すること:

- `README.md`（canonical, JA）⇔ `README.en.md`（mirror, EN）
- `CLAUDE.md`（canonical, JA）⇔ `AGENTS.md`（同一内容をコピー）
- `docs/source/index.rst`（canonical, JA）⇔ `docs/source/index.en.rst`（mirror, EN, `:orphan:`）
- `docs/source/guide/*.ja.md`（canonical, JA）⇔ `docs/source/guide/*.md`（mirror, EN）
  - `quickstart` / `plotting` / `animation` / `inp` / `units` / `boundaries` / `backtrace` / `distributed`

詳しいワークフローは `docs` skill（`.claude/skills/docs/SKILL.md`）を参照。

</important>
