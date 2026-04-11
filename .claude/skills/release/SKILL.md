---
name: release
description: Bump the version, tag, push, and create a GitHub release with structured release notes. Use when the user asks to release a new version, cut a release, or bump the version.
---

# release

Create a new release of emout.

## Procedure

### 1. Determine the new version

Check the current version and commits since the last tag:

```bash
grep 'version' pyproject.toml | head -1
git log --oneline $(git describe --tags --abbrev=0)..HEAD
```

Choose the bump level:

- **patch** (2.10.0 → 2.10.1): bug fixes only
- **minor** (2.10.0 → 2.11.0): new features, non-breaking
- **major** (2.10.0 → 3.0.0): breaking changes (API removal, Python version drop)

If unclear, ask the user.

### 2. Pre-flight checks

Run in parallel:

```bash
python -m pytest tests/ -q
ruff check emout/ tests/
ruff format --check emout/ tests/
```

All must pass before proceeding. Do **not** release with failing tests.

### 3. Bump version in all locations

- `pyproject.toml` → `version = "X.Y.Z"`
- `docs/source/conf.py` → `release = 'X.Y.Z'`

### 4. Commit, tag, push

```bash
git add pyproject.toml docs/source/conf.py
git commit -m "Bump version to X.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

### 5. Write release notes and create GitHub release

#### Release note format

Use the following template (inspired by pydantic / rich / Keep a Changelog):

```markdown
<!-- 1-2 sentence summary of the release in Japanese -->
境界モデルに円柱型を追加し、単位変換の不具合を修正したリリースです。

## Highlights
- `CylinderBoundary` を追加 (abc1234)
- `UnitTranslator` の逆変換精度の問題を修正 (def5678)

## New Features
- Add `CylinderBoundary` for finbound cylinder type (abc1234)

## Bug Fixes
- Fix reverse unit conversion precision loss (def5678)

## Changes
- Drop Python 3.8 support (1234567)

## Documentation
- Add boundaries guide (ja/en) (89abcde)

**Full Changelog**: https://github.com/Nkzono99/emout/compare/vPREV...vX.Y.Z
```

#### Rules for writing

- **Summary** (冒頭): 1-2 文の日本語。非開発者にも伝わるように書く。
- **Highlights**: 最も重要な変更を 2-3 個。ユーザーが「アップデートすべきか」を判断できる情報。
- **Categories**: 以下の 4+1 カテゴリから該当するものだけ使う:
  - `New Features` — 新しいクラス・メソッド・CLI コマンド
  - `Bug Fixes` — 正しさに関する修正
  - `Changes` — 破壊的変更、依存関係変更、動作変更
  - `Documentation` — ドキュメント・翻訳の変更
  - `Infrastructure` — テスト・CI・リファクタリング（ユーザーに直接影響しない場合は省略可）
- **各エントリ**: 英語、動詞で始める（Add / Fix / Update / Remove / Drop）、末尾にコミットハッシュ短縮形 `(abc1234)`。PR があれば `#123` 形式でリンク。
- **Infrastructure は省略可**: テスト追加やCI修正だけのリリースでは書かなくてよい。ユーザー向けの変更がある場合のみ記載。
- **Full Changelog**: `compare/vPREV...vX.Y.Z` のリンクを末尾に必ず付ける。

#### How to generate

```bash
# Commit log to work from
git log --oneline $(git describe --tags --abbrev=0)..HEAD

# Create release
gh release create vX.Y.Z --title "vX.Y.Z" --notes "$(cat <<'EOF'
...release notes here...
EOF
)"
```

### 6. Verify

- Check that the release page looks correct: `gh release view vX.Y.Z`
- Confirm the PyPI publish workflow ran and succeeded. emout has
  `.github/workflows/python-publish.yml` configured with
  `on: release: types: [published]`, so `gh release create` triggers
  it automatically within a few seconds:

  ```bash
  # The newest run should be vX.Y.Z / release / success
  gh run list --workflow=python-publish.yml --limit 3

  # Independent check against PyPI itself (uses the public JSON API)
  curl -sS https://pypi.org/pypi/emout/json \
    | python -c "import json,sys; d=json.load(sys.stdin); print('latest:', d['info']['version'])"
  ```

  If the workflow run is still `in_progress`, wait and re-check. If it
  finishes `failure`, inspect with `gh run view <run-id> --log` — do
  **not** delete the tag to retry; instead, fix forward with a new
  patch release.

## Common mistakes

- **Forgetting `docs/source/conf.py`** — the Sphinx docs embed the version string.
- **Tagging before committing** — the tag must point at the version-bump commit.
- **Not running tests** — a released tag with broken tests is hard to undo.
- **Wrong bump level** — dropping Python 3.8 is a breaking change (major or minor, ask the user).
- **Highlights が空** — 必ず 1 個以上入れる。ユーザーが最初に読む場所。
- **Summary を英語で書く** — emout のユーザーは日本語話者が多いため、冒頭は日本語で。
- **PyPI チェックをスキップ** — GitHub リリースが作れても PyPI 側の publish が失敗していることはあるので、上の `gh run list` と `curl` の両方を実行してから「完了」と報告する。
