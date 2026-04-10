---
name: release
description: Bump the version, tag, push, and create a GitHub release. Use when the user asks to release a new version, cut a release, or bump the version.
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

### 5. Create GitHub release

Generate release notes from the commit log. Group by category:

- **Breaking Changes** — API removals, Python version drops
- **New Features** — new classes, methods, CLI commands
- **Bug Fixes** — correctness fixes
- **Infrastructure** — tests, CI, refactoring, harness
- **Documentation** — docs, translations

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --notes "..."
```

### 6. Verify

- Check that the release page looks correct: `gh release view vX.Y.Z`
- If a PyPI publish workflow exists, confirm it triggered

## Common mistakes

- **Forgetting `docs/source/conf.py`** — the Sphinx docs embed the version string.
- **Tagging before committing** — the tag must point at the version-bump commit.
- **Not running tests** — a released tag with broken tests is hard to undo.
- **Wrong bump level** — dropping Python 3.8 is a breaking change (major or minor, ask the user).
