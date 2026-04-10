# 変更時チェックリスト

- 変更した機能のテストを追加・更新したか
- `pytest -q` でグリーンを維持しているか
- `ruff check emout/ tests/` と `ruff format --check emout/ tests/` がクリーンか（pre-commit が自動で走るが、CI でも検証される）
- 公開 API を変えるなら `backward-compat.md` に従っているか
- ドキュメント変更なら `doc-pairs.md` に従っているか
- optional 依存を触る場合、未導入時に import で壊れないか
- `emout.plot.surface_cut` の `viz.py` 経由は `scikit-image` / `matplotlib` 必須
- `docs/build/`、`__pycache__/`、一時生成物は編集対象外
