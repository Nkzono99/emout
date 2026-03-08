# AGENTS.md

このファイルは、このリポジトリで Codex/LLM エージェントが開発・保守を行うための実務ガイドです。

## 1. プロジェクト概要

- `emout` は、EMSES の出力ファイル（主に `*.h5`, `plasma.inp`, `icur`, `pbody`）を Python で読み込み、解析・可視化するライブラリです。
- 公開入口は `emout.Emout`（`emout/__init__.py`）です。
- コア機能は以下に分かれます。
  - I/O と Facade: `emout/emout/facade.py`, `emout/emout/io/`
  - データモデル: `emout/emout/data/`
  - 可視化: `emout/plot/`
  - 入力パラメータ・単位系: `emout/utils/emsesinp.py`, `emout/utils/units.py`, `emout/emout/units.py`
  - 実験機能（依存追加あり）: `emout/emout/backtrace/`, `emout/distributed/`

## 2. まず読むファイル

- `README.md`（利用者向け API と使用例）
- `pyproject.toml`（依存関係・Python バージョン・配布設定）
- `tests/conftest.py`（最小データセットの作り方）
- `tests/data/test_data.py`（データアクセスの期待仕様）
- `tests/plot/test_contour3d.py`（3D 可視化 API の期待仕様）

## 3. 開発環境セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

ドキュメントを触る場合のみ:

```bash
pip install -r docs/requirements.txt
```

## 4. よく使うコマンド

```bash
# 全テスト
pytest -q

# 対象テストのみ
pytest tests/plot/test_contour3d.py -q
pytest tests/data/test_data.py -q

# ドキュメントビルド
sphinx-build -b html docs/source docs/build/html

# パッケージビルド
python -m build
```

## 5. 実装時の必須ルール

- 互換性を最優先にする。
  - `Emout.__getattr__` の動的解決（`p{species}`, `r[eb][xyz]`, `{name}{axis1}{axis2}`）は既存ユーザーコードに直結するため破壊しない。
- 軸順序を崩さない。
  - グリッドデータは基本 `(t, z, y, x)`、3D ボリュームは `(z, y, x)` 前提。
- ファイル命名規則を守る。
  - グリッド: `{name}00_0000.h5`
  - 粒子: `p{species}{comp}(e?){seg}_{part}.h5`
- 単位変換は `plasma.inp` 1 行目の `!!key dx=[...],to_c=[...]` に依存する。未設定ケース（`unit is None`）を壊さない。
- `docs/build/`、`__pycache__/`、一時生成物は原則編集対象外。

## 6. 変更時チェックリスト

- 変更した機能に対応するテストを追加・更新したか。
- `pytest -q` か対象テストを実行し、失敗理由を記録したか。
- 公開 API や挙動を変えた場合、`README.md` と `docs/source/` を更新したか。
- optional 依存機能（`vdsolverf`, `dask`）を触る場合、依存未導入時に import で壊れないか確認したか。

## 7. 現在のテストベースライン（2026-03-08）

`pytest -q` の結果は `48 passed / 3 failed`。

- `tests/data/test_data.py::test_open_data`
- `tests/data/test_data.py::test_data_type`
  - `emout.data` の公開互換性に関する失敗
- `tests/data/test_data.py::test_name2unit[nd12p]`
  - `name2unit` の密度名パターン期待値との不一致

新規作業では、対象変更がこの既知失敗に関係するかを明示し、関係しない場合は失敗を増やさないこと。
