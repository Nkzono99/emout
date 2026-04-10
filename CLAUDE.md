# emout 開発ガイド

> **やりとりの言語: 日本語。** コード・コミットメッセージ・docstring は英語。ユーザーが英語で話しかけた場合のみ英語に切り替える。

> **`CLAUDE.md` と `AGENTS.md` は同一内容。** 片方を変更したらもう一方にコピーすること。

## プロジェクト概要

`emout` は EMSES シミュレーション出力（`*.h5`, `plasma.inp`, `plasma.toml`）を Python で読み込み・解析・可視化するライブラリ。公開入口は `emout.Emout`。

| モジュール | 役割 |
|-----------|------|
| `emout/core/facade.py`, `emout/core/io/` | I/O & Facade |
| `emout/core/data/` | データモデル |
| `emout/plot/` (`surface_cut/` 含む) | 可視化・メッシュ境界描画 |
| `emout/core/boundaries/` | MPIEMSES finbound 境界モデル |
| `emout/utils/emsesinp.py`, `emout/utils/units.py` | 入力パラメータ・単位系 |
| `emout/distributed/` | リモート実行 (Dask, 共有セッション) |
| `emout/core/backtrace/` | バックトレース (実験的) |

## セットアップ & テスト

```bash
pip install -e ".[dev]"   # 開発インストール (pytest, ruff, pre-commit)
pre-commit install        # git commit 時に ruff lint+format を自動実行
pytest -q                 # 全テスト
```

テストベースライン: **1321 passed**（`toml2inp` 未導入時は一部 skipped）

## MPIEMSES 外部資料

境界 (`data.boundaries`) を触るときは必ず参照:

- `/home/b/b36291/large0/Github/MPIEMSES3D/docs/Parameters.md` — `boundary_types` パラメータ一覧
- `/home/b/b36291/large0/Github/MPIEMSES3D/src/physics/collision/surfaces.F90` / `objects.F90` — Fortran 実装

## ドメインルール

<important>
後方互換・軸順序・単位変換・ドキュメント管理のルールは `.claude/rules/` に分離されている。
すべてのセッションで自動ロードされるため CLAUDE.md に書く必要はないが、
実装時に必ず従うこと。特に `backward-compat.md` は最重要。
</important>

## 用意済みツール

| 種類 | 名前 | いつ使うか |
|------|------|-----------|
| skill | `run-tests` | テストベースラインの確認 |
| skill | `add-mesh-surface` | `mesh.py` に新メッシュクラスを追加するとき |
| skill | `add-boundary` | `boundaries/` に新境界型を追加するとき |
| skill | `release` | バージョンを上げてリリースするとき |
| skill | `harness-improve` | この harness 自体を改善するとき |
| agent | `finbound-investigator` | MPIEMSES3D のパラメータ仕様を Fortran ソースから調査するとき |

## CLI

`emout server start/stop/status` — Dask スケジューラ管理。`~/.emout/server.json` に接続情報を保存。
