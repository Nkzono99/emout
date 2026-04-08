# CLAUDE.md

> **やりとりの言語: 日本語。** ユーザーとの対話・質問・進捗報告・コミットメッセージのレビューコメントなど、人間に向けて出力するテキストはすべて日本語で書くこと。コード・ファイル名・コミットメッセージ本文・コードコメント・ログ出力・docstring などは従来どおり英語のままで構わない（既存スタイルに合わせる）。ユーザーが英語で話しかけてきた場合のみ英語に切り替えてよい。

Claude Code 固有の作業ガイドです。汎用的な開発ガイドは `AGENTS.md` にまとまっています（セクション番号は対応させてあります）。このファイルは `AGENTS.md` を前提に **Claude Code の harness をどう使うか** を上乗せする形で書いています。

## 1. プロジェクト概要

→ `AGENTS.md §1` を参照。

## 2. まず読むファイル

→ `AGENTS.md §2` を参照。加えて Claude Code セッションで毎回頭に入れておくと良いもの:

- `tests/test_boundaries.py` — `data.boundaries` の挙動を再現するのに `InpFile` を一時ファイルから組む手順がある。F90 namelist の疎配列アクセスの test fixture としても有用。
- `emout/plot/surface_cut/mesh.py` — メッシュクラスがすべて一ファイルにあるので、類似クラスを追加する時の雛形として最短距離。
- `.claude/skills/` — このリポジトリ固有の skill 群。`add-mesh-surface` / `add-boundary` / `run-tests` / `harness-improve` が使える。

## 3. 参照すべき外部ドキュメント

→ `AGENTS.md §3` を参照。Claude Code から調査する場合は `Explore` エージェントに MPIEMSES3D のパスを渡すと深く調べられる。例:

```
Agent(
  description="Extract finbound param spec",
  subagent_type="Explore",
  prompt="..."
)
```

`/home/b/b36291/large0/Github/MPIEMSES3D/` 以下は `Read`/`Grep`/`Glob` でもそのまま触れる（permission が通っている）。

## 4. 開発環境セットアップ

→ `AGENTS.md §4` を参照。

## 5. よく使うコマンド

→ `AGENTS.md §5` を参照。Claude Code では以下の形で `Bash` から呼ぶ（`TOML` 系の統合テストが壊れているため通常は除外する）:

```bash
python -m pytest tests/ -q \
  --ignore=tests/utils/test_toml_converter.py \
  --ignore=tests/utils/test_toml_integration.py
```

## 6. 実装時の必須ルール

→ `AGENTS.md §6` を参照。Claude Code で作業する時に特に踏みがちな落とし穴:

- **symlink を壊さない**: `CLAUDE.md` / `AGENTS.md` は独立ファイルとして併存させている（元は symlink だった）。両方を更新する場合は両ファイルを個別に編集する（片方だけ更新して symlink 復活させない）。
- **Pylance の "unused import/variable" 警告は新規追加の瞬間だけ出ることが多い**。その後実際に使えば警告は消える。過剰反応して import を削除しないこと。
- **`f90nml` の疎配列は 2D と 1D で `start_index` の形が違う**。`_get_scalar` / `_get_vector` ヘルパを使うか、直接触るなら以下を想定:
  - 1D: `start_index[name] == [start_for_dim1]`
  - 2D: `start_index[name] == [None, start_for_dim2]`（dim1 は完全記述、dim2 が疎）
  - この違いで boundary 境界クラスの実装が分岐している。

## 7. surface_cut / メッシュ境界 API の設計メモ

→ `AGENTS.md §7` を参照。Claude Code から新しいメッシュクラスを足す場合は `Skill(skill="add-mesh-surface")` を使うと雛形とチェックリストが即座に出る。

## 8. data.boundaries API の設計メモ

→ `AGENTS.md §8` を参照。新しい finbound 境界型を足す場合は `Skill(skill="add-boundary")` を使う。スキルの中に MPIEMSES docs の参照パスと、`BoundaryCollection._build` の分岐ポイントが書いてある。

## 9. 変更時チェックリスト

→ `AGENTS.md §9` を参照。Claude Code 向けの追加:

- `TaskCreate` で作業を 3 ステップ以上に分解する価値があるかを最初に判断する。単発の編集は TaskCreate 不要。
- 編集対象に手を入れる前に **必ず `Read` で現状を確認** する（これは harness の約束事だが、Edit/Write は直前の Read を要求する）。
- 変更をコミットするのは **ユーザーが明示的に頼んだ時のみ**。自動コミットはしないこと。今セッションのように `適宜コミットして良い` と許可されている場合のみ、自然な境界で切ってコミットする。

## 10. 現在のテストベースライン

→ `AGENTS.md §10` を参照。

## 11. 作業ログ用メモ欄

→ `AGENTS.md §11` を参照。

## 12. Claude Code 固有の skill / agent / settings

### 用意済み skill（`.claude/skills/`）

- **`harness-improve`** — この harness 自体（`CLAUDE.md`, `AGENTS.md`, `.claude/skills/*`, `.claude/agents/*`, `.claude/settings.json`）をレビュー・改善するメタスキル。自分自身（`harness-improve` スキル）も改善対象に含む。セッション終了直前、harness の指示がズレ始めたと感じた時、もしくは参考リポジトリ／URL を引数に渡してベンチマークしたい時に呼ぶ。
- **`run-tests`** — このプロジェクトのテストベースラインを走らせるショートカット。既知の壊れた TOML テストを除外して実行する。
- **`add-mesh-surface`** — `emout/plot/surface_cut/mesh.py` に新しい `MeshSurface3D` サブクラスを追加する手順の雛形。`_plane_mesh` 系ヘルパの使い分け・`__all__` の更新・テスト追加までカバー。
- **`add-boundary`** — `emout/emout/boundaries.py` に MPIEMSES の新しい境界型を追加する手順の雛形。`_BOUNDARY_CLASS_MAP` 登録・`use_si` 対応・f90nml 疎配列アクセスまでカバー。

使い方: `Skill(skill="<name>")` または `/<name>` で呼び出す。引数を取る skill は `/<name> <args>` の形を受け取る。

### 用意済み agent（`.claude/agents/`）

- **`finbound-investigator`** — MPIEMSES3D の `docs/Parameters.md` と `src/physics/collision/*.F90` を読み込み、指定された `boundary_type` / `boundary_types` のパラメータ仕様や Fortran 実装を抽出する調査役。`data.boundaries` を拡張する時の背景調査に使う。

使い方: `Agent(subagent_type="finbound-investigator", prompt="...")`。

### 設定ファイル（`.claude/`）

- **`settings.json`** — 共有・コミット対象の権限ベースライン。`allow` は広く（Read/Edit/Write/Bash/Skill/Agent などをそのまま許可）、`ask` は空、`deny` は最小限（`sudo`, `rm -rf /` 系のみ）。コントリビュータは clone した時点でこの設定を引き継ぐ。
- **`settings.local.json`** — 個人ユーザー固有の追加権限。`.gitignore` 済み。手で広げる場合はここに追記する（`settings.json` の方は不必要に変更しないこと）。

settings の編集後は `python -c "import json; json.load(open('.claude/settings.json'))"` で構文を検証する。
