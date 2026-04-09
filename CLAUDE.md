# emout 開発ガイド

> **やりとりの言語: 日本語。** ユーザーとの対話・質問・進捗報告・コミットメッセージのレビューコメントなど、人間に向けて出力するテキストはすべて日本語で書くこと。コード・ファイル名・コミットメッセージ本文・コードコメント・ログ出力・docstring などは従来どおり英語のままで構わない（既存スタイルに合わせる）。ユーザーが英語で話しかけてきた場合のみ英語に切り替えてよい。

> **`CLAUDE.md` と `AGENTS.md` は同一内容で管理されている。** 変更はどちらか一方に行い、もう一方にもコピーすること。

このファイルは、Claude Code / Codex / その他の LLM エージェントがこのリポジトリで開発・保守を行うための実務ガイドです。

## 1. プロジェクト概要

- `emout` は、EMSES の出力ファイル（主に `*.h5`, `plasma.inp`, `plasma.toml`, `icur`, `pbody`）を Python で読み込み、解析・可視化するライブラリです。
- 公開入口は `emout.Emout`（`emout/__init__.py`）です。
- コア機能は以下に分かれます。
  - I/O と Facade: `emout/emout/facade.py`, `emout/emout/io/`
  - データモデル: `emout/emout/data/`
  - 可視化: `emout/plot/`（特に `emout/plot/surface_cut/` のメッシュ境界描画 API）
  - 境界モデル: `emout/emout/boundaries.py`（MPIEMSES3D の `finbound` / legacy 境界を Python から扱う）
  - 入力パラメータ・単位系: `emout/utils/emsesinp.py`, `emout/utils/units.py`, `emout/emout/units.py`
  - リモート実行（実験的）: `emout/distributed/`（Dask Actor、`remote_figure` コンテキスト、CLI サーバー）
  - バックトレース（実験的）: `emout/emout/backtrace/`

## 2. まず読むファイル

- `README.md` / `README.en.md`（利用者向け API と使用例）
- `pyproject.toml`（依存関係・Python バージョン・配布設定）
- `tests/conftest.py`（最小データセットの作り方）
- `tests/data/test_data.py`（データアクセスの期待仕様）
- `tests/plot/test_contour3d.py`（3D 可視化 API の期待仕様）
- `tests/plot/test_surface_cut_mesh.py`（明示メッシュサーフェスと `Data3d.plot_surfaces` の期待仕様）
- `tests/test_boundaries.py`（`data.boundaries` / 境界クラスの期待仕様。`InpFile` を一時ファイルから組む fixture としても有用）
- `emout/plot/surface_cut/mesh.py`（メッシュクラスが一ファイルにあるので類似クラスの雛形として最短距離）
- `.claude/skills/`（`add-mesh-surface` / `add-boundary` / `run-tests` / `harness-improve` が使える）

## 3. 参照すべき外部ドキュメント

- `/home/b/b36291/large0/Github/MPIEMSES3D/docs/Parameters.md` / `Parameters.en.md`
  - `&ptcond` の `boundary_type` / `boundary_types`（finbound / 複合モード）と各幾何形状のパラメータ一覧がある。`data.boundaries` を触るときは必ず参照。
- `/home/b/b36291/large0/Github/MPIEMSES3D/src/physics/collision/surfaces.F90` / `objects.F90`
  - legacy `*-hole` モード（`rectangle-hole`, `cylinder-hole` など）が実際にどのスカラ／配列インデックス（`xlrechole(1)`, `zlrechole(2)` など）を読むかの根拠コード。

Claude Code から調査する場合は `Explore` エージェントに MPIEMSES3D のパスを渡すと深く調べられる。`/home/b/b36291/large0/Github/MPIEMSES3D/` 以下は `Read`/`Grep`/`Glob` でもそのまま触れる。また `Agent(subagent_type="finbound-investigator", prompt="...")` で境界パラメータの詳細調査を委譲できる。

## 4. 開発環境セットアップ

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

## 5. よく使うコマンド

```bash
# 全テスト
pytest -q

# 対象テストのみ
pytest tests/plot/test_contour3d.py -q
pytest tests/plot/test_surface_cut_mesh.py -q
pytest tests/test_boundaries.py -q
pytest tests/data/test_data.py -q

# ドキュメントビルド
sphinx-build -b html docs/source docs/build/html

# パッケージビルド
python -m build
```

Claude Code からは `Bash` で `python -m pytest tests/ -q` を直接呼ぶか、`Skill(skill="run-tests")` を使う。

## 6. 実装時の必須ルール

- **後方互換性を最優先にする。** 利用者がまずまずいる前提で、公開 API の削除・改名・挙動変更は原則行わない。
  - `Emout.__getattr__` の動的解決（`p{species}`, `r[eb][xyz]`, `{name}{axis1}{axis2}`）は既存ユーザーコードに直結するため破壊しない。
  - 整理が必要な場合は旧名をラッパーとして残し、 `warnings.warn(..., DeprecationWarning, stacklevel=2)` を出してから新実装へ委譲する。少なくとも 1 マイナーリリースは共存させ、削除時期はユーザーに確認する。
  - 「新メソッドを追加して既存メソッドを温存する」パターンは推奨。例: `Data2d.cmap()` / `Data2d.contour()` を `plot(mode=...)` と並存させて追加。
  - 挙動を変えるときは新キーワード引数（デフォルトは旧挙動）として追加し、デフォルトを切り替える前にアナウンスする。
- **軸順序を崩さない。**
  - グリッドデータは基本 `(t, z, y, x)`、3D ボリュームは `(z, y, x)` 前提。
  - `Data3d.axisunits` は末尾が x、`[-1]`/`[-2]`/`[-3]` がそれぞれ x/y/z の `UnitTranslator`。
- **ファイル命名規則を守る。**
  - グリッド: `{name}00_0000.h5`
  - 粒子: `p{species}{comp}(e?){seg}_{part}.h5`
- **単位変換は `plasma.inp` 1 行目の `!!key dx=[...],to_c=[...]` に依存する。**
  - 未設定ケース（`unit is None`）を壊さない。
  - grid→SI は `data.unit.length.reverse(x)`、SI→grid は `.trans(x)`。スカラでも NumPy 配列でも動く。
- **`docs/build/`、`__pycache__/`、一時生成物は原則編集対象外。**
- Pylance の "unused import/variable" 警告は新規追加の瞬間だけ出ることが多い。過剰反応して import を削除しないこと。
- **`f90nml` の疎配列は 2D と 1D で `start_index` の形が違う**。`_get_scalar` / `_get_vector` ヘルパを使うか、直接触るなら以下を想定:
  - 1D: `start_index[name] == [start_for_dim1]`
  - 2D: `start_index[name] == [None, start_for_dim2]`（dim1 は完全記述、dim2 が疎）

## 7. surface_cut / メッシュ境界 API の設計メモ

`emout/plot/surface_cut/mesh.py` に置く `MeshSurface3D` サブクラスは、MPIEMSES の各幾何形状と 1 対 1 の関係で設計されている。新しい形状を追加する場合は `Skill(skill="add-mesh-surface")` で雛形を出すか、以下のパターンを踏む。

- ベースは `MeshSurface3D`（抽象クラス）。実装すべきは `mesh() -> (V, F)` だけ。
- 共通ヘルパ（同ファイル内）：`_orthonormal_frame`, `_center_to_3vec`, `_axial_range`, `_resolve_theta_range` / `_sample_theta`, `_plane_mesh`, `_disc_mesh` / `_annulus_mesh` / `_rect_with_hole_mesh`, `_combine_meshes`
- `MeshSurface3D.__add__` で `a + b` が `CompositeMeshSurface` になる。
- `MeshSurface3D.render(**style_kwargs)` は `RenderItem` を返す。

## 8. data.boundaries API の設計メモ

`emout/emout/boundaries.py` が `data.boundaries[i]` / `data.boundaries.mesh()` のすべてを担う。新しい finbound 境界型を足す場合は `Skill(skill="add-boundary")` を使う。

- 入口は `BoundaryCollection(inp, unit)`。
- `_build` は complex モード（`boundary_types(*)` 走査）と legacy 単独モードの 2 分岐。
- `_BOUNDARY_CLASS_MAP` に登録する。未登録の型は `collection.skipped` に残る。
- f90nml の疎配列は `_get_scalar` / `_get_vector` を使う（Fortran 1-indexed）。
- `HollowCylinderMeshSurface` = 矩形スラブ + 円筒穴。`DiskMeshSurface` とは別物。
- legacy `rectangle-hole` / `cylinder-hole`: `zlrechole(2)` (Fortran index 2) に注意。

新しい境界型の追加手順:
1. `mesh.py` にメッシュクラスを追加 → `__all__` と `__init__.py` にも追記
2. `boundaries.py` に `Boundary` サブクラスを追加
3. `_BOUNDARY_CLASS_MAP` に登録
4. `tests/test_boundaries.py` にテスト追加

## 9. 変更時チェックリスト

- 変更した機能に対応するテストを追加・更新したか。
- `pytest -q` を実行し、グリーンを維持しているか確認したか（現ベースラインは §10 参照）。
- 公開 API を変えるときは §6 の後方互換ルールに従っているか。
- 公開 API や挙動を変えた場合、`README.md` / `README.en.md` と `docs/source/` を更新したか。
- **ドキュメントの日英ペアを同じ PR で両方更新すること。** 対象:
  - `README.md` ⇔ `README.en.md`
  - `CLAUDE.md` ⇔ `AGENTS.md`（同一内容）
  - `docs/source/guide/*.md` ⇔ `docs/source/guide/*.ja.md`（`quickstart` / `plotting` / `animation` / `inp` / `units` / `boundaries` / `distributed` の 7 ペア）
- optional 依存機能を触る場合、依存未導入時に import で壊れないか。
- `emout.plot.surface_cut` の `viz.py` 経由の機能は `scikit-image` / `matplotlib` 必須。

Claude Code 固有:
- `TaskCreate` で作業を 3 ステップ以上に分解する価値があるか最初に判断する。
- 編集前に **必ず `Read` で現状を確認**。
- コミットは **ユーザーが許可した時のみ**。

## 10. 現在のテストベースライン（2026-04-09 更新）

対象: `pytest -q`

結果: **172 passed**（`toml2inp` 未インストール環境では 19 件が skipped、153 passed）

過去の既知失敗は 2026-04-09 に解消済み:
- `tests/utils/test_toml_converter.py` — 削除 API のテストを除去し `TomlData` / `load_toml` のみに再構成。
- `tests/utils/test_toml_integration.py` — `shutil.which("toml2inp")` で skipif を掛けた。

## 11. 作業ログ用メモ欄

- `HollowCylinderMeshSurface` は矩形スラブ + 円筒穴モデル（2026-04-08 変更）。古い annular 形状は `DiskMeshSurface` に移した。
- `data.boundaries.mesh()` は `CompositeMeshSurface` を返す。`children` 属性で個別メッシュに触れる。
- `Data3d.plot_surfaces(surfaces, ...)` は 3D スカラー場に明示メッシュを重ねて描画する。
- `emout/distributed/remote_render.py` に `RemoteSession` (Dask Actor) + proxy クラス群を追加（2026-04-09）。backtrace 計算結果を worker メモリに保持し、可視化パラメータだけ変えて再レンダリングする設計。
- `emout/distributed/remote_figure.py` に `remote_figure()` コンテキストマネージャを追加（2026-04-09）。matplotlib.pyplot をモンキーパッチしてコマンドを記録、worker で一括再生して PNG bytes を返す。
- `emout/cli.py` に `emout server start/stop/status` CLI を追加（2026-04-09）。`~/.emout/server.json` に起動情報を書き出し、スクリプトから `connect()` なしで自動接続する。
- `Data._try_remote_plot()` が Dask session の有無で 3 段階に分岐: (1) `remote_figure` 内ならコマンド記録、(2) session ありならデータ転送モード（スライスだけ転送しローカル描画）、(3) なければ従来ローカル。
- `BoundaryCollection.plot()` で境界メッシュを単体で 3D 表示できるようになった（2026-04-09）。

## 12. 用意済み skill / agent / settings

### skill（`.claude/skills/`）

| skill | 用途 |
| --- | --- |
| `harness-improve` | この harness 自体をレビュー・改善するメタスキル |
| `run-tests` | テストベースラインを走らせるショートカット |
| `add-mesh-surface` | `mesh.py` に新しい `MeshSurface3D` サブクラスを追加する雛形 |
| `add-boundary` | `boundaries.py` に新しい境界型を追加する雛形 |

使い方: `Skill(skill="<name>")` または `/<name>` で呼び出す。

### agent（`.claude/agents/`）

| agent | 用途 |
| --- | --- |
| `finbound-investigator` | MPIEMSES3D の finbound パラメータ仕様を Fortran ソースから調査する |

使い方: `Agent(subagent_type="finbound-investigator", prompt="...")`。

### CLI エントリポイント

`pyproject.toml` の `[project.scripts]` に `emout = "emout.cli:main"` が登録されている。

| コマンド | 用途 |
| --- | --- |
| `emout server start` | Dask スケジューラ + ワーカーを起動。`~/.emout/server.json` に接続情報を保存 |
| `emout server stop` | 停止 |
| `emout server status` | 起動中のアドレス・ワーカー数を表示 |

### settings（`.claude/`）

- `settings.json` — 共有・コミット対象の権限ベースライン。編集後は `python -c "import json; json.load(open('.claude/settings.json'))"` で構文検証する。
- `settings.local.json` — 個人用（`.gitignore` 済み）。
