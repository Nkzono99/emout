# AGENTS.md

> **やりとりの言語: 日本語。** ユーザーとの対話・質問・進捗報告・PR 説明など、人間に向けて出力するテキストはすべて日本語で書くこと。コード・ファイル名・コミットメッセージ本文・コードコメント・ログ出力・docstring などは従来どおり英語のままで構わない（既存スタイルに合わせる）。ユーザーが英語で話しかけてきた場合のみ英語に切り替えてよい。

このファイルは、このリポジトリで Codex/LLM エージェントが開発・保守を行うための実務ガイドです。Claude Code 固有の補足は `CLAUDE.md` を参照してください（両ファイルはこのセクション構成で揃えてあります）。

## 1. プロジェクト概要

- `emout` は、EMSES の出力ファイル（主に `*.h5`, `plasma.inp`, `plasma.toml`, `icur`, `pbody`）を Python で読み込み、解析・可視化するライブラリです。
- 公開入口は `emout.Emout`（`emout/__init__.py`）です。
- コア機能は以下に分かれます。
  - I/O と Facade: `emout/emout/facade.py`, `emout/emout/io/`
  - データモデル: `emout/emout/data/`
  - 可視化: `emout/plot/`（特に `emout/plot/surface_cut/` のメッシュ境界描画 API）
  - 境界モデル: `emout/emout/boundaries.py`（MPIEMSES3D の `finbound` / legacy 境界を Python から扱う）
  - 入力パラメータ・単位系: `emout/utils/emsesinp.py`, `emout/utils/units.py`, `emout/emout/units.py`
  - 実験機能（依存追加あり）: `emout/emout/backtrace/`, `emout/distributed/`

## 2. まず読むファイル

- `README.md` / `README.en.md`（利用者向け API と使用例）
- `pyproject.toml`（依存関係・Python バージョン・配布設定）
- `tests/conftest.py`（最小データセットの作り方）
- `tests/data/test_data.py`（データアクセスの期待仕様）
- `tests/plot/test_contour3d.py`（3D 可視化 API の期待仕様）
- `tests/plot/test_surface_cut_mesh.py`（明示メッシュサーフェスと `Data3d.plot_surfaces` の期待仕様）
- `tests/test_boundaries.py`（`data.boundaries` / 境界クラスの期待仕様）

## 3. 参照すべき外部ドキュメント

- `/home/b/b36291/large0/Github/MPIEMSES3D/docs/Parameters.md` / `Parameters.en.md`
  - `&ptcond` の `boundary_type` / `boundary_types`（finbound / 複合モード）と各幾何形状のパラメータ一覧がある。`data.boundaries` を触るときは必ず参照。
- `/home/b/b36291/large0/Github/MPIEMSES3D/src/physics/collision/surfaces.F90` / `objects.F90`
  - legacy `*-hole` モード（`rectangle-hole`, `cylinder-hole` など）が実際にどのスカラ／配列インデックス（`xlrechole(1)`, `zlrechole(2)` など）を読むかの根拠コード。複合モード内で legacy 名を使うケースの実装もここにある。

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

# TOML 系のこわれた統合テストを除外して実行（通常ローカル開発用）
pytest -q --ignore=tests/utils/test_toml_converter.py --ignore=tests/utils/test_toml_integration.py

# ドキュメントビルド
sphinx-build -b html docs/source docs/build/html

# パッケージビルド
python -m build
```

## 6. 実装時の必須ルール

- **互換性を最優先にする。**
  - `Emout.__getattr__` の動的解決（`p{species}`, `r[eb][xyz]`, `{name}{axis1}{axis2}`）は既存ユーザーコードに直結するため破壊しない。
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

## 7. surface_cut / メッシュ境界 API の設計メモ

`emout/plot/surface_cut/mesh.py` に置く `MeshSurface3D` サブクラスは、MPIEMSES の各幾何形状と 1 対 1 の関係で設計されている。新しい形状を追加する場合は以下のパターンを踏む。

- ベースは `MeshSurface3D`（抽象クラス）。実装すべきは `mesh() -> (V, F)` だけ。
- 共通ヘルパ（同ファイル内）：
  - `_orthonormal_frame(axis)` — `axis` から `(axis_unit, e1, e2)` を作る
  - `_center_to_3vec(center)` — 2 or 3 要素 → 長さ 3 の NumPy 配列
  - `_axial_range(length, tmin, tmax)` — 長さ指定・範囲指定の両方に対応
  - `_resolve_theta_range` / `_sample_theta` — 角度方向の部分範囲サポート
  - `_plane_mesh(points, expected_normal, wrap_u)` — `(nv, nu, 3)` グリッドを三角形化
  - `_disc_mesh` / `_annulus_mesh` / `_rect_with_hole_mesh` — 円板・環・穴付き矩形の個別ヘルパ
  - `_combine_meshes(meshes)` — `(V, F)` タプルの連結
- `MeshSurface3D.__add__` で `a + b` が `CompositeMeshSurface` になる。複数形状をまとめたいときは `+` でつなぐか `CompositeMeshSurface([...])` を使う。
- `MeshSurface3D.render(**style_kwargs)` は `emout.plot.surface_cut.RenderItem` を返す。`plot_surfaces` に渡す形式になる。

## 8. data.boundaries API の設計メモ

`emout/emout/boundaries.py` が `data.boundaries[i]` / `data.boundaries.mesh()` のすべてを担う。

- 入口は `BoundaryCollection(inp, unit)`。`inp` は `emout.utils.InpFile`、`unit` は `emout.utils.Units`（無い場合は `None` を許容するが `use_si=True` はエラー）。
- `BoundaryCollection._build` は 2 分岐する:
  1. `inp.boundary_type == "complex"`（finbound） → `boundary_types(*)` 配列を走査し、各エントリに対応する `Boundary` サブクラスを作る。
  2. `inp.boundary_type` が `_LEGACY_SINGLE_BODY_TYPES`（`flat-surface`, `rectangle-hole`, `cylinder-hole`）のいずれか → 単一 `Boundary` を作る。
- サポート対象の MPIEMSES 境界型は `_BOUNDARY_CLASS_MAP` に登録する。未登録の型は `collection.skipped` に理由付きで残る。
- f90nml の疎配列を読むには `_get_scalar(pt, name, ib_fortran)` / `_get_vector(pt, name, ib_fortran)` を使う。Fortran は 1-indexed なので `ib_fortran = python_index + 1`。
- 2D 配列（例: `sphere_origin(3, nbt)`）は f90nml で `outer_list[ib - start][component]` の形に展開される。`start_index` は `[None, start_for_dim2]` になっていることが多い。
- `HollowCylinderMeshSurface` は **矩形スラブ + 円筒穴** を表すクラスで、MPIEMSES の `disk` とは別物。`disk` には `DiskMeshSurface`、平面＋穴の薄板には `PlaneWithCircleMeshSurface` を使う。
- legacy `rectangle-hole` / `cylinder-hole` の穴境界は Fortran の `surfaces.F90` の `add_rectangle_hole_surface` / `add_cylinder_hole_surface` に合わせて読む:
  - `xl = xlrechole(1)`, `xu = xurechole(1)`, `yl = ylrechole(1)`, `yu = yurechole(1)`, `zl = zlrechole(2)`, `zu = zssurf`
  - これは Fortran 側が明示的にそうしているので、Python 側も同じ慣習に従う。

新しい境界型の追加手順:
1. `emout/plot/surface_cut/mesh.py` に必要ならメッシュクラスを追加し、`__all__` と `emout/plot/surface_cut/__init__.py` にも追記する。
2. `emout/emout/boundaries.py` に `Boundary` サブクラスを追加し、`_build_params(use_si)` でパラメータを読む／`_build_mesh(params)` でメッシュクラスを生成する。
3. `_BOUNDARY_CLASS_MAP` に登録する（legacy なら `_LEGACY_SINGLE_BODY_TYPES` にも）。
4. `tests/test_boundaries.py` に、`InpFile` を一時ファイルから組み立てるテストを追加する（他のテストを雛形として流用する）。

## 9. 変更時チェックリスト

- 変更した機能に対応するテストを追加・更新したか。
- `pytest -q --ignore=tests/utils/test_toml_converter.py --ignore=tests/utils/test_toml_integration.py` を実行し、既知失敗以外の赤を増やしていないか確認したか。
- 公開 API や挙動を変えた場合、`README.md` / `README.en.md` と `docs/source/` を更新したか。
- optional 依存機能（`vdsolverf`, `dask`, `scikit-image`, `pyvista`）を触る場合、依存未導入時に import で壊れないか確認したか。
- `emout.plot.surface_cut` の `viz.py` 経由の機能（`plot_surfaces` など）は `scikit-image` / `matplotlib` 必須。`import` を守るガードを壊さないこと。

## 10. 現在のテストベースライン（2026-04-09 更新）

対象: `pytest -q`

結果: **172 passed**（`toml2inp` 未インストール環境では 19 件が skipped、153 passed）

過去の既知失敗は 2026-04-09 に解消済み:
- `tests/utils/test_toml_converter.py` — 削除された `load_toml_as_namelist` / `_convert_v*` のテストを削除し、現存する `TomlData` / `load_toml` のみをテストするよう再構成。
- `tests/utils/test_toml_integration.py` — `shutil.which("toml2inp")` で skipif を掛け、バイナリ未インストール環境では skip されるようにした。インストール済みなら通常通り走る。

## 11. 作業ログ用メモ欄

- surface_cut の `HollowCylinderMeshSurface` は **矩形スラブ + 円筒穴** モデル。以前は円筒 + 円筒穴だったが、2026-04-08 に形状を変更している。古い annular 形状は `DiskMeshSurface` に移した。
- `data.boundaries.mesh()` は `CompositeMeshSurface` を返す。子要素は順序を保って連結される。`children` 属性で各境界のメッシュサーフェスに直接触れる。
- `Data3d.plot_surfaces(surfaces, ...)` は 3D スカラー場スライスに明示メッシュを重ねて描画する便利メソッド。bare な `MeshSurface3D` を渡すと `render()` 相当のデフォルトスタイルで自動ラップされる。
