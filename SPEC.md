# SPEC.md

この文書は `emout` の現行実装と README/テストから整理した、保守開発用の技術仕様です。

## 1. 目的とスコープ

- 目的:
  - EMSES 出力（`*.h5`, `plasma.inp`, `icur`, `pbody`）を Python で一貫して扱う。
  - 単位変換付きでデータ参照・可視化を可能にする。
- スコープ:
  - グリッドデータの読込/連結/スライス
  - 粒子データの時系列読込
  - 入力パラメータの参照・保存
  - 2D/3D 可視化
  - 実験機能として backtrace と分散実行

## 2. 入出力契約

### 2.1 ディレクトリ構成

- 主ディレクトリ:
  - `plasma.inp`
  - `{name}00_0000.h5`（例: `phisp00_0000.h5`, `ex00_0000.h5`）
  - 任意で `icur`, `pbody`
- 追加ディレクトリ:
  - `append_directories=[...]` で明示指定
  - `ad="auto"` の場合、`<main>_2`, `<main>_3`, ... を探索

### 2.2 HDF5 期待構造

- グリッド:
  - 先頭グループ名は物理量名（例: `phisp`）
  - 配下キーは時刻（例: `0000`, `0001`, ...）
  - 各データセットは `(nz, ny, nx)`
- 粒子:
  - 1 ファイル 1 成分（`x`, `y`, `z`, `vx`, `vy`, `vz`, `tid`）
  - ファイル名パターン: `p{species}{comp}(e?){seg}_{part}.h5`

### 2.3 単位変換キー

- `plasma.inp` 先頭行に以下があれば SI 変換有効:
  - `!!key dx=[...],to_c=[...]`
- 無い場合は `unit is None` として扱う（変換は行わない）。

## 3. 公開 API 契約

### 3.1 `Emout` Facade

- クラス: `emout.Emout`
- コンストラクタ:
  - `directory`
  - `append_directories` または `ad`
  - `inpfilename`（既定: `plasma.inp`）
- プロパティ:
  - `directory`, `append_directories`, `inp`, `unit`
  - `icur`, `pbody`
  - `backtrace`（実験機能）

### 3.2 動的属性解決

`Emout.__getattr__` の解決規則:

1. `p([1-9])` に一致: 粒子シリーズ (`ParticlesSeries`) を返す
2. `r[eb][xyz]`: relocation 済み HDF5 を生成して読み込む
3. `(.+)([xyz])([xyz])`: 2 成分ベクトルとして `VectorData2d` を返す
4. その他: `GridDataSeries` を返す

この順序は互換性上の重要契約。

### 3.3 `GridDataSeries`/`Data*`

- `GridDataSeries[i]` は `Data3d`
- `GridDataSeries[i:j]` は `Data4d`
- 追加スライスで `Data2d`/`Data1d` に降格
- 軸順序は `t, z, y, x` を前提
- 追加ディレクトリは `chain()` で時系列連結

### 3.4 `InpFile`

- `inp["group"]["name"]` と `inp["name"]`（グループ省略）をサポート
- `inp.group.name` 形式の属性アクセスをサポート
- `save()` 時に単位キーを先頭行へ書き戻せる

## 4. 単位系仕様

- 単位マッピングは `emout/emout/units.py` の `build_name2unit_mapping` で定義。
- `t` は `ifdiag * dt` を反映する変換器を使用。
- 空間軸は `unit.length`、電磁場や密度は名前に応じた `UnitTranslator` を割当。
- 変換キー欠損時は単位変換機能を無効化しても API 呼び出しは継続可能であること。

## 5. 可視化仕様

### 5.1 2D/アニメーション

- `Data.plot()` / `Data.gifplot()` 系 API を提供。
- ベクトルデータは `VectorData2d` として stream 表示可能。

### 5.2 3D 等値面 (`contour3d`)

- 入力ボリュームは `(nz, ny, nx)`。
- `dx` は scalar または `(dx, dy, dz)`。
- `bounds_xyz` または `roi_zyx` のどちらか一方で ROI 指定。
- `clabel_*` で等値面ラベルの桁/指数表記を制御。

## 6. 実験機能仕様

### 6.1 Backtrace

- `vdsolverf` 依存。
- `get_backtrace(s)` / `get_probabilities` で軌道・確率を計算。
- 返却値は `BacktraceResult`, `MultiBacktraceResult`, `ProbabilityResult`。

### 6.2 分散実行

- `emout.distributed` は Python 3.10+ で有効。
- Dask Scheduler/Worker を SLURM 前提で起動するラッパーを提供。

## 7. 品質保証仕様

- テスト実行:
  - `pytest -q`
  - 変更箇所の対象テストを優先実行
- ドキュメント:
  - `sphinx-build -b html docs/source docs/build/html`
- 重要回帰ポイント:
  - 動的属性解決の後方互換
  - 軸順序 (`tzyx`, `zyx`) とスライス挙動
  - 単位変換キー有無の両ケース

## 8. 既知ギャップ（2026-03-08 時点）

`pytest -q` は 3 件失敗。

- `emout.data` エクスポート互換性の不足
- `name2unit` の `nd12p` パターン不一致

保守方針として、これらは優先的に解消し、以後は CI で回帰を防ぐ。
