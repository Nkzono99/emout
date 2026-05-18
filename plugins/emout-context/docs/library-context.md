Lang: [日本語](library-context.md) | [English](library-context.en.md)

# emout Library Context

この文書は、plugin skill が emout 利用者を案内するときの共通前提をまとめます。詳細な API は同梱 `references/` の guide を優先してください。

## 基本モデル

- 公開入口は `emout.Emout`。
- 典型的な初期化は `data = emout.Emout("output_dir")`。
- 入力ファイルと出力ディレクトリが分かれている場合は `emout.Emout(input_path="/path/to/plasma.toml", output_directory="output_dir")`。
- 継続出力を結合する場合は `emout.Emout("output_dir", ad="auto")`。
- グリッドデータのスライス軸順序は `(t, z, y, x)`。

## よく使う属性

| 属性 | 内容 |
| --- | --- |
| `data.phisp` | 電位の時系列グリッドデータ |
| `data.nd1p` | 種1 数密度 |
| `data.j1x`, `data.j1y`, `data.j1z` | 種1 電流密度成分 |
| `data.j1xy`, `data.j1xyz` | 自動結合された 2D / 3D ベクトルデータ |
| `data.icur`, `data.pbody` | テキスト診断出力を読んだ `pandas.DataFrame` |
| `data.inp` | `plasma.inp` / `plasma.toml` 由来の入力パラメータ |
| `data.toml` | TOML 入力がある場合の structured parameter |
| `data.unit` | EMSES と SI の単位変換 |
| `data.boundaries` | finbound 境界の Python オブジェクト |

## 単位変換

SI 変換は、入力ファイルに単位変換メタデータがある場合に利用できます。

```python
data.unit.v.trans(1.0)       # SI -> EMSES
data.unit.v.reverse(1.0)     # EMSES -> SI
data.phisp[-1].val_si        # SI 値の ndarray
```

単位変換メタデータがない出力では、`val_si` を使えると断定せず、まず入力ファイルの `!!key` または `[meta.unit_conversion]` を確認します。

## 可視化の考え方

- 1D/2D は `plot()`、`cmap()`、`contour()` を基本にする。
- 時系列は `gifplot()` で GIF/HTML にできる。
- 3D は `plot3d()` を使い、必要なら `pip install "emout[pyvista]"` を案内する。
- 境界は `data.boundaries` から `mesh()` を作り、`plot_surfaces` などに渡す。
- HPC では `emout server start` と `Emout.remote()` / `remote_figure()` で計算ノードに処理を寄せられる。
- 大規模可視化 script では、`RemoteSession` を直接作るより `Emout.remote()`、`remote_scope()`、`remote_figure()`、`RemoteFigure` を使う。`RemoteSession` は共有 Dask Actor の内部名として説明する。

## 利用者支援の境界

emout は生成済み EMSES 出力の読み込み・解析・可視化ライブラリです。既存の analysis script の改善や、自然言語の依頼からの可視化 script 作成は emout plugin の担当範囲です。MPIEMSES3D の入力パラメータ設計や simulator 実行失敗は、MPIEMSES3D context plugin の担当範囲です。
