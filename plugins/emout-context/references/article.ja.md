# 論文用データの記録・再生（`EMOUT_ARTICLE_MODE`）

article 機能は、図作成スクリプトが実際に使った最小限のグリッドデータだけを記録し、
元の巨大な EMSES シミュレーション出力なしで同じ図を再生成するための仕組みです。
論文投稿時の公開データ、補足資料、共同研究者への再現用 bundle を作る用途を想定しています。

通常の解析スクリプトはそのまま使い、環境変数または `Emout()` の引数で
`normal` / `record` / `replay` を切り替えます。

## いつ使うか

- `plot()` や `to_numpy()` で使ったスライスだけを公開したい。
- Jupyter notebook や 1 本の script で作った複数 figure を 1 つの公開 bundle にまとめたい。
- 複数の simulation output を比較する図を、別環境でも再現できる形で保存したい。
- `plasma.inp` / `plasma.toml` / 境界メッシュ / 小さな診断ファイルも含めて再現したい。

particle データ、backtrace、remote execution そのものは article replay bundle には含まれません。
replay は「記録済みのグリッドスライスと入力メタデータから図を再生成する」用途に絞られています。

## クイックスタート

図作成スクリプトは通常どおり書きます。スライスの軸順序は `(t, z, y, x)` です。

```python
import emout

data = emout.Emout("output_dir")
ymid = data.inp.ny // 2

data.phisp[-1, :, ymid, :].plot(cmap="viridis")
arr = data.ex[-1, :, ymid, :].to_numpy()
```

同じスクリプトを、環境変数で record / replay に切り替えます。

```bash
# Normal run
python figure.py

# Record article data
EMOUT_ARTICLE_MODE=record \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
python figure.py

# Replay from recorded article data
EMOUT_ARTICLE_MODE=replay \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
python figure.py
```

引数で明示することもできます。

```python
data = emout.Emout(
    "output_dir",
    article_mode="record",
    article_records_path="article-records",
)
```

## 保存されるもの

record mode では、`plot()` と `to_numpy()` が materialize したデータだけを保存します。
例えば `data.phisp[-1, :, ymid, :].plot()` は、その 2D スライスだけを `data.h5` に保存します。
同じ field と selector が複数回使われても、重複して保存されません。

`Data3d.plot_surfaces()` は 3D field を使うため、そのままでは公開データが大きくなりがちです。
article record mode では `bounds` が渡された場合、`data.phisp[-1].plot_surfaces(..., bounds=bounds)`
のような呼び出しを検出し、描画範囲に 1 grid padding を加えた 3D ROI だけを保存します。
保存済み ROI は元の global 座標に合わせて replay されるため、境界メッシュや mask surface との位置関係は保たれます。
`bounds` を渡さない場合、必要範囲を判断できないため 3D 全体が保存対象になります。

時間平均も公開データ用に記録できます。`data.phisp[-20:].mean()` は time axis の lazy reduction として扱われ、
平均に使った 20 timestep をそのまま保存するのではなく、平均後のデータだけを保存します。
`plot_surfaces(..., bounds=bounds)` に接続した場合は、bounds 周辺の ROI だけを各 timestep から読み、
平均後 ROI を保存します。

```python
field = data.phisp[-20:].mean()
field.plot_surfaces(data.boundaries, bounds=bounds, mode="cmap")
```

`mean()` が返す field からも `field.inp`、`field.unit`、`field.boundaries` を参照できます。
そのため、図作成関数が `data.inp` や `data.boundaries` を読む場合でも、平均 field を渡して同じ構造で使えます。

| ファイル | 内容 | 用途 |
| --- | --- | --- |
| `manifest.json` | 記録した field、selector、shape、slice axes、単位情報 | replay 時に requested slice と保存済み slice を照合する |
| `data.h5` | 記録済みの NumPy 配列 | `Data1d` / `Data2d` / `Data3d` として再構築する |
| `source.json` | 元 simulation path、basename、recorded files の hash | 別環境で source を対応付け、改変を検出する |
| `plasma.inp` | 入力ファイル | `data.inp`、単位変換、境界メッシュ再構築 |
| `plasma.toml` | TOML 入力ファイル | `data.toml` の再現 |
| `icur`, `pbody` | 小さな診断ファイル（存在する場合） | `data.icur` / `data.pbody` の再現 |

`data.h5` 内の dataset は HDF5 gzip 圧縮で保存されます。replay 側では HDF5 が透過的に展開するため、
通常の `plot()` / `to_numpy()` の使い方は変わりません。

## ディレクトリ構造

基本構造は `records-path/datasets/<source>/<article-name>/` です。
`article_name` を省略すると `default` になります。

```text
article-records/
└── datasets/
    └── output_dir-012345abcd/
        ├── source.json
        └── default/
            ├── manifest.json
            ├── data.h5
            ├── plasma.inp
            ├── plasma.toml
            ├── icur
            └── pbody
```

`<source>` は通常、source directory の basename と絶対パス hash から作られます。
別環境では絶対パスが変わるため、replay はまず `<source>` の直接一致を試し、
見つからない場合は `source.json` の basename で対応付けます。

## 複数 figure と複数 simulation

`EMOUT_ARTICLE_NAME` を指定すると、`fig1` や `fig2` のように figure ごとに分けられます。
指定しない場合は `default` にまとまるため、notebook や 1 本の script で作るすべての figure を
1 つの bundle にできます。

```bash
EMOUT_ARTICLE_MODE=record \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
EMOUT_ARTICLE_NAME=fig1 \
python figure.py
```

同じ `article_name` で `Emout()` を作り直すと、既存 bundle に未記録スライスだけを追記します。
Jupyter のセル再実行や、関数内で `Emout()` を作り直す script でも同じ bundle を使えます。

複数の simulation output を扱う場合は source ごとに保存先が分かれます。
同じ basename の output が複数ある場合は、公開先でも同じ source を選べるように
`article_source_name` を指定してください。

```python
data = [
    emout.Emout("case_a/output", article_source_name="case_a"),
    emout.Emout("case_b/output", article_source_name="case_b"),
]
```

record / replay の両方で同じ `article_source_name` を使うと、絶対パスが変わっても
`article-records/datasets/case_a/default/` のような安定した保存先を使えます。

## archive と公開データサイズ

`article_archive` を有効にすると、各 bundle を archive として自動保存します。

```python
data = emout.Emout(
    "output_dir",
    article_mode="record",
    article_records_path="article-records",
    article_archive="zip",
)
```

```bash
EMOUT_ARTICLE_MODE=record \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
EMOUT_ARTICLE_ARCHIVE=zip \
python figure.py
```

| 指定 | 作成される archive |
| --- | --- |
| `article_archive=True` / `EMOUT_ARTICLE_ARCHIVE=1` | `<article-name>.tar.gz` |
| `article_archive="tar.gz"` / `EMOUT_ARTICLE_ARCHIVE=tar.gz` | `<article-name>.tar.gz` |
| `article_archive="zip"` / `EMOUT_ARTICLE_ARCHIVE=zip` | `<article-name>.zip` |

replay 時は展開済み directory がなくても、対応する `.tar.gz` または `.zip` があれば自動展開します。
zip はアップロード先が `.tar.gz` を受け付けない場合や、Windows で展開しやすい形式にしたい場合に便利です。

## replay でできること

replay mode の `emout.Emout()` は、元の HDF5 出力ではなく記録済み bundle を読む proxy を返します。
記録済みスライスであれば、通常の `Data` と同じように `plot()` や `to_numpy()` が使えます。

```python
data = emout.Emout(
    "output_dir",
    article_mode="replay",
    article_records_path="article-records",
)

data.phisp[-1, :, ymid, :].plot()
arr = data.ex[-1, :, ymid, :].to_numpy()
```

vector alias も、必要な component が記録されていれば使えます。

```python
data.exz[-1, :, ymid, :].plot()
```

入力メタデータと境界も replay できます。

```python
data.boundaries.plot()
data.phisp[-1].plot_surfaces(data.boundaries, bounds=bounds)
icur = data.icur
pbody = data.pbody
```

未記録のスライスにアクセスすると例外になります。これは公開 bundle に図の再現に必要なデータが
含まれているかを確認するための挙動です。

## 設定一覧

| 引数 | 環境変数 | 既定値 | 意味 |
| --- | --- | --- | --- |
| `article_mode` | `EMOUT_ARTICLE_MODE` | `normal` | `normal` / `record` / `replay` を切り替える |
| `article_records_path` / `records_path` | `EMOUT_ARTICLE_RECORDS_PATH` / `EMOUT_RECORDS_PATH` | なし | bundle の root directory |
| `article_name` | `EMOUT_ARTICLE_NAME` | `default` | figure や notebook 単位の bundle 名 |
| `article_source_name` | `EMOUT_ARTICLE_SOURCE_NAME` | なし | 複数 source を別環境でも安定して識別する名前 |
| `article_archive` | `EMOUT_ARTICLE_ARCHIVE` | なし | `tar.gz` または `zip` archive を作成する |

## よくあるハマりどころ

- `record` / `replay` では `article_records_path` が必須です。環境変数または引数で指定してください。
- 複数 source が同じ basename（例: `case_a/output`, `case_b/output`）を持つ場合は、`article_source_name` を指定してください。
- `np.asarray(data.phisp[...])` よりも `data.phisp[...].to_numpy()` を使うと、記録対象であることが明示されます。
- `plot_surfaces()` で公開データを小さくしたい場合は `bounds` を渡してください。`data.phisp[-1, :, ymid, :]` のような 2D slice はそのまま 2D として保存されますが、`data.phisp[-1]` のような 3D selection は `bounds` なしでは 3D 全体になり得ます。
- `data.phisp[-20:].mean()` の既定は time axis 平均です。空間方向を平均したい場合は、意図が分かるように `mean(axis="x")` などを明示してください。
- record mode の remote rendering では worker 側で使った slice が記録されます。replay では particle、backtrace、remote execution は提供されません。必要なら元データで実行してから、可視化に使うグリッドスライスを record してください。
- 元 script がランダムな描画設定や外部ファイルに依存する場合、article bundle だけでは完全再現できません。図作成 script も一緒に公開してください。

## 関連クラス

詳細なシグネチャは API リファレンス（`emout.article` と `emout.core.facade`）を参照してください。

- `emout.Emout` — `article_mode` / `article_records_path` / `article_name` などの公開入口
- `ArticleRecorder` — record mode で slice、metadata、archive を保存する内部クラス
- `ArticleReplayEmout` — replay mode で記録済み bundle を読む proxy
