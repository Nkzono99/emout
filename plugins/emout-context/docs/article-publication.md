Lang: [日本語](article-publication.md) | [English](article-publication.en.md)

# Article Data Publication Workflow

この文書は、論文投稿・公開データ・補足資料向けに emout の article record/replay 機能を案内するときの標準方針です。

## 基本方針

- 通常の可視化 script はできるだけ変えず、`EMOUT_ARTICLE_MODE=record` / `replay` または `Emout(..., article_mode=...)` で切り替える。
- 公開するのは「可視化に使った最小限のグリッドデータ」であり、元の巨大な HDF5 全体ではない。
- 2D plot では `data.phisp[-1, :, ymid, :]` のように先に slice する。
- 3D `plot_surfaces()` では `bounds` を渡し、保存対象を描画範囲周辺の ROI にする。
- 時間平均では `data.phisp[-20:].mean()` を使う。平均に使った全 timestep ではなく、平均後のデータだけを保存する。

## 推奨コード

```python
import emout

data = emout.Emout("output_dir", article_records_path="article-records")

# 2D slice: 保存されるのはこの 2D slice だけ
ymid = data.inp.ny // 2
data.phisp[-1, :, ymid, :].plot()

# Time-mean 3D surface: 各 timestep から bounds ROI だけを読み、平均後 ROI だけを保存
field = data.phisp[-20:].mean()
field.plot_surfaces(data.boundaries, bounds=bounds, mode="cmap")
```

`mean()` が返す field は `field.inp`、`field.unit`、`field.boundaries` を持つため、既存の helper が `data.inp` や `data.boundaries` を読む場合でも、平均 field を渡せます。

```python
def build_items_and_geom(data):
    zs = data.unit.length.reverse(data.inp.zssurf)
    boundaries = data.boundaries.mesh(theta_range=[0, np.pi])
    ...

field = data.phisp[-20:].mean()
items, geom = build_items_and_geom(field)
field.plot_surfaces(items, bounds=bounds, mode="cmap")
```

## record / replay 実行

環境変数で切り替える運用が最も簡単です。可視化 script 側は通常どおり `emout.Emout("output_dir")` と書きます。

```bash
# 通常実行: 元データを読む
python figure.py

# 記録: 可視化に使ったデータだけを article-records/ に保存
EMOUT_ARTICLE_MODE=record \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
python figure.py

# 再現: 元の巨大な HDF5 ではなく article-records/ から読む
EMOUT_ARTICLE_MODE=replay \
EMOUT_ARTICLE_RECORDS_PATH=article-records \
python figure.py
```

よく使う環境変数は次の通りです。

| 環境変数 | 意味 |
| --- | --- |
| `EMOUT_ARTICLE_MODE=record` | record mode。使われた slice / 平均後データ / 入力メタデータを保存する |
| `EMOUT_ARTICLE_MODE=replay` | replay mode。保存済み bundle から同じ script を再実行する |
| `EMOUT_ARTICLE_RECORDS_PATH=article-records` | article bundle の保存 root |
| `EMOUT_ARTICLE_NAME=fig1` | figure や notebook 単位の bundle 名。省略時は `default` |
| `EMOUT_ARTICLE_SOURCE_NAME=case_a` | 複数 simulation や別環境 replay のための安定した source 名 |
| `EMOUT_ARTICLE_ARCHIVE=zip` | record 後に `.zip` archive を作る。`tar.gz` も指定可能 |

同じ notebook や script の全 figure を 1 つの公開 bundle にまとめたい場合は、`EMOUT_ARTICLE_NAME` を省略して `default` に集めます。
figure ごとに分けたい場合だけ `EMOUT_ARTICLE_NAME=fig1` などを指定します。

複数 simulation を扱う場合や、公開先でパスが変わる場合は `article_source_name` を使います。

```python
data = [
    emout.Emout("case_a/output", article_source_name="case_a"),
    emout.Emout("case_b/output", article_source_name="case_b"),
]
```

アップロード制限がある場合は archive を有効にします。

```bash
EMOUT_ARTICLE_ARCHIVE=zip python figure.py
```

## レビュー時の確認点

- `plot_surfaces()` に `bounds` が渡されているか。なければ 3D 全体保存になり得る。
- `data.phisp[-20:].mean()` は time axis 平均。空間平均を意図する場合は `mean(axis="x")` など明示する。
- `np.asarray(data.phisp[...])` より `data.phisp[...].to_numpy()` が望ましい。
- article replay では particle、backtrace、remote execution 自体は提供されない。必要な可視化グリッドだけを record する。
