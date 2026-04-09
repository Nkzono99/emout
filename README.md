Lang: [日本語](README.md) | [English](README.en.md)

# emout

[![PyPI version](https://img.shields.io/pypi/v/emout.svg)](https://pypi.org/project/emout/)
[![Python](https://img.shields.io/pypi/pyversions/emout.svg)](https://pypi.org/project/emout/)
[![Docs](https://github.com/Nkzono99/emout/actions/workflows/docs.yaml/badge.svg)](https://nkzono99.github.io/emout/)
[![CodeQL](https://github.com/Nkzono99/emout/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/Nkzono99/emout/actions/workflows/codeql-analysis.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[EMSES](https://github.com/Nkzono99/MPIEMSES3D) シミュレーション出力の解析・可視化 Python ライブラリ**

emout は:

- `.h5` グリッド出力 + `plasma.inp` / `plasma.toml` を 1 行で読み込む Facade
- データ次元から自動で最適なビューを選ぶ 1D / 2D / 3D プロット
- `!!key dx=...,to_c=...` を解析して生成する EMSES ⇄ SI 単位変換器（30+ の物理量に対応）
- EMSES の finbound 境界を Python オブジェクトとして扱える境界 API

---

- **ドキュメント:** [ユーザーガイド（日本語）](https://nkzono99.github.io/emout/guide/quickstart.ja.html) | [API リファレンス](https://nkzono99.github.io/emout/api/emout.html) | [User Guide (English)](https://nkzono99.github.io/emout/guide/quickstart.html)
- **ノートブック例:** [月面帯電シミュレーション結果の可視化](https://nbviewer.org/github/Nkzono99/examples/blob/main/examples/emout/example.ipynb)

---

## インストール

```bash
pip install emout

# 3D 可視化（PyVista）を使う場合
pip install "emout[pyvista]"
```

---

## クイックスタート

```python
import emout

data = emout.Emout("output_dir")

# 最終ステップの電位を xz 平面（y=ny/2）でプロット — これだけで SI 単位付きカラーマップ
data.phisp[-1, :, data.inp.ny // 2, :].plot()
```

変数名は EMSES のファイル名から自動解決されます:

```python
data.phisp          # 電位 (GridDataSeries — 時系列)
data.nd1p           # 種1 数密度
data.j1x            # 種1 電流密度 x成分
data.j1xy           # j1x + j1y 自動結合 → 2D ベクトル
data.j1xyz          # 3D ベクトル
data.icur, data.pbody  # テキスト出力 (pandas DataFrame)
```

スライスの軸順序は `(t, z, y, x)` です。

---

## 機能ガイド

各機能の詳しい使い方はユーザーガイドを参照してください。

| 機能 | できること | ガイド |
| --- | --- | --- |
| **プロット** | `plot()` / `cmap()` / `contour()` で 1D/2D プロット | [→ プロット](https://nkzono99.github.io/emout/guide/plotting.ja.html) |
| **アニメーション** | `gifplot()` で GIF/HTML 生成、複数パネルレイアウト | [→ アニメーション](https://nkzono99.github.io/emout/guide/animation.ja.html) |
| **パラメータ** | `data.inp.nx`, `data.toml.species[0].wp` | [→ パラメータ](https://nkzono99.github.io/emout/guide/inp.ja.html) |
| **単位変換** | `data.unit.v.reverse(1.0)`, `data.phisp[-1].val_si` | [→ 単位変換](https://nkzono99.github.io/emout/guide/units.ja.html) |
| **境界メッシュ** | `data.boundaries.mesh()`, `plot_surfaces` へのオーバーレイ | [→ 境界メッシュ](https://nkzono99.github.io/emout/guide/boundaries.ja.html) |
| **3D (PyVista)** | `plot3d(mode="box"/"stream"/"quiver")` | [→ クイックスタート](https://nkzono99.github.io/emout/guide/quickstart.ja.html) |
| **リモート実行** | Dask Actor で計算ノードに処理を委譲、ローカルは画像だけ | [→ リモート実行](https://nkzono99.github.io/emout/guide/distributed.ja.html) |

---

## 代表的な使い方

### プロット

```python
data.phisp[-1, 100, :, :].plot()                       # 2D カラーマップ
data.phisp[-1, 100, :, :].contour()                     # 等高線
data.nd1p[-1, 100, :, :].plot(norm="log", vmin=1e-3)    # 対数スケール
data.j1xy[-1, 100, :, :].plot()                          # ストリームライン
data.phisp[-1, :, 32, 32].plot()                         # 1D プロファイル
```

### アニメーション

```python
data.phisp[:, 100, :, :].gifplot()                                  # Jupyter インライン
data.phisp[:, 100, :, :].gifplot(action="save", filename="out.gif") # GIF 保存
```

### 単位変換

```python
data.unit.v.trans(1.0)       # SI → EMSES
data.phisp[-1].val_si        # 全 3D 配列を SI [V] で取得
```

### 境界メッシュ

```python
data.boundaries[0].mesh()                   # 個別境界の MeshSurface3D
data.phisp[-1].plot_surfaces(               # フィールド上にオーバーレイ
    ax=ax, surfaces=data.boundaries,
)
```

### 粒子データ

```python
p4 = data.p4                               # 種4
p4.vx[0].val_si.to_series().hist(bins=200)  # 速度分布
```

<details>
<summary>追加出力の結合 / 入出力パスの分離</summary>

```python
# 継続出力の結合
data = emout.Emout("output_dir", ad="auto")

# 入力ファイルと出力ディレクトリを分離
data = emout.Emout(input_path="/path/to/plasma.toml", output_directory="output_dir")
```

</details>

### リモート実行 (Dask) — 実験的

HPC の計算ノードにデータ処理を委譲し、ログインノードにはプロット画像だけを返します。
**コードの書き方はローカル実行と全く同じ**で、サーバーが起動していれば自動的にリモートになります。

```bash
# ターミナルでサーバーを起動（1 回だけ）
emout server start --partition gr20001a --memory 60G
```

```python
# スクリプト / Jupyter — サーバーがあれば自動リモート、なければローカル
data.phisp[-1, :, 100, :].plot()    # 2D スライスだけ転送
plt.xlabel("x [m]")                 # ローカル matplotlib で追記可能

# 全操作をサーバーで実行（ローカルにはPNG画像のみ）
from emout.distributed import remote_figure

with remote_figure():
    data.phisp[-1, :, 100, :].plot()
    plt.axhline(y=50, color="red")
    plt.title("カスタムタイトル")

# open/close 形式 — 既存コードへの導入が容易
from emout.distributed import RemoteFigure

rf = RemoteFigure()
rf.open()
data.phisp[-1, :, 100, :].plot()
rf.close()

# Jupyter セルマジック — セル先頭に書くだけ
# %load_ext emout.distributed.remote_figure
# %%remote_figure
# data.phisp[-1, :, 100, :].plot()
```

backtrace の重い計算もサーバーで実行し、可視化パラメータだけ変えて何度でも再描画できます。

**複数シミュレーションの比較**も可能です:

```python
data_a = emout.Emout("/sim_a")
data_b = emout.Emout("/sim_b")
result_a = data_a.backtrace.get_probabilities(...)
result_b = data_b.backtrace.get_probabilities(...)

with remote_figure(figsize=(12, 5)):
    plt.subplot(1, 2, 1)
    result_a.vxvz.plot()
    plt.subplot(1, 2, 2)
    result_b.vxvz.plot()
```

→ [リモート実行ガイド](https://nkzono99.github.io/emout/guide/distributed.ja.html)

<details>
<summary>実験的機能（ポアソン方程式 / バックトレース）</summary>

```python
# ポアソン方程式
from emout.utils import poisson
phi = poisson(rho, dx=dx, btypes=btypes, epsilon_0=cn.epsilon_0)

# バックトレース（要 vdist-solver-fortran）
result = data.backtrace.get_probabilities(x, y, z, vx, vy, vz, ispec=0)
result.vxvz.plot()
```

</details>

---

## コントリビュート

バグ報告・機能提案・PR を歓迎します。

- **バグ / 質問:** [GitHub Issues](https://github.com/Nkzono99/emout/issues) に再現手順を添えて投稿してください
- **PR:** `main` から作業ブランチを切り、`pytest -q` がグリーンの状態で送ってください
- **ドキュメント:** `README.md`（日本語）と `README.en.md`（英語）は対応する形で維持されています。片方を更新したらもう片方にも反映してください

開発環境のセットアップやディレクトリ構成は [AGENTS.md](AGENTS.md) にまとまっています。

---

## ライセンス

[MIT License](LICENSE)

## リンク

- [ユーザーガイド（日本語）](https://nkzono99.github.io/emout/guide/quickstart.ja.html) | [User Guide (English)](https://nkzono99.github.io/emout/guide/quickstart.html)
- [API リファレンス](https://nkzono99.github.io/emout/api/emout.html)
- [EMSES (MPIEMSES3D)](https://github.com/Nkzono99/MPIEMSES3D)
- [サンプルノートブック](https://nbviewer.org/github/Nkzono99/examples/blob/main/examples/emout/example.ipynb)
