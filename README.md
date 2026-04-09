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
- `!!key dx=...,to_c=...` を解析して生成する EMSES ⇄ SI 単位変換器
- EMSES の finbound 境界を Python オブジェクトとして扱える境界 API

---

- **ドキュメント（日本語）:** [ユーザーガイド](https://nkzono99.github.io/emout/guide/quickstart.ja.html) | [API リファレンス](https://nkzono99.github.io/emout/api/emout.html)
- **ノートブック例:** [月面帯電シミュレーション結果の可視化](https://nbviewer.org/github/Nkzono99/examples/blob/main/examples/emout/example.ipynb)

EMSES の出力ファイル（`.h5`）とパラメータファイル（`plasma.inp` / `plasma.toml`）を読み込み、
数行のコードでデータの閲覧・プロット・アニメーション作成・SI 単位への変換ができます。

---

## 主な特徴

- **最小コードで可視化** — `data.phisp[-1, :, 100, :].plot()` の 1 行で SI 単位付きの 2D プロットが描ける
- **時系列アニメーション** — `gifplot()` 一発で GIF / HTML、`build_frame_updater()` で複数パネル
- **自動 SI 単位変換** — `plasma.inp` 1 行目の `!!key dx=...,to_c=...` から 30+ の物理量の単位換算器を生成
- **動的変数解決** — `data.phisp`, `data.nd1p`, `data.j1xy`, `data.j1xyz` のように EMSES のファイル名から直接アクセス
- **3D 可視化** — PyVista 連携、finbound 境界メッシュの自動生成 (`data.boundaries`)
- **パラメータファイル対応** — 旧式 `plasma.inp` と新式 `plasma.toml` の両方に対応、TOML のネスト構造は `data.toml` でそのまま参照

---

## 目次

1. [インストール](#インストール)
2. [クイックスタート](#クイックスタート)
3. [データの読み込み](#データの読み込み)
4. [プロット (`plot`)](#プロット-plot)
5. [アニメーション (`gifplot`)](#アニメーション-gifplot)
6. [パラメータファイル (`data.inp`)](#パラメータファイル-datainp)
7. [単位変換 (`data.unit`)](#単位変換-dataunit)
8. [粒子データ](#粒子データ)
9. [データマスク](#データマスク)
10. [追加出力の結合](#追加出力の結合)
11. [3D プロット (PyVista)](#3d-プロット-pyvista)
12. [境界メッシュ (`data.boundaries`)](#境界メッシュ-databoundaries)
13. [ポアソン方程式の求解 (実験的)](#ポアソン方程式の求解-実験的)
14. [バックトレース (実験的)](#バックトレース-実験的)
15. [コントリビュート](#コントリビュート)

---

## インストール

```bash
pip install emout
```

3D 可視化（PyVista）を使う場合:

```bash
pip install "emout[pyvista]"
```

---

## クイックスタート

```python
import emout

data = emout.Emout("output_dir")

# 最終ステップの電位を xz 平面（y=ny/2）でプロット
data.phisp[-1, :, data.inp.ny // 2, :].plot()
```

これだけで SI 単位付きのカラーマップが表示されます。

---

## データの読み込み

以下のようなディレクトリ構造を前提とします:

```
output_dir/
├── plasma.inp          # パラメータファイル
├── phisp00_0000.h5     # 電位
├── nd1p00_0000.h5      # 種1の数密度
├── nd2p00_0000.h5      # 種2の数密度
├── j1x00_0000.h5       # 種1の電流密度 x成分
├── ex00_0000.h5        # 電場 x成分
├── bz00_0000.h5        # 磁場 z成分
└── ...
```

```python
import emout

data = emout.Emout("output_dir")

# 入力ファイルと出力ディレクトリを分離
data = emout.Emout(input_path="/path/to/plasma.toml", output_directory="output_dir")

# 変数名は EMSES のファイル名から自動解決
data.phisp          # 電位 (時系列)
len(data.phisp)     # タイムステップ数
data.phisp[0].shape # (nz, ny, nx)

data.nd1p           # 種1の数密度
data.j1x            # 種1の電流密度 x成分
data.bz             # 磁場 z成分

# ベクトルデータ（自動結合）
data.j1xy           # j1x + j1y → 2D ベクトル
data.j1xyz          # j1x + j1y + j1z → 3D ベクトル

# 再配置データ
data.rex            # 再配置された電場 x成分

# テキスト出力（pandas DataFrame）
data.icur           # 電流データ
data.pbody          # 導体データ
```

---

## プロット (`plot`)

`plot()` はデータの次元に応じて自動的に適切な描画を行います。
**最もよく使う機能です。**

### 2D カラーマップ

```python
# 最終ステップの xz 平面 (y=ny//2)
data.phisp[-1, :, data.inp.ny // 2, :].plot()

# xy 平面 (z=100)
data.phisp[-1, 100, :, :].plot()
```

### 1D ラインプロット

```python
# z 軸方向のプロファイル (x=32, y=32)
data.phisp[-1, :, 32, 32].plot()
```

### 主なオプション

| パラメータ | 説明 | デフォルト |
| --- | --- | --- |
| `use_si` | SI 単位系で軸ラベル・値を表示 | `True` |
| `show` | `plt.show()` を呼ぶ | `False` |
| `savefilename` | 画像ファイルとして保存 | `None` |
| `vmin`, `vmax` | カラーバー範囲 | 自動 |
| `cmap` | カラーマップ | 独自 gray-jet |
| `norm` | `'log'` で対数スケール | `None` |
| `mode` | `'cm'`, `'cont'`, `'cm+cont'` (2D) | `'cm'` |

```python
# SI 単位で保存
data.phisp[-1, 100, :, :].plot(savefilename="phisp.png")

# 対数スケール
data.nd1p[-1, 100, :, :].plot(norm="log", vmin=1e-3, vmax=20)

# 等高線表示
data.phisp[-1, 100, :, :].plot(mode="cont")

# ベクトル場（ストリームライン）
data.j1xy[-1, 100, :, :].plot()
```

#### 2D ショートカットメソッド

2D データでは `mode=` をわざわざ渡す代わりに、意図を名前に刻んだ 2 つのショートカットが使えます。
引数は `plot()` と同じで（`mode` だけ受け付けません）、後から読み返すときに意図がすぐ分かります。

```python
# mode='cm' と等価
data.phisp[-1, 100, :, :].cmap()

# mode='cont' と等価
data.phisp[-1, 100, :, :].contour()
```

`plot(mode='cm+cont')` のように重ね描きしたい場合や、1D / 3D プロットは従来どおり `plot()` を使ってください。

---

## アニメーション (`gifplot`)

時系列データから GIF / HTML アニメーションを作成します。**2 番目によく使う機能です。**

### 基本的な使い方

```python
# Jupyter Notebook でインライン表示（デフォルト）
data.phisp[:, 100, :, :].gifplot()

# GIF ファイルとして保存
data.phisp[:, 100, :, :].gifplot(action="save", filename="phisp.gif")

# matplotlib ウィンドウで表示
data.phisp[:, 100, :, :].gifplot(action="show")
```

### 主なオプション

| パラメータ | 説明 | デフォルト |
| --- | --- | --- |
| `action` | `'to_html'`, `'save'`, `'show'`, `'return'`, `'frames'` | `'to_html'` |
| `filename` | `action='save'` 時の保存先 | `None` |
| `axis` | アニメーション軸 | `0` |
| `interval` | フレーム間隔 [ms] | `200` |
| `use_si` | SI 単位系 | `True` |
| `vmin`, `vmax` | カラーバー範囲 | 自動 |
| `norm` | `'log'` で対数スケール | `None` |

### 複数パネルアニメーション

```python
# フレームアップデータを作成
updater0 = data.phisp[:, 100, :, :].gifplot(action="frames", mode="cmap")
updater1 = data.phisp[:, 100, :, :].build_frame_updater(mode="cont")
updater2 = data.nd1p[:, 100, :, :].build_frame_updater(mode="cmap", vmin=1e-3, vmax=20, norm="log")
updater3 = data.nd2p[:, 100, :, :].build_frame_updater(mode="cmap", vmin=1e-3, vmax=20, norm="log")
updater4 = data.j2xy[:, 100, :, :].build_frame_updater(mode="stream")

# レイアウトを定義（3重リスト: [行][列][重ね合わせ]）
layout = [
    [
        [updater0, updater1],
        [updater2],
        [updater3, updater4],
    ]
]

animator = updater0.to_animator(layout=layout)
animator.plot(action="to_html")  # or "save", "show"
```

---

## パラメータファイル (`data.inp`)

`plasma.inp`（または `plasma.toml`）を辞書風オブジェクトとして読み込みます。

```python
# グループ名 + パラメータ名でアクセス
data.inp["tmgrid"]["nx"]    # → 例: 256
data.inp["plasma"]["wp"]    # → 例: [1.0, 0.05]

# グループ名を省略（あいまいでなければ）
data.inp["nx"]

# 属性アクセス
data.inp.tmgrid.nx
data.inp.nx
```

### よく使うパラメータ

```python
# グリッドサイズ
nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz

# 時間ステップ
dt = data.inp.dt
ifdiag = data.inp.ifdiag  # 出力間隔

# 粒子数
nspec = data.inp.nspec  # 粒子種数

# 境界条件
data.inp.mtd_vbnd  # 各軸の境界条件 (0=periodic, 1=Dirichlet, 2=Neumann)
```

### TOML 形式 (`plasma.toml`)

`plasma.toml` がディレクトリに存在する場合、`toml2inp` コマンドで `plasma.inp` を自動生成してから読み込みます:

```python
data = emout.Emout("output_dir")  # plasma.toml があれば toml2inp → plasma.inp を生成
data.inp.nx  # 同じインターフェース
data.toml    # 構造化 TOML に直接アクセス (TomlData)
data.toml.species[0].wp  # ネスト構造のまま参照可能
```

> **注意:** `toml2inp` コマンドが PATH に必要です（[MPIEMSES3D](https://github.com/Nkzono99/MPIEMSES3D) に同梱）。
> `species_groups` などの `*_groups` は読み込み時に各 entry へ展開され、`data.toml` からは除外されます。

---

## 単位変換 (`data.unit`)

EMSES 内部単位と SI 単位の相互変換を行います。

> **前提条件:** `plasma.inp` の 1 行目に `!!key dx=[0.5],to_c=[10000.0]` のような記述が必要です。
> `dx` はグリッド間隔 [m]、`to_c` は EMSES 内部の光速値です。

### 単位変換器の使い方

```python
# SI → EMSES
data.unit.v.trans(1.0)      # 1 m/s → EMSES 速度単位

# EMSES → SI
data.unit.v.reverse(1.0)    # 1 EMSES速度単位 → m/s
```

### SI 値の直接取得

```python
# .val_si プロパティでデータを SI 単位に変換
phisp_V = data.phisp[-1].val_si         # 電位 [V]
j1z_A_m2 = data.j1z[-1].val_si          # 電流密度 [A/m^2]
nd1p_m3 = data.nd1p[-1].val_si          # 数密度 [/m^3]
```

### 利用可能な単位一覧

<details>
<summary>クリックで展開</summary>

| 名前 | 物理量 | SI 単位 |
| --- | --- | --- |
| `phi` | 電位 | V |
| `E` | 電場 | V/m |
| `B` | 磁束密度 | T |
| `J` | 電流密度 | A/m^2 |
| `n` | 数密度 | /m^3 |
| `rho` | 電荷密度 | C/m^3 |
| `v` | 速度 | m/s |
| `t` | 時間 | s |
| `f` | 周波数 | Hz |
| `length` | 長さ | m |
| `q` | 電荷 | C |
| `m` | 質量 | kg |
| `W` | エネルギー | J |
| `w` | エネルギー密度 | J/m^3 |
| `P` | パワー | W |
| `T` | 温度 | K |
| `F` | 力 | N |
| `a` | 加速度 | m/s^2 |
| `i` | 電流 | A |
| `N` | フラックス | /m^2s |
| `c` | 光速 | m/s |
| `eps` | 誘電率 | F/m |
| `mu` | 透磁率 | H/m |
| `C` | 静電容量 | F |
| `L` | インダクタンス | H |
| `G` | コンダクタンス | S |
| `q_m` | 比電荷 | C/kg |
| `qe` | 素電荷 | C |
| `qe_me` | 電子比電荷 | C/kg |
| `kB` | ボルツマン定数 | J/K |
| `e0` | 真空誘電率 | F/m |
| `m0` | 真空透磁率 | N/A^2 |

</details>

---

## 粒子データ

EMSES の粒子出力（`p4xe00_0000.h5`, `p4vxe00_0000.h5` 等）を自動でグルーピングします。

```python
# 種4の粒子データ
p4 = data.p4

# 成分ごとの時系列
p4.x, p4.y, p4.z         # 位置
p4.vx, p4.vy, p4.vz      # 速度
p4.tid                     # トレースID

# pandas Series に変換（ヒストグラム等に便利）
data.p4.vx[0].val_si.to_series().hist(bins=200)
```

---

## データマスク

<details>
<summary>例を表示</summary>

```python
# 平均値以下をマスク
data.phisp[1].masked(lambda phi: phi < phi.mean())

# 手動で同等の処理
phi = data.phisp[1].copy()
phi[phi < phi.mean()] = float("nan")
```

</details>

---

## 入力ファイルと出力ディレクトリの分離

入力パラメータファイルと出力ファイルが別の場所にある場合:

```python
# 入力ファイルのフルパスを指定
data = emout.Emout(input_path="/path/to/plasma.toml", output_directory="output_dir")

# 出力ディレクトリのみ指定（入力ファイルは output_dir 内を探索）
data = emout.Emout("output_dir")

# 従来通りの使い方（すべて同じディレクトリ）
data = emout.Emout("output_dir")
```

| パラメータ | 説明 | デフォルト |
| --- | --- | --- |
| `directory` | 基準ディレクトリ | `"./"` |
| `input_path` | 入力ファイルのフルパス（例: `/path/to/plasma.toml`） | `None` |
| `output_directory` | 出力ファイルのディレクトリ | `directory` と同じ |

---

## 追加出力の結合

<details>
<summary>例を表示</summary>

シミュレーションを継続実行した場合:

```python
# 手動指定
data = emout.Emout("output_dir", append_directories=["output_dir_2", "output_dir_3"])

# 自動検出
data = emout.Emout("output_dir", ad="auto")
```

</details>

---

## 3D プロット (PyVista)

<details>
<summary>例を表示</summary>

```bash
pip install "emout[pyvista]"
```

```python
# 3D ボリュームレンダリング
data.phisp[-1, :, :, :].plot3d(mode="box", show=True)

# 2D スライスを 3D 空間に配置
data.phisp[-1, 100, :, :].plot3d(show=True)

# 3D ベクトル場
data.j1xyz[-1].plot3d(mode="stream", show=True)
data.j1xyz[-1].plot3d(mode="quiver", show=True)
```

### メッシュサーフェス描画

```python
import matplotlib.pyplot as plt
from emout.plot.surface_cut import (
    BoxMeshSurface, CylinderMeshSurface, HollowCylinderMeshSurface,
    RenderItem, plot_surfaces,
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

plot_surfaces(
    ax,
    field=field3d,
    surfaces=[
        RenderItem(BoxMeshSurface(0, 10, 0, 6, 0, 4, faces=("zmax", "xmax")), style="field"),
        RenderItem(
            CylinderMeshSurface(center=(5, 3, 2), axis="z", radius=1.5, length=4.0, parts=("side", "top")),
            style="solid", solid_color="0.7", alpha=0.5,
        ),
    ],
)
```

</details>

---

## 境界メッシュ (`data.boundaries`)

MPIEMSES の `&ptcond` で定義された finbound / legacy 境界型を Python オブジェクトとして扱えます。
各境界は `MeshSurface3D` を返し、3D フィールドプロットへのオーバーレイや個別カスタマイズが可能です。

### アクセス

```python
# コレクション全体
data.boundaries                 # BoundaryCollection (iterable, indexable)
len(data.boundaries)
data.boundaries.types           # 各境界の boundary_types 文字列リスト
data.boundaries.skipped         # [(index, type_name, reason), ...]

# 個別の境界
data.boundaries[0]              # SphereBoundary / CylinderBoundary / ... などのサブクラス
data.boundaries[0].mesh()       # → MeshSurface3D (デフォルトは SI 単位)
data.boundaries[0].mesh(use_si=False)  # グリッド単位のまま

# 全境界を一つの複合メッシュとして取得
data.boundaries.mesh()          # → CompositeMeshSurface
```

### 3D フィールドプロットへのオーバーレイ

`Data3d.plot_surfaces` に `data.boundaries` を渡すと、等位面・スライスの上に境界形状を重ねて描画できます。

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

data.phisp[-1].plot_surfaces(
    ax=ax,
    surfaces=data.boundaries,         # 全境界を自動で RenderItem 化
)
plt.show()
```

### 合成とカスタマイズ

```python
# Boundary + Boundary → BoundaryCollection
combined = data.boundaries[0] + data.boundaries[1]

# 境界ごとにスタイルを変える
data.phisp[-1].plot_surfaces(
    ax=ax,
    surfaces=data.boundaries.render(
        per={
            0: dict(style="solid", solid_color="0.7"),
            1: dict(alpha=0.5),
        },
    ),
)
```

### 対応している境界型

`boundary_type = "complex"` モードで `boundary_types(i)` に指定できる形状と、
legacy 単独モードで `boundary_type = "..."` に直接指定できる型の両方をサポートしています。

| カテゴリ | 型名 |
| --- | --- |
| 閉じた立体 | `sphere`, `cuboid` |
| 円柱 | `cylinderx`, `cylindery`, `cylinderz`, `open-cylinderx/y/z` |
| 平板 | `rectangle`, `circlex/y/z`, `diskx/y/z`, `plane-with-circlex/y/z` |
| Legacy 単独モード | `flat-surface`, `rectangle-hole`, `cylinder-hole` |

詳細なパラメータ定義は [MPIEMSES3D Parameters.md](https://github.com/Nkzono99/MPIEMSES3D/blob/main/docs/Parameters.md) を参照してください。未登録の型は `data.boundaries.skipped` に `(index, 型名, 理由)` として残り、警告にはなりません。

---

## ポアソン方程式の求解 (実験的)

<details>
<summary>例を表示</summary>

```python
import numpy as np
import scipy.constants as cn
from emout import Emout
from emout.utils import poisson

data = Emout("output_dir")
dx = data.inp.dx
rho = data.rho[-1].val_si
btypes = ["pdn"[i] for i in data.inp.mtd_vbnd]

phisp = poisson(rho, dx=dx, btypes=btypes, epsilon_0=cn.epsilon_0)
```

</details>

---

## バックトレース (実験的)

<details>
<summary>例を表示</summary>

```bash
pip install git+https://github.com/Nkzono99/vdist-solver-fortran.git
```

```python
# 確率分布の計算
probability_result = data.backtrace.get_probabilities(
    128, 128, 60,
    (-data.inp.path[0] * 3, data.inp.path[0] * 3, 10),
    0,
    (-data.inp.path[0] * 3, 0, 10),
    ispec=0,
)
probability_result.vxvz.plot()

# バックトレース軌道の計算と描画
particles = probability_result.particles
prob_1d = probability_result.probabilities.ravel()
alpha_values = np.nan_to_num(prob_1d / prob_1d.max())

backtrace_result = data.backtrace.get_backtraces_from_particles(particles, ispec=0)
backtrace_result.xz.plot(color="black", alpha=alpha_values)
```

### Dask クラスタ連携

```python
from emout.distributed import start_cluster, stop_cluster

client = start_cluster(
    partition="gr20001a",
    processes=1, cores=112, memory="60G",
    walltime="03:00:00",
    scheduler_port=32332,
)

# 以降の data.backtrace API は計算ノード上で実行される
result = data.backtrace.get_probabilities(...)
stop_cluster()
```

</details>

---

## コントリビュート

バグ報告・機能提案・PR を歓迎します。

- **バグ / 質問:** [GitHub Issues](https://github.com/Nkzono99/emout/issues) に再現手順を添えて投稿してください
- **PR:** `main` から作業ブランチを切り、`pytest -q` がグリーンの状態で送ってください（現在のベースラインは `AGENTS.md §10` を参照）
- **ドキュメント:** `README.md` (日本語) と `README.en.md` (英語) は対応する形で維持されています。片方を更新したらもう片方にも反映してください

開発環境のセットアップやディレクトリ構成は [AGENTS.md](AGENTS.md) にまとまっています。

---

## ライセンス

[MIT License](LICENSE)

## リンク

- [ユーザーガイド（日本語）](https://nkzono99.github.io/emout/guide/quickstart.ja.html)
- [API リファレンス](https://nkzono99.github.io/emout/api/emout.html)
- [EMSES (MPIEMSES3D)](https://github.com/Nkzono99/MPIEMSES3D)
- [サンプルノートブック](https://nbviewer.org/github/Nkzono99/examples/blob/main/examples/emout/example.ipynb)
