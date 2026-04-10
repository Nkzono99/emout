Lang: [日本語](plotting.ja.md) | [English](plotting.md)

# プロット (`plot`)

`plot()` は emout で最もよく使う機能です。データの次元に応じて自動的に適切な可視化を選択します。

## 2D カラーマップ

3D ボリュームを 2D にスライスするとカラーマップが描画されます:

```python
import emout

data = emout.Emout("output_dir")

# 最終ステップの xz 平面（y = ny//2）
data.phisp[-1, :, data.inp.ny // 2, :].plot()

# z=100 での xy 平面
data.phisp[-1, 100, :, :].plot()
```

## 1D ラインプロット

1D にスライスするとラインプロットが描画されます:

```python
# x=32, y=32 での z 軸方向のプロファイル
data.phisp[-1, :, 32, 32].plot()
```

## 主なオプション

| パラメータ | 型 | 説明 | デフォルト |
| --- | --- | --- | --- |
| `use_si` | `bool` | SI 単位で軸ラベル・値を表示 | `True` |
| `show` | `bool` | `plt.show()` を呼ぶ | `False` |
| `savefilename` | `str` | ファイルに保存 | `None` |
| `vmin` | `float` | カラーバーの最小値 | 自動 |
| `vmax` | `float` | カラーバーの最大値 | 自動 |
| `cmap` | colormap | Matplotlib カラーマップ | 独自 gray-jet |
| `norm` | `str` | `'log'` で対数カラースケール | `None` |
| `mode` | `str` | `'cm'`（カラーマップ）, `'cont'`（等高線）, `'cm+cont'`（両方） | `'cm'` |
| `title` | `str` | タイトルを指定 | 自動生成 |
| `xlabel` | `str` | x 軸ラベルを指定 | 自動生成 |
| `ylabel` | `str` | y 軸ラベルを指定 | 自動生成 |

## 使用例

### ファイルに保存

```python
data.phisp[-1, 100, :, :].plot(savefilename="phisp.png")
```

### 対数スケール

```python
data.nd1p[-1, 100, :, :].plot(norm="log", vmin=1e-3, vmax=20)
```

### 等高線表示

```python
data.phisp[-1, 100, :, :].plot(mode="cont")
```

### カラーマップ + 等高線の重ね描き

```python
data.phisp[-1, 100, :, :].plot(mode="cm+cont")
```

### ベクトル場（ストリームライン）

2D ベクトルデータはストリームラインで描画されます:

```python
data.j1xy[-1, 100, :, :].plot()
```

## SI 単位と EMSES 単位

デフォルトでは `plot()` が軸ラベルと値を SI 単位に変換します。EMSES の生の単位で表示するには:

```python
data.phisp[-1, 100, :, :].plot(use_si=False)
```

## SI 値の直接取得

`.val_si` プロパティで SI 単位の NumPy 配列を取得できます:

```python
phisp_V = data.phisp[-1].val_si       # 電位 [V]
j1z_A_m2 = data.j1z[-1].val_si        # 電流密度 [A/m^2]
nd1p_m3 = data.nd1p[-1].val_si        # 数密度 [/m^3]
```

## データマスク

プロット前に特定の領域をマスクできます:

```python
# 平均値以下をマスクしてプロット
data.phisp[1].masked(lambda phi: phi < phi.mean()).plot()
```

## 3D プロット（PyVista）

3D 可視化にはオプションの PyVista 依存をインストールします:

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

メッシュサーフェスを指定して面ごとに 3D レンダリングできます:

```python
import matplotlib.pyplot as plt
from emout.plot.surface_cut import (
    BoxMeshSurface,
    CylinderMeshSurface,
    HollowCylinderMeshSurface,
    RenderItem,
    plot_surfaces,
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

plot_surfaces(
    ax,
    field=field3d,  # surface_cut.Field3D
    surfaces=[
        RenderItem(
            BoxMeshSurface(0, 10, 0, 6, 0, 4, faces=("zmax", "xmax")),
            style="field",
        ),
        RenderItem(
            CylinderMeshSurface(
                center=(5, 3, 2), axis="z", radius=1.5, length=4.0,
                parts=("side", "top"),
            ),
            style="solid",
            solid_color="0.7",
            alpha=0.5,
        ),
    ],
)
```
