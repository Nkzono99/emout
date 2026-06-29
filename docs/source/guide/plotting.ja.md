# プロット (`plot`)

`plot()` は emout でいちばんよく使う機能です。スライス後の次元に応じて、自動で 1D ラインプロットと 2D カラーマップを切り替えます。

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

`plot()` は既定で軸ラベルと値を SI 単位に変換します。EMSES の内部単位のまま表示したい場合は:

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

PyVista backend を使うと、2D スライスを 3D 空間に配置したり、3D scalar / vector field を対話的に描画できます。詳しい API と重ね描きの例は [PyVista 可視化](pyvista.ja.md) を参照してください。

```bash
pip install emout
```

```python
# 3D scalar volume surface
data.phisp[-1, :, :, :].plot3d(mode="box", show=True)

# 3D scalar isosurfaces
data.phisp[-1].plot3d(mode="contour", levels=[0.0, 5.0], show=True)

# 2D slice placed in 3D space
data.phisp[-1, 100, :, :].plot3d(show=True)

# 3D vector field: plot3d() defaults to PyVista streamlines
data.j1xyz[-1].plot3d(mode="stream", show=True)
data.j1xyz[-1].plot3d(mode="quiver", show=True)

# Streamlines seeded from an xy plane, with tube radius scaled by |v|
data.j1xyz[-1].plot3d(seed_mode="plane", seed_plane="xy", tube_radius="magnitude", show=True)

# Overlay MPIEMSES boundaries as solid transparent surfaces
data.phisp[-1].plot3d(mode="contour", levels=[0.0], surfaces=data.boundaries, show=True)
data.j1xyz[-1].plot3d(surfaces=data.boundaries, show=True)
```

3D ストリームラインの seed は `seed_mode` で選べます。既定の `sphere` は従来どおり中心付近から開始します。`plane` は 2D streamline に近い見え方になりやすく、`volume` は領域全体、`surface` は境界メッシュ上から開始します。任意の開始点を固定したい場合は `seed_points` を渡します:

```python
data.j1xyz[-1].plot3d(seed_mode="plane", seed_plane="xz", seed_position="center")
data.j1xyz[-1].plot3d(seed_mode="volume", n_points=1000)
data.j1xyz[-1].plot3d(seed_mode="surface", seed_surface=data.boundaries)
data.j1xyz[-1].plot3d(seed_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
```

## メッシュサーフェスの描画

PyVista の `surfaces=` は `data.boundaries`、`Boundary`、`MeshSurface3D`、`RenderItem` を受け取り、透明な実体面として重ね描きします。境界面をスカラー値で塗る、等高線を境界面上に描く、といった field-sampled な描画には既存の matplotlib ベースの `plot_surfaces` を使います:

PyVista の 3D プロットも `filename=` で保存できます。`.png` などの画像拡張子は screenshot として保存します。`.html` は PyVista の Jupyter/trame 追加依存が入っている環境では interactive HTML として保存できます。既存の 2D plot と同じく `savefilename=` も互換エイリアスとして使えます:

```python
data.phisp[-1].plot3d(mode="contour", levels=[0.0], filename="phisp_iso.png")
data.j1xyz[-1].plot(surfaces=data.boundaries, filename="j1_stream.html")
```

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
