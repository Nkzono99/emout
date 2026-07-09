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

PyVista backend を使うと、2D スライスを 3D 空間に配置したり、3D scalar / vector field を描画できます。このページでは入口だけ示します。mode、重ね描き、保存、HPC での使い方は [PyVista 可視化](pyvista.ja.md) を参照してください。

```python
# 3D scalar volume surface
data.phisp[-1, :, :, :].plot3d(mode="box", show=True)

# 2D slice placed in 3D space
data.phisp[-1, 100, :, :].plot3d(show=True)

# 3D vector field
data.j1xyz[-1].plot3d(mode="stream", show=True)
```

## メッシュサーフェスの描画

境界を 3D field に重ねたい場合は、まず `data.boundaries.plot3d()` または `plot3d(..., surfaces=data.boundaries)` を使います。境界メッシュの合成、境界ごとの style、field-sampled な `plot_surfaces()` の詳細は [境界メッシュ](boundaries.ja.md) を参照してください。

```python
data.phisp[-1].plot3d(mode="contour", levels=[0.0], filename="phisp_iso.png")
data.j1xyz[-1].plot3d(surfaces=data.boundaries, filename="j1_stream.png")
```
