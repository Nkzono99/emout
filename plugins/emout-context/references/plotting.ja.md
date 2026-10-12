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

## 空間平均後の断面

時間範囲の空間平均を描画するときは `keep_metadata=True` を指定します。
縮約した軸を除き、残る軸の座標・単位と次元に合う `Data` クラスを保持します。
既定の `False` は従来の NumPy 縮約の挙動を維持するため、平均後の描画には使わないでください。
`mean()` の時間平均の既定動作は変わりません。

```python
yz = data.phisp[:].mean(axis="x", keep_metadata=True)
yz[-1].plot()  # Horizontal: y; vertical: z
profile = data.phisp[:].mean(axis=("x", "y"), keep_metadata=True)
profile[-1].plot()  # z profile
```

空間平均は選択範囲をメモリに読み込みます。論文用の記録・再生でも同じ呼び出しを使えますが、
記録されるのは縮約前の選択範囲です。

## 3D 等値面（Matplotlib）

`Data3d.plot(mode="cont")` で切り出し・間引き後の等値面を元のグリッド座標に配置するには、
`coordinates="grid"` を指定します。各軸の開始座標と間隔を使い、`use_si=True` では
各軸を SI 単位に変換します。単位情報がない場合はグリッド単位を使います。
既定の `coordinates="local"` は、原点ゼロ・間隔1（SI 時は x 軸の単位長）の従来表示です。

```python
roi = data.phisp[-1, 2:8:2, 3:9:2, 4:16:2]
roi.plot(mode="cont", coordinates="grid", use_si=False, levels=[8.0], show=False)
```

`offsets=(x, y, z)` は表示単位で座標に加算します。文字列の `"left"`、`"center"`、
`"right"` は、それぞれ選択範囲の最初・中央のサンプル・最後をゼロに移します。
`bounds_xyz` もオフセット適用後の表示座標で指定します。

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
