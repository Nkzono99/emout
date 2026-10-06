# クイックスタート

## インストール

```bash
pip install emout
```

PyVista による 3D 可視化も標準インストールに含まれます。

> Dask によるリモート実行は Python 3.10 以上で自動的に有効になります（別途インストール不要）。

## シミュレーションデータの読み込み

```python
import emout

data = emout.Emout("output_dir")
```

`Emout` はディレクトリ内の HDF5 ファイルとパラメータファイル（`plasma.inp` または `plasma.toml`）を読み込み、
EMSES のファイル名規則から変数名を自動で決めます:

| 属性 | ファイルパターン | 説明 |
| --- | --- | --- |
| `data.phisp` | `phisp00_0000.h5` | 静電ポテンシャル |
| `data.nd1p` | `nd1p00_0000.h5` | 種1 の数密度 |
| `data.j1x` | `j1x00_0000.h5` | 種1 の電流密度 (x成分) |
| `data.ex` | `ex00_0000.h5` | 電場 (x成分) |
| `data.bz` | `bz00_0000.h5` | 磁場 (z成分) |
| `data.rex` | `ex` から再配置 | 再配置された電場 (x成分) |
| `data.j1xy` | `j1x` + `j1y` | 2D ベクトル（自動結合） |
| `data.j1xyz` | `j1x` + `j1y` + `j1z` | 3D ベクトル（自動結合） |
| `data.icur` | `icur`（テキスト） | 流入電流データ（pandas DataFrame、`.val_si` で SI 変換） |
| `data.ocur` | `ocur`（テキスト） | 流出電流データ（pandas DataFrame、`.val_si` で SI 変換） |
| `data.pbody` | `pbody`（テキスト） | 導体電位データ（pandas DataFrame、`.val_si` で SI 変換） |

HDF5 由来の各属性は時系列オブジェクトで、タイムステップを指定すると NumPy 互換の配列が返ります:

```python
len(data.phisp)       # タイムステップ数
data.phisp[0].shape   # (nz, ny, nx)
data.phisp[-1]        # 最終ステップ
```

## 最初のプロット

```python
# 最終ステップの xz 平面（y = ny/2）の電位 2D カラーマップ
data.phisp[-1, :, data.inp.ny // 2, :].plot()
```

2D または 1D にスライスした後、`.plot()` を呼ぶだけで SI 単位付きの図が表示されます。

> **注意: スライスの軸順序は `(t, z, y, x)`** — NumPy の一般的な `(x, y, z)` 慣習とは逆です。
> 上の `data.phisp[-1, :, data.inp.ny // 2, :]` は
> `t=-1`（最終ステップ）、`z=:`（全範囲）、`y=ny/2`（固定）、`x=:`（全範囲）を意味し、
> 結果として xz 平面が得られます。`emout` のインデックスは常にこの順なので、
> 他のコードから持ち込んだスライスはまずこの順に並べ替えてください。

## ベクトル場と成分ごとの値

データの役割に応じて、次の型を使います。

| 型 | 役割 |
| --- | --- |
| `Data1d`〜`Data4d` | 1成分のグリッドデータ。NumPy配列に座標・単位情報を付けたもの |
| `VectorData` | 同じ形・座標を持つ2〜3成分の場。スライス・描画・成分ごとの演算に対応 |
| `ComponentValues` | 1点の値、集計結果、グリッド情報のない配列など。物理的な成分名を保持 |
| `Group` | 任意のオブジェクトへの要素別操作。物理的な成分名やグリッドは扱わない |

`data.exz` や `data.exyz` は従来どおり `VectorData` を返します。
スライスの軸順序は `(t, z, y, x)` です。スライス後は残った軸の順序を使い、
下の2次元の `field` は `(z, x)` の順にインデックスを指定します。

```python
import numpy as np

field = data.exz[-1, :, data.inp.ny // 2, :]
field.components["x"]       # Physical x component
field.components["z"]       # Physical z component
field.component_axes        # ("x", "z")
field.to_numpy()            # Shape: (component, z, x)

(-field).plot()
np.add(field, 1.0)          # Component-wise NumPy arithmetic

sample = field[0, 0]        # ComponentValues: one point
means = field.mean()       # ComponentValues: per-component means
means.components["z"]
means.objs                  # Existing element-wise access remains available
```

成分名を持つオペランド同士の演算では、保存順にかかわらず同じ物理成分を対応付けます。
たとえば `data.exz` と `data.ezx` の加算では x同士、z同士を加算します。
成分の集合、配列の形、グリッド座標が異なる場同士の演算は `ValueError` になります。
通常のNumPy配列は各成分にブロードキャストされ、`Group` のオペランドは従来どおり位置で対応付けます。
要素数の異なる `Group` 同士の演算も `ValueError` になり、末尾の要素を黙って捨てることはありません。
演算は保持している値に対して行い、異なる単位系を自動で換算するものではありません。

読み込んだ `Data` にブール配列・整数配列による添字や `None` による新しい軸を使うと、
結果は座標情報を持たない通常のNumPy配列になります。ベクトル場なら、各成分の配列を
`ComponentValues` にまとめて返します。グリッドを保ったままプロット領域をマスクするには、
`.masked()` を使います（[プロットガイド](plotting.ja.md)）。

`VectorData2d` と `VectorData3d` は引き続き `VectorData` の別名です。
`.objs`、`.attrs`、`.x_data`、`.y_data`、`.z_data` も維持します。
これらの `*_data` は**保存順の1番目・2番目・3番目**という従来の意味なので、
`exz` の `.y_data` は z成分です。物理成分を指定するときは `.components["z"]` を使ってください。
`.components` は読み取り専用の対応表ですが、含まれる配列自体は従来どおり扱えます。
共有するグリッドの座標配列は `.axis(i)` で取得できます。`i` は現在の配列の軸番号です。

## 追加出力の結合

出力が複数のディレクトリに分かれている場合（途中で再投入したジョブなど）:

```python
# 自動検出
data = emout.Emout("output_dir", ad="auto")

# 手動指定
data = emout.Emout("output_dir", append_directories=["output_dir_2", "output_dir_3"])
```

## 粒子データ

粒子の出力は種ごとに 1 つのオブジェクトにまとまっています:

```python
p4 = data.p4              # 種4
p4.x, p4.y, p4.z          # 位置の時系列
p4.vx, p4.vy, p4.vz       # 速度の時系列
p4.tid                     # トレース ID

# pandas Series に変換
data.p4.vx[0].val_si.to_series().hist(bins=200)
```
