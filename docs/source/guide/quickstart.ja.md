Lang: [日本語](quickstart.ja.md) | [English](quickstart.md)

# クイックスタート

## インストール

```bash
pip install emout
```

3D 可視化（PyVista）を使う場合:

```bash
pip install "emout[pyvista]"
```

## シミュレーションデータの読み込み

```python
import emout

data = emout.Emout("output_dir")
```

`Emout` はディレクトリ内の HDF5 ファイルとパラメータファイル（`plasma.inp` または `plasma.toml`）をスキャンします。
変数名は EMSES のファイル名規則から自動的に解決されます:

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
| `data.icur` | `icur`（テキスト） | 電流データ（pandas DataFrame） |
| `data.pbody` | `pbody`（テキスト） | 導体データ（pandas DataFrame） |

各属性は時系列オブジェクトです。タイムステップでインデックスすると NumPy 互換の配列を返します:

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

スライスの軸順序は `(t, z, y, x)` です。2D または 1D にスライスした後、`.plot()` を呼ぶだけで SI 単位付きの図が表示されます。

## 追加出力の結合

シミュレーションが追加ディレクトリに継続出力した場合:

```python
# 自動検出
data = emout.Emout("output_dir", ad="auto")

# 手動指定
data = emout.Emout("output_dir", append_directories=["output_dir_2", "output_dir_3"])
```

## 粒子データ

EMSES の粒子出力は種ごとに自動グルーピングされます:

```python
p4 = data.p4              # 種4
p4.x, p4.y, p4.z          # 位置の時系列
p4.vx, p4.vy, p4.vz       # 速度の時系列
p4.tid                     # トレース ID

# pandas Series に変換
data.p4.vx[0].val_si.to_series().hist(bins=200)
```
