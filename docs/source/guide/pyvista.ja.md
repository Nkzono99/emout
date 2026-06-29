# PyVista 可視化 (`plot3d` / `plot_pyvista`)

PyVista backend は、2D スライスを 3D 空間に置く、3D スカラー場を volume / slice / contour として描く、3D ベクトル場を streamlines / quiver として描くための 3D 可視化 API です。通常の 1D/2D 解析は [プロット](plotting.ja.md) の `plot()` / `cmap()` / `contour()` を使い、3D の視点操作や重ね描きが必要なときに PyVista を使います。

## 入口の選び方

| 対象 | 推奨 API | 戻り値 |
| --- | --- | --- |
| 2D scalar slice | `data.phisp[-1, 100, :, :].plot3d(...)` | `pyvista.Plotter` |
| 3D scalar volume | `data.phisp[-1].plot3d(mode=...)` | `pyvista.Plotter` |
| 3D vector field | `data.j1xyz[-1].plot3d(mode=..., backend="pyvista")` | `pyvista.Plotter` |
| Boundary meshes | `data.boundaries.plot3d(plotter=...)` | `pyvista.Plotter` |
| Backtrace / trace paths | `trace.plot3d(plotter=...)` | `pyvista.Plotter` |
| Mesh construction only | `emout.plot.pyvista_plot.create_*_mesh(...)` | PyVista mesh object |

`Data2d.plot3d()` と `Data3d.plot3d()` は `plot_pyvista()` の alias です。`VectorData.plot3d()` は既定で PyVista backend を使います。Matplotlib 3D backend を使う場合は `backend="mpl"` を指定します。

## インストール

PyVista は emout の通常依存としてインストールされます。3D 表示を使う環境でも追加の extra 指定は不要です。

```bash
pip install emout
```

古い環境や editable install で `ModuleNotFoundError` が出る場合は、emout を再インストールして依存関係を更新してください。

## スカラー場

### 2D スライスを 3D 空間に置く

2D slice は `pyvista.StructuredGrid` の平面として描画されます。`plot3d()` は plotter を返すので、あとからカメラや screenshot を操作できます。

```python
import emout

data = emout.Emout("output_dir")

plotter = data.phisp[-1, 100, :, :].plot3d(
    cmap="viridis",
    clim=(-20, 20),
    show_edges=False,
    show=False,
)
plotter.show()
```

### 3D ボリュームを描く

3D scalar は `mode` で描き方を切り替えます。

| `mode` | 描画内容 | 主な追加オプション |
| --- | --- | --- |
| `"box"` | 外側 surface | `opacity`, `show_edges` |
| `"volume"` | volume rendering | `opacity` |
| `"slice"` | 直交 slice | `cmap`, `clim` |
| `"contour"` | 等値面 | `contour_levels`, `opacity` |

```python
data.phisp[-1].plot3d(mode="box", opacity=0.4, show=True)
data.phisp[-1].plot3d(mode="volume", opacity="sigmoid", show=True)
data.phisp[-1].plot3d(mode="slice", cmap="coolwarm", clim=(-50, 50), show=True)
data.phisp[-1].plot3d(mode="contour", contour_levels=12, show=True)
```

画像として保存したい場合は、`show=False` で plotter を受け取り、PyVista の `screenshot()` を使います。

```python
plotter = data.phisp[-1].plot3d(mode="contour", contour_levels=10, show=False)
plotter.screenshot("phisp_contour.png")
plotter.close()
```

## ベクトル場

3 成分を持つ `VectorData` は、`mode="stream"` / `"streamline"` で streamlines、`mode="quiver"` / `"vec"` で glyph arrows を描けます。

```python
data.j1xyz[-1].plot3d(
    mode="stream",
    backend="pyvista",
    n_points=300,
    tube_radius=0.02,
    show=True,
)

data.j1xyz[-1].plot3d(
    mode="quiver",
    backend="pyvista",
    skip=(3, 3, 2),
    factor=0.4,
    show=True,
)
```

streamline の seed は PyVista の `mesh.streamlines()` に渡されます。必要なら `source_center`、`source_radius`、`n_points` を調整してください。quiver は `skip` で間引き、`factor` で矢印の長さを調整します。

## 重ね描き

### 同じ plotter に追加する

各 PyVista API は `plotter=` を受け取ります。同じ plotter に複数の layer を追加すると、scalar slice、volume outline、vector streamlines を同じ 3D scene に重ねられます。

```python
plotter = data.phisp[-1].plot3d(
    mode="slice",
    cmap="coolwarm",
    clim=(-50, 50),
    show=False,
)

data.j1xyz[-1].plot3d(
    mode="stream",
    backend="pyvista",
    plotter=plotter,
    tube_radius=0.02,
    color="white",
    show=True,
)
```

境界メッシュも同じ plotter に追加できます。`data.boundaries.plot3d()` は collection 全体、`data.boundaries[0].plot3d()` は単独境界、`data.boundaries.mesh().plot3d()` は構築済み `MeshSurface3D` を描画します。

```python
plotter = data.phisp[-1].plot3d(mode="slice", show=False)
data.boundaries.plot3d(
    plotter=plotter,
    color="0.7",
    opacity=0.35,
    show_edges=True,
    show=True,
)
```

境界メッシュ生成引数は `mesh_kwargs`、境界 index ごとの上書きは `per` で渡します。Matplotlib / `plot_surfaces()` 経路の詳細は [境界メッシュ](boundaries.ja.md) を参照してください。

### Backtrace / trace 軌跡を重ねる

`data.trace.*(..., get_trace=True)` の戻り値は `plot3d()` を持ち、既存 plotter に軌跡を追加できます。確率を計算している場合は、`alpha="auto"` で軌道ごとの確率を透明度に使います。

```python
trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=(-5.0, 5.0, 16),
    vy=0.0,
    vz=(-5.0, 5.0, 16),
    get_trace=True,
)

plotter = data.phisp[-1].plot3d(mode="slice", show=False)
trace.plot3d(
    plotter=plotter,
    direction="forward",
    tube_radius=0.05,
    color="black",
    show=True,
)
```

`get_probabilities=False` で trace だけ作った場合、確率由来の alpha はありません。必要なら `alpha=0.3` のように明示してください。

## 単位と軸順序

グリッドデータの軸順序は `(t, z, y, x)`、3D volume は `(z, y, x)` です。PyVista helper はこの順序を読み取り、PyVista 側の `(x, y, z)` 座標へ並べ替えます。

`use_si=True` が既定で、単位 metadata がある場合は座標と値を SI 単位へ変換します。単位 metadata がないデータでは `use_si=True` を渡しても内部的に grid / raw 値のまま描画します。

`offsets=(x_offset, y_offset, z_offset)` には数値または `"left"` / `"center"` / `"right"` を指定できます。複数 layer を重ねるときは、すべての layer で同じ `use_si` と `offsets` を使ってください。

## リモート実行と HPC

PyVista 描画は plotter をローカル Python プロセス上に作る API です。`data.remote()` / `remote_figure()` は Matplotlib 画像レンダリング向けなので、`Data3d.plot3d()` の PyVista scene を worker に保持して再描画する用途ではありません。

KUDPC などの login node では、重い PyVista 描画や大きな 3D データ読み込みを直接実行せず、計算ノードや対応する可視化環境へ回してください。スクリプト内では `show=False` で plotter を作り、`screenshot()` で画像保存する形にすると batch 実行しやすくなります。

```python
plotter = data.phisp[-1].plot3d(mode="slice", show=False)
plotter.screenshot("phisp_slice.png")
plotter.close()
```

## 低水準 helper

通常は `plot3d()` / `plot_pyvista()` を使います。PyVista mesh を自分で加工したい場合だけ、`emout.plot.pyvista_plot` の helper を直接使います。

```python
from emout.plot.pyvista_plot import (
    create_plane_mesh,
    create_surface_mesh,
    create_volume_mesh,
    create_vector_mesh3d,
)

mesh, scalar_name, axis_labels, scalar_label = create_volume_mesh(data.phisp[-1])
```

低水準 helper は PyVista オブジェクトを返すだけで、emout の remote rendering や article recording の高水準 API ではありません。

## よくあるエラー

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| `ModuleNotFoundError: pyvista` | 古い環境または依存関係が未更新 | `pip install -U emout` |
| `Data2d with time axis is not supported` | `t` 軸を含む 2D slice を渡した | 時刻を 1 つに固定して空間 2D slice にする |
| `plot_pyvista ... requires spatial axes x,y,z` | 3D 空間軸が揃っていない | `data.phisp[-1]` のように time だけ固定する |
| streamlines が出ない | seed 数や seed 半径が小さい | `n_points` / `source_radius` を増やす |
| layer がずれる | `use_si` や `offsets` が layer 間で違う | すべての layer で同じ指定にする |
