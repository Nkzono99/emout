# 境界メッシュ (`data.boundaries`)

MPIEMSES の `&ptcond` で定義した finbound / legacy 境界を、
`data.boundaries` から Python オブジェクトとして触れます。各境界は
`MeshSurface3D` を返すので、3D フィールドプロットに重ね描きしたり、個別にスタイルを変えたりできます。

> **注意: `mesh()` の単位は SI がデフォルト**。`data.boundaries[0].mesh()` は
> 既定で SI 単位 (m) の頂点座標を返します。グリッド単位のメッシュが欲しい場合
> （例: EMSES のグリッド上の別処理にそのまま渡したい場合）は
> `mesh(use_si=False)` を明示してください。`plot_surfaces()` へオーバーレイするときは、
> フィールド側の軸単位と揃えておかないと境界だけずれて見える原因になります。

## アクセス

```python
# コレクション全体
data.boundaries                 # BoundaryCollection (iterable, indexable)
len(data.boundaries)
data.boundaries.types           # 各境界の boundary_types 文字列リスト
data.boundaries.skipped         # [(index, type_name, reason), ...]

# 個別の境界
data.boundaries[0]              # サブクラス (SphereBoundary, CylinderBoundary, ...)
data.boundaries[0].btype        # 例: "sphere", "cylinderz"
data.boundaries[0].mesh()       # → MeshSurface3D (デフォルトは SI 単位)
data.boundaries[0].mesh(use_si=False)  # グリッド単位のまま
```

## 3D フィールドプロットへのオーバーレイ

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

## 複合メッシュ

全境界をひとつのメッシュに結合して取り出すこともできます:

```python
composite = data.boundaries.mesh()    # → CompositeMeshSurface
V, F = composite.mesh()               # 生の (頂点, 面) 配列
```

## 合成とスタイル指定

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

## メッシュパラメータのオーバーライド

`mesh()` にキーワード引数を渡すと、シミュレーション設定を変えずに解像度や形状を調整できます:

```python
# 特定の境界の角度分割を細かくする
data.boundaries[0].mesh(ntheta=64)

# コレクション経由で個別にオーバーライド
data.boundaries.mesh(per={0: dict(ntheta=64)})
```

## 対応している境界型

`boundary_type = "complex"` モードの `boundary_types(i)` と legacy 単独モードの両方に対応しています。

| カテゴリ | 型名 |
| --- | --- |
| 閉じた立体 | `sphere`, `cuboid` |
| 円柱 | `cylinderx/y/z`, `open-cylinderx/y/z` |
| 平板 | `rectangle`, `circlex/y/z`, `diskx/y/z`, `plane-with-circlex/y/z` |
| Legacy 単独モード | `flat-surface`, `rectangle-hole`, `cylinder-hole` |

パラメータの詳細は [MPIEMSES3D Parameters.md](https://github.com/Nkzono99/MPIEMSES3D/blob/main/docs/Parameters.md) を参照してください。

未登録の型は `data.boundaries.skipped` に `(index, 型名, 理由)` として記録され、エラーにはなりません。
このリストを確認すれば、どの境界が無視されたかが分かります。

## 利用可能なメッシュサーフェスクラス

各境界型は下表の `MeshSurface3D` サブクラスに対応します
（クラス自体は `emout.plot.surface_cut` から直接インポートできます）:

| クラス | 対応する型 |
| --- | --- |
| `SphereMeshSurface` | `sphere` |
| `BoxMeshSurface` | `cuboid` |
| `RectangleMeshSurface` | `rectangle` |
| `CircleMeshSurface` | `circlex/y/z` |
| `CylinderMeshSurface` | `cylinderx/y/z`, `open-cylinderx/y/z` |
| `DiskMeshSurface` | `diskx/y/z` |
| `PlaneWithCircleMeshSurface` | `plane-with-circlex/y/z` |
| `HollowCylinderMeshSurface` | `cylinder-hole` |
| `CompositeMeshSurface` | コレクション全体の `mesh()` 出力 |

これらのクラスは `data.boundaries` を経由せず単独でインスタンス化してメッシュを構築することもできます。
