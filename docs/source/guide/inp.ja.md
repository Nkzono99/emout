# パラメータファイル (`data.inp`)

emout は EMSES のパラメータファイル（`plasma.inp` / `plasma.toml`）を読み込み、属性アクセスできる辞書風のオブジェクトとして提供します。

## パラメータへのアクセス

```python
import emout

data = emout.Emout("output_dir")

# 辞書スタイル（グループ名 + パラメータ名）
data.inp["tmgrid"]["nx"]    # → 例: 256
data.inp["plasma"]["wp"]    # → 例: [1.0, 0.05]

# グループ名を省略（名前が一意なら）
data.inp["nx"]              # → data.inp["tmgrid"]["nx"] と同じ

# 属性スタイル
data.inp.tmgrid.nx
data.inp.nx                 # グループ名省略
```

> **注意: 省略形は重複名を検出しません。** 同じ名前が複数グループに存在する場合、
> `data.inp["nx"]` はエラーを出さず **たまたま最初にヒットしたグループの値**
> を返します。重複の可能性があるパラメータ（例えばどちらのグループに属するか
> 曖昧なもの）は、必ず `data.inp["group_name"]["param"]` のように明示してください。

## よく使うパラメータ

### グリッド

| パラメータ | グループ | 説明 |
| --- | --- | --- |
| `nx`, `ny`, `nz` | `tmgrid` | グリッドの次元数 |
| `dx`, `dy`, `dz` | `tmgrid` | グリッド間隔（EMSES 単位。SI は `data.unit.length` を参照） |

```python
nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz
```

### 時間

| パラメータ | グループ | 説明 |
| --- | --- | --- |
| `dt` | `tmgrid` | 時間ステップ |
| `ifdiag` | `tmgrid` | 出力間隔（`ifdiag` ステップごとに出力） |
| `nstep` | `tmgrid` | 総ステップ数 |

```python
dt = data.inp.dt
total_time_steps = data.inp.nstep
output_interval = data.inp.ifdiag
```

### プラズマ

| パラメータ | グループ | 説明 |
| --- | --- | --- |
| `nspec` | `plasma` | 粒子種数 |
| `wp` | `plasma` | 種ごとのプラズマ周波数 |
| `qm` | `plasma` | 種ごとの比電荷 |
| `path` | `plasma` | 種ごとの熱速度 |
| `peth` | `plasma` | 熱速度（別名） |

```python
nspec = data.inp.nspec
wp = data.inp.wp        # プラズマ周波数のリスト
qm = data.inp.qm        # 比電荷のリスト
```

### 境界条件

| パラメータ | グループ | 説明 |
| --- | --- | --- |
| `mtd_vbnd` | `emissn` | 各軸の境界タイプ（0=周期, 1=ディリクレ, 2=ノイマン） |

```python
btypes = data.inp.mtd_vbnd  # 例: [0, 2, 0] → 周期-ノイマン-周期
```

## TOML 形式 (`plasma.toml`)

`plasma.toml` が存在する場合、emout は自動的に `toml2inp` を実行して `plasma.inp` を生成し、それを読み込みます。
`toml2inp` コマンドは [MPIEMSES3D](https://github.com/Nkzono99/MPIEMSES3D) に同梱されています。

```python
data = emout.Emout("output_dir")
data.inp.nx          # ファイル形式に関係なく同じインターフェース
```

### 生の TOML 構造へのアクセス

`data.toml` で TOML のネイティブ構造に直接アクセスできます:

```python
data.toml                        # TomlData ラッパー（plasma.inp のみの場合は None）
data.toml.tmgrid.nx              # 属性アクセス
data.toml["tmgrid"]["nx"]        # 辞書アクセス
data.toml.species[0].wp          # ネスト構造へのアクセス
```

TOML 形式では namelist グループに対応するセクションヘッダーを使います:

```toml
[tmgrid]
nx = 256
ny = 256
nz = 512
dt = 0.5

[[species]]
wp = 1.0
qm = -1.0

[[species]]
wp = 0.05
qm = 0.001
```

### 入力ファイルと出力ディレクトリの分離

```python
data = emout.Emout(input_path="/path/to/plasma.toml", output_directory="output_dir")
```

## 単位変換キー

`plasma.inp` の 1 行目に単位変換キーを記述できます:

```text
!!key dx=[0.5],to_c=[10000.0]
```

- `dx`: グリッド間隔（メートル [m]）
- `to_c`: EMSES 内部単位での光速

このキーにより `data.unit` を通じた SI 単位変換が有効になります。`plasma.toml` では以下のように記述します:

```toml
[meta.unit_conversion]
dx = 0.5
to_c = 10000.0
```

変換キーが存在しない場合、`data.unit` は `None` になり、SI 関連機能（`val_si`, `use_si=True`）は EMSES の生の単位のまま動作します。
