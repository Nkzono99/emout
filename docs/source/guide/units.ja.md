# 単位変換 (`data.unit`)

emout は EMSES 内部（規格化）単位と SI 単位の双方向変換を提供します。

## 前提条件

単位変換にはパラメータファイルに変換キーが必要です。

**`plasma.inp` 形式** — 1 行目:

```text
!!key dx=[0.5],to_c=[10000.0]
```

**`plasma.toml` 形式:**

```toml
[meta.unit_conversion]
dx = 0.5
to_c = 10000.0
```

- `dx`: グリッド間隔（メートル [m]）
- `to_c`: EMSES 規格化単位での光速

キーが存在しない場合、`data.unit` は `None` になります。
この状態では `data.unit.v.trans(...)` / `.reverse(...)` や `data.phisp[-1].val_si`
はすべて `AttributeError` になるので、次のいずれかで対処してください:

- **推奨:** `plasma.inp` 先頭に `!!key ...` 行を追加する
  （`plasma.toml` なら `[meta.unit_conversion]` セクションを追加）
- 単位変換なしで解析する場合は、`use_si=False` を指定して生のグリッド値で作業する
  （`plot(use_si=False)`、`np.asarray(data.phisp[-1])` など）
- スクリプト内で分岐する場合は `if data.unit is not None:` でガードする

## 単位変換器の使い方

各物理量は `data.unit.<名前>` で `UnitTranslator` にアクセスできます:

```python
import emout

data = emout.Emout("output_dir")

# SI → EMSES
emses_velocity = data.unit.v.trans(1e5)     # 1e5 m/s → EMSES

# EMSES → SI
si_velocity = data.unit.v.reverse(4.107)    # 4.107 EMSES → m/s
```

### `trans(value)` と `reverse(value)`

| メソッド | 変換方向 | 用途 |
| --- | --- | --- |
| `trans(x)` | SI → EMSES | 初期条件の設定、理論値との比較 |
| `reverse(x)` | EMSES → SI | シミュレーション結果の解釈 |

> **覚え方:** `trans` は「simulation に translate（持ち込む）」、
> `reverse` は「reality（SI）に revert（戻す）」。方向を迷ったら
> 「解析するときはほぼ必ず `reverse`」と覚えておくと実用的です。

## データからの SI 値の直接取得

グリッドデータ配列の `.val_si` プロパティで SI 単位の値を取得できます:

```python
# 電位 [V]
phisp_V = data.phisp[-1].val_si

# 電流密度 [A/m^2]
j1z_A_m2 = data.j1z[-1].val_si

# 数密度 [/m^3]
nd1p_m3 = data.nd1p[-1].val_si
```

スライスしたデータでも使えます:

```python
# 2D スライスの SI 値
phi_slice = data.phisp[-1, :, 32, :].val_si
```

## プロットでの SI 単位

デフォルトでは `plot()` は軸ラベルとカラーバーに SI 単位を使用します:

```python
data.phisp[-1, 100, :, :].plot()              # SI 単位（デフォルト）
data.phisp[-1, 100, :, :].plot(use_si=False)  # EMSES 単位
```

## 利用可能な単位変換器

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

## 時間軸の単位カスタマイズ

デフォルトでは時間軸は秒 (SI) で表示されます。プラズマ周波数規格化時間（$\omega_{pe} t$）に切り替えることもできます:

```python
from emout.emout.units import wpet_unit

# 以降のすべてのプロットに対してグローバルに登録
emout.Emout.name2unit["t"] = wpet_unit
```

## 変換器の合成

`UnitTranslator` オブジェクトは乗算で合成できます:

```python
# 出力ステップ → SI 秒の変換
t_translator = data.unit.t * UnitTranslator(data.inp.ifdiag * data.inp.dt, 1)
```
