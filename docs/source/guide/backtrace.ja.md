# バックトレース (`data.backtrace`) — 実験的

`data.backtrace` は、EMSES の電磁場に対して粒子の軌道を逆向きに積分する
バックトレース機能への入口です。到達確率（`get_probabilities`）や
個別粒子の軌跡（`get_backtrace` / `get_backtraces`）を計算でき、
結果は専用コンテナに包まれて返るので `.vxvz.plot()` のような短い記法で
可視化まで一直線につなげられます。

> **要件:** バックトレースは外部の [`vdist-solver-fortran`](https://github.com/Nkzono99/vdist-solver-fortran)
> パッケージ (`vdsolverf`) に依存します。`pip install vdist-solver-fortran` で
> インストールしてください。未インストールの環境では `data.backtrace.*` の
> 呼び出し時に `ImportError` が出ます。

## いつ使うか

- ある観測点に粒子が到達する **位相空間分布** を見たい
- 注目した粒子がどの領域から来たのかを **軌跡として追いたい**
- 到達粒子の **エネルギースペクトル** を描きたい

バックトレースは `data.inp.dt` と EMSES の保存済みフィールドを使って
後ろ向きに ODE を解くため、大きな `max_step` を指定するとそれなりに時間が
かかります。HPC ノードに処理を任せたい場合は、リモート実行と
組み合わせてください（後述）。

## クイックスタート

```python
import emout

data = emout.Emout("output_dir")

# 単一粒子のバックトレース
position = (20.0, 32.0, 40.0)
velocity = (1.0e5, 0.0, -2.0e5)
bt = data.backtrace.get_backtrace(position, velocity, ispec=0)

bt.tx.plot()      # t vs x の軌跡
bt.xvz.plot()     # x vs vz 位相空間

# 複数粒子を一括処理
import numpy as np
positions = np.array([[20, 32, 40], [21, 32, 40], [22, 32, 40]], dtype=float)
velocities = np.zeros_like(positions)
velocities[:, 0] = 1.0e5
many = data.backtrace.get_backtraces(positions, velocities, ispec=0)

many.xz.plot(alpha=0.5)    # 全軌跡を重ね描き

# 6D 位相空間での到達確率
result = data.backtrace.get_probabilities(
    x=20.0, y=32.0, z=40.0,
    vx=(-3e5, 3e5, 64),
    vy=0.0,
    vz=(-3e5, 3e5, 64),
    ispec=0,
)

result.vxvz.plot(cmap="viridis")      # vx-vz 平面上のヒートマップ
result.plot_energy_spectrum(scale="log")
```

## 単一粒子：`get_backtrace`

`get_backtrace(position, velocity, ispec=0, ...)` は 1 本の軌跡を計算し、
:class:`BacktraceResult` を返します。戻り値はタプル展開もできます。

```python
bt = data.backtrace.get_backtrace(position, velocity, ispec=0, max_step=50000)

ts, prob, positions, velocities = bt   # タプル展開
print(bt)                                # <BacktraceResult: n_steps=...>
```

| 属性 | 形状 | 意味 |
| --- | --- | --- |
| `bt.ts` | `(N,)` | 時刻（EMSES 単位） |
| `bt.probability` | `(N,)` | 到達確率の時系列 |
| `bt.positions` | `(N, 3)` | `[x, y, z]` |
| `bt.velocities` | `(N, 3)` | `[vx, vy, vz]` |

### ショートハンドでの可視化

`bt.pair(var1, var2)` は 2 変数を切り出して :class:`XYData` を返します。
`var1`, `var2` には `t`, `x`, `y`, `z`, `vx`, `vy`, `vz` が使え、連結
属性名 (`bt.tx`, `bt.xvz`, `bt.yz`, ...) としても取り出せます。

```python
bt.tx.plot()                 # = bt.pair("t", "x")
bt.xvz.plot()                # = bt.pair("x", "vz")
bt.yz.plot(color="black")    # 軌道の xy 投影
```

`XYData.plot()` はデフォルトで SI 単位に変換し、軸ラベルも自動生成します
（`use_si=False` で EMSES 単位のまま表示できます）。`gap=...` を渡すと、
周期境界のジャンプなど長い飛びを NaN で切って線が繋がらないようにできます。

## 複数粒子：`get_backtraces`

`get_backtraces(positions, velocities, ispec=0, n_threads=4, ...)` は
:class:`MultiBacktraceResult` を返します。`positions` と `velocities` は
`(N, 3)` 配列であることが必要です。

```python
ts, probs, pos_list, vel_list, last = many
many.xz.plot(alpha=np.clip(probs, 0, 1))    # 確率に応じた透明度
many.sample(50, random_state=0).tvx.plot()   # ランダムに 50 本だけ表示
many.sample(slice(0, 10)).tx.plot()         # 先頭 10 本を選ぶ
```

| 属性 | 形状 | 意味 |
| --- | --- | --- |
| `ts_list` | `(N_traj, N_steps)` | |
| `probabilities` | `(N_traj,)` | 最終到達確率 |
| `positions_list` | `(N_traj, N_steps, 3)` | |
| `velocities_list` | `(N_traj, N_steps, 3)` | |
| `last_indexes` | `(N_traj,)` | 有効データの終端 index（パディング用） |

`many.pair("t", "x")` は :class:`MultiXYData` を返し、`.plot()` で
全軌跡を重ね描きします。`alpha` はスカラーでもスカラーの配列
（長さ `N_traj`）でも渡せるので、確率の重み付けに便利です。

### Particle オブジェクトから直接実行

`vdsolverf.core.Particle` をすでに手元に持っている場合
（例えば `ProbabilityResult.particles` をそのまま流したい）、
`get_backtraces_from_particles(particles, ...)` が使えます。

```python
from vdsolverf.core import Particle

particles = [Particle(p, v) for p, v in zip(positions, velocities)]
many = data.backtrace.get_backtraces_from_particles(particles, ispec=0)
```

`ProbabilityResult` から得られた粒子リストをそのまま渡して、
確率が高い粒子だけ軌跡を描く、といった連結が可能です:

```python
result = data.backtrace.get_probabilities(...)
bt = data.backtrace.get_backtraces_from_particles(result.particles, ispec=0)
bt.xz.plot(alpha=np.clip(result.probabilities, 0, 1))
```

## 到達確率：`get_probabilities`

`get_probabilities(x, y, z, vx, vy, vz, ispec=0, ...)` は 6 次元の位相空間
グリッドを切って、各グリッド点から粒子をバックトレースし、到達確率を
:class:`ProbabilityResult` として返します。

各軸は次のいずれかで指定できます:

- `(start, stop, n)` のタプル — 等間隔グリッド
- 明示的な配列やリスト — 任意の値
- スカラー — サイズ 1 の軸（後で `pair()` すると自動的にスクイーズされます）

```python
result = data.backtrace.get_probabilities(
    x=20.0, y=32.0, z=40.0,     # 位置は固定点
    vx=(-3e5, 3e5, 64),         # vx を 64 点で走査
    vy=0.0,
    vz=(-3e5, 3e5, 64),         # vz を 64 点で走査
    ispec=0,
    max_step=10000,
    n_threads=8,
)
```

### 2D ヒートマップへの射影

`result.pair(var1, var2)` は選ばれていない 4 軸を台形則で積分し、
:class:`HeatmapData` を返します。`bt` と同じく連結属性でもアクセスできます。

```python
result.vxvz.plot(cmap="viridis")   # = result.pair("vx", "vz")
result.xvx.plot()                  # x-vx 平面
result.yz.plot(cmap="plasma")      # y-z 平面
```

`HeatmapData.plot()` はカラーバー付きの `pcolormesh` を描き、軸ラベルには
SI 単位が付きます（`use_si=False` でグリッド単位に戻せます）。
追加のキーワード引数はすべて `pcolormesh` にそのまま渡されるので、
`vmin` / `vmax` や `norm=LogNorm(...)` で配色を制御できます。
`offsets=("center", 0)` のように渡すと、軸をセンタリングしたり
数値シフトを掛けたりもできます。

### エネルギースペクトル

`plot_energy_spectrum(energy_bins=None, scale="log")` は到達粒子の
エネルギーフラックスヒストグラムを描きます。ビンは整数（本数）、
配列（ビン境界）のどちらも受け付けます。

```python
result.plot_energy_spectrum(scale="log", energy_bins=80)
```

内部で `plasma.inp` の `wp` や光電流設定から基準数密度 `n0` を計算し、
各位相点の確率に掛けてから積分しています（`nflag_emit == 2` の光電子は
`path` と `curf` から導出）。

### ヒストグラム配列だけ欲しい場合

`result.energy_spectrum(energy_bins=...)` は `(hist, bin_edges)` を返すので、
独自の加工や別ライブラリ（例：matplotlib の `ax.step`、seaborn）への
受け渡しに使えます。

## リモート実行との連携

`data.backtrace` は `Emout` ファサード経由で `remote_open_kwargs` を持ち、
emout server が動いていれば自動的にワーカー上で実行されます
（結果は `RemoteProbabilityResult` / `RemoteBacktraceResult` プロキシとして返ります）。
ワーカー上にキャッシュされているので、プロットパラメータを変えても
再計算は走りません。

```python
from emout.distributed import remote_figure

result = data.backtrace.get_probabilities(...)   # ワーカー上で 1 回だけ計算

with remote_figure():
    result.vxvz.plot(cmap="viridis")

with remote_figure():
    result.plot_energy_spectrum(scale="log")

result.drop()   # 不要になったらワーカーメモリを解放
```

明示的な remote 記法を使いたいときは `data.remote().backtrace...` に
切り替えます。返り値は同じ専用プロキシです:

```python
from emout.distributed import remote_scope, remote_figure

with remote_scope():
    rdata = data.remote()
    bt = rdata.backtrace.get_backtrace(position, velocity, ispec=0)
    result = rdata.backtrace.get_probabilities(...)

    with remote_figure():
        bt.tx.plot()
        result.vxvz.plot()
```

リモート実行全体の仕組み・環境変数・サーバー起動については
[リモート実行ガイド](distributed.ja.md) を参照してください。

### `fetch()` でローカルに取り出す

matplotlib で細かくカスタマイズしたい（独自アノテーション、共通カラーバー、
subplot への差し込み、など）場合は、`fetch()` で結果を小さなローカル
オブジェクトに変換できます:

```python
heatmap = result.vxvz.fetch()      # ローカルの HeatmapData
fig, ax = plt.subplots()
heatmap.plot(ax=ax, cmap="plasma")
ax.axhline(y=0, color="red", linestyle="--")
```

## 関連クラス

詳細なシグネチャは API リファレンス（`emout.core.backtrace` パッケージ）を参照してください。

- `BacktraceWrapper` — `data.backtrace` の実体
- `BacktraceResult` / `MultiBacktraceResult` — 軌跡コンテナ
- `ProbabilityResult` — 6D 確率グリッドとヒートマップ射影
- `XYData` / `MultiXYData` / `HeatmapData` — 可視化用の軽量コンテナ
