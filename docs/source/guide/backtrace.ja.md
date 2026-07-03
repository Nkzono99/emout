# バックトレース (`data.trace`) — 実験的

`data.trace` は、6D 位相空間グリッドから粒子を作り、到達確率、backward trace、forward trace をまとめて扱う workflow API です。新しい解析コードでは `data.trace.forward()` / `data.trace.backward()` / `data.trace.both()` を使ってください。

従来の `data.backtrace` は、単一粒子や任意の `Particle` 配列を直接渡したい既存コード向けの低水準 API として残っています。このページでは本線を `data.trace` にし、`data.backtrace` の経路は折りたたみ内にまとめています。

> **要件:** バックトレースは外部の [`vdist-solver-fortran`](https://github.com/Nkzono99/vdist-solver-fortran) パッケージ (`vdsolverf`) に依存します。`pip install vdist-solver-fortran` でインストールしてください。未インストールの環境では `data.trace.*` の呼び出し時に `ImportError` が出ます。

## 入力単位の約束

`data.trace` に渡す `x` / `y` / `z` / `vx` / `vy` / `vz`、`dt`、`probability_dt` は **すべて EMSES シミュレーション単位**です。emout は SI から自動変換しません。

SI 値から指定したい場合は、呼び出し前に `data.unit` で EMSES 単位へ変換してください:

```python
position = (
    data.unit.length.trans(0.20),  # m -> EMSES length
    data.unit.length.trans(0.32),
    data.unit.length.trans(0.40),
)
vx_scan = (
    data.unit.v.trans(-3.0e5),     # m/s -> EMSES velocity
    data.unit.v.trans(3.0e5),
    64,
)
vz_scan = (
    data.unit.v.trans(-3.0e5),
    data.unit.v.trans(3.0e5),
    64,
)
```

`data.unit` は `plasma.inp` の `!!key dx=...,to_c=...` ヘッダー（または `plasma.toml` の `[meta.unit_conversion]`）がある場合にだけ利用できます。単位変換 metadata がない場合は、EMSES 単位の値を直接渡してください。

`TraceResult.phases`、`TraceResult.particles`、`trace.traces.positions_list` などの配列も EMSES 単位のまま保持されます。一方で、`trace.plot()` / `trace.plot_traces()` / `trace.traces.xz.plot()` は unit metadata がある場合に既定で SI 単位へ変換して表示します（`use_si=False` で EMSES 単位表示）。

`dt` と `probability_dt` は EMSES 時間単位の非負の刻み幅です。`None` の場合は `abs(data.inp.dt)` を使います。`data.trace.backward()` は solver に `+dt`、`data.trace.forward()` は内部で符号を反転して `-dt` を渡します。`data.trace.both()` は同じ `dt` から backward に `+dt`、forward に `-dt` を使います。到達確率計算は `probability_dt` を backward 側の符号で使います。負の値を渡すと `ValueError` になります。

## いつ使うか

- ある観測点に到達する粒子の **位相空間分布** を見たい
- 到達粒子を **backward / forward の軌跡** として追いたい
- 到達粒子の **エネルギースペクトル** を描きたい
- 確率と軌跡を同じ粒子集合から計算して、確率を alpha に使いたい

バックトレースは保存済みフィールドを使って ODE を解くため、大きな `max_step` や細かい位相空間グリッドでは時間がかかります。HPC ノードに処理を任せたい場合は、リモート実行と組み合わせてください（後述）。

## クイックスタート

```python
import emout

data = emout.Emout("output_dir")

vx_scan = (data.unit.v.trans(-3e5), data.unit.v.trans(3e5), 64)
vz_scan = (data.unit.v.trans(-3e5), data.unit.v.trans(3e5), 64)

trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    ispec=0,
    get_trace=True,
    get_probabilities=True,
    max_step=10000,
    n_threads=8,
)

trace.plot("vx", "vz", cmap="viridis")       # 到達確率 heatmap
trace.plot_traces("x", "z")                  # 確率 alpha 付き軌跡
trace.probabilities.plot_energy_spectrum(scale="log")
```

スカラー値だけを渡すと、位相空間グリッドのサイズ 1 軸として扱われます。確率計算なしで 1 本の軌跡だけ欲しい場合も `data.trace` で書けます:

```python
single = data.trace.backward(
    x=20.0, y=32.0, z=40.0,
    vx=data.unit.v.trans(1.0e5),
    vy=0.0,
    vz=data.unit.v.trans(-2.0e5),
    ispec=0,
    get_trace=True,
    get_probabilities=False,
)

single.plot_traces("t", "x")
single.traces.xvz.plot()
```

## Workflow API: `data.trace`

`data.trace.backward()` / `data.trace.forward()` / `data.trace.both()` は常に `TraceResult` を返します。要求しなかった payload は `None` です。

### 確率と軌跡をまとめて得る

```python
trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    get_trace=True,
    get_probabilities=True,
)

trace.probabilities        # ProbabilityResult
trace.forward_traces       # MultiBacktraceResult
trace.backward_traces      # None
trace.alpha                # np.clip(trace.probabilities.probabilities, 0, 1)

trace.plot("vx", "vz")     # 到達確率 heatmap
trace.plot_traces("x", "z")
```

`trace.plot()` は確率がある場合は `trace.probabilities.pair(...).plot()`、確率がない場合は `trace.plot_traces()` に自動で振り分けます。明示したい場合は `kind="probability"` / `kind="trace"` を渡します。

### 軌跡だけを得る

`get_probabilities=False` にすると、確率計算を省略して phase-space grid から粒子だけを作り、軌跡だけを得ます。この場合 `trace.probabilities` と `trace.alpha` は `None` で、`plot_traces()` は一様な alpha で描画します。

```python
trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    get_trace=True,
    get_probabilities=False,
)

trace.forward_traces.xz.plot(alpha=0.3)
trace.plot_traces("x", "z", alpha=0.3)
```

### backward と forward を同時に得る

`both()` は同じ phase-space grid から backward / forward の両方を計算します。確率も要求した場合、確率計算は 1 回だけです。

```python
trace = data.trace.both(..., get_trace=True)
trace.backward_traces.xz.plot(alpha=trace.alpha)
trace.forward_traces.xz.plot(alpha=trace.alpha)
trace.plot_traces("x", "z", direction="backward")
trace.plot_traces("x", "z", direction="forward")
```

### 3D に重ねる

`plot3d()` は PyVista plotter を返します。既存の 3D field / boundary 表示に重ねたい場合は、その plotter を渡します。

```python
plotter = data.phisp[-1].plot3d(mode="slice", show=False)
trace.plot3d(plotter=plotter, direction="forward", tube_radius=0.05, show=True)
```

## 結果の描画

### 2D ヒートマップへの射影

`trace.probabilities.pair(var1, var2)` は選ばれていない 4 軸を台形則で積分し、`HeatmapData` を返します。`trace.plot(var1, var2)` はこの操作を短く書く入口です。

```python
trace.plot("vx", "vz", cmap="viridis")
trace.probabilities.xvx.plot()
trace.probabilities.yz.plot(cmap="plasma")
```

`HeatmapData.plot()` はカラーバー付きの `pcolormesh` を描き、軸ラベルには SI 単位が付きます（`use_si=False` でグリッド単位に戻せます）。追加のキーワード引数はすべて `pcolormesh` にそのまま渡されるので、`vmin` / `vmax` や `norm=LogNorm(...)` で配色を制御できます。

### エネルギースペクトル

エネルギースペクトルは確率 payload 側の `ProbabilityResult` から描きます。

```python
trace.probabilities.plot_energy_spectrum(scale="log", energy_bins=80)
hist, bin_edges = trace.probabilities.energy_spectrum(energy_bins=80)
```

内部で `plasma.inp` の `wp` や光電流設定から基準数密度 `n0` を計算し、各位相点の確率に掛けてから積分しています（`nflag_emit == 2` の光電子は `path` と `curf` から導出）。

## リモート実行との連携

`data.trace` は `Emout` ファサード経由で `remote_open_kwargs` を持ち、emout server が動いていれば既定でワーカー上で実行されます（結果は `RemoteTraceResult` プロキシとして返ります）。ワーカー上にキャッシュされているので、プロットパラメータを変えても再計算は走りません。

```python
from emout.distributed import remote_figure

trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    get_trace=True,
)

with remote_figure():
    trace.plot("vx", "vz", cmap="viridis")

with remote_figure():
    trace.plot_traces("x", "z")

trace.drop()   # 不要になったらワーカーメモリを解放
```

明示的な remote 記法を使いたいときは `data.remote().trace...` に切り替えます:

```python
from emout.distributed import remote_scope, remote_figure

with remote_scope():
    rdata = data.remote()
    trace = rdata.trace.forward(
        x=20.0, y=32.0, z=40.0,
        vx=vx_scan,
        vy=0.0,
        vz=vz_scan,
        get_trace=True,
    )

    with remote_figure():
        trace.plot("vx", "vz")
        trace.plot_traces("x", "z")
```

リモート実行全体の仕組み・環境変数・サーバー起動については [リモート実行ガイド](distributed.ja.md) を参照してください。

### `fetch()` でローカルに取り出す

matplotlib で細かくカスタマイズしたい（独自アノテーション、共通カラーバー、subplot への差し込み、など）場合は、`fetch()` で結果を小さなローカルオブジェクトに変換できます:

```python
local_trace = trace.fetch()
heatmap = local_trace.probabilities.vxvz
fig, ax = plt.subplots()
heatmap.plot(ax=ax, cmap="plasma")
ax.axhline(y=0, color="red", linestyle="--")
```

<details>
<summary>既存コード向け: 低水準 API `data.backtrace` を表示</summary>

## 低水準 API: `data.backtrace`

`data.backtrace` は、`data.trace` が内部で使っている `BacktraceWrapper` です。1 本の軌跡を明示的な `position` / `velocity` で計算したい場合、`vdsolverf.core.Particle` を直接渡したい場合、または solver に渡す `dt` の符号を直接制御したい場合に使います。新しい位相空間 workflow は `data.trace` を優先してください。

### 単一粒子: `get_backtrace`

```python
position = (20.0, 32.0, 40.0)
velocity = (
    data.unit.v.trans(1.0e5),
    0.0,
    data.unit.v.trans(-2.0e5),
)

bt = data.backtrace.get_backtrace(position, velocity, ispec=0, max_step=50000)

bt.tx.plot()                 # = bt.pair("t", "x")
bt.xvz.plot()                # = bt.pair("x", "vz")
bt.yz.plot(color="black")    # 軌道の yz 投影
```

`bt.ts`、`bt.probability`、`bt.positions`、`bt.velocities` は EMSES 単位の配列です。`XYData.plot()` はデフォルトで SI 単位に変換し、軸ラベルも自動生成します（`use_si=False` で EMSES 単位のまま表示できます）。

### 複数粒子: `get_backtraces`

```python
import numpy as np

positions = np.array([[20, 32, 40], [21, 32, 40], [22, 32, 40]], dtype=float)
velocities = np.zeros_like(positions)
velocities[:, 0] = data.unit.v.trans(1.0e5)

many = data.backtrace.get_backtraces(positions, velocities, ispec=0)
many.xz.plot(alpha=0.5)
many.sample(50, random_state=0).tvx.plot()
```

`positions` と `velocities` は対応する粒子ごとの `(N, 3)` 配列です。Cartesian product の位相空間グリッドを作りたい場合は `data.trace` を使ってください。

### Particle オブジェクトを直接渡す

```python
from vdsolverf.core import Particle

particles = [Particle(p, v) for p, v in zip(positions, velocities)]
many = data.backtrace.get_backtraces_from_particles(particles, ispec=0)
```

`ProbabilityResult` から得られた粒子リストをそのまま渡して、確率が高い粒子だけ軌跡を描く、といった連結もできます:

```python
result = data.backtrace.get_probabilities(...)
bt = data.backtrace.get_backtraces_from_particles(result.particles, ispec=0)
bt.xz.plot(alpha=np.clip(result.probabilities, 0, 1))
```

### 到達確率: `get_probabilities`

`data.trace` は内部で `data.backtrace.get_probabilities(...)` を呼び、`ProbabilityResult` を `trace.probabilities` として保持します。低水準 API を直接使う場合は次の形です:

```python
result = data.backtrace.get_probabilities(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    ispec=0,
    max_step=10000,
    n_threads=8,
)

result.vxvz.plot(cmap="viridis")
result.plot_energy_spectrum(scale="log")
```

### MPI backend

`parallel="mpi"` / `parallel="srun"` は低水準 `get_probabilities()` の backend オプションです。`data.trace` にも `**kwargs` として渡せます。

```python
trace = data.trace.forward(
    x=20.0, y=32.0, z=40.0,
    vx=vx_scan,
    vy=0.0,
    vz=vz_scan,
    max_step=10000,
    parallel="srun",
    ntasks=8,
    n_threads=2,
    cpus_per_task=2,
)
```

</details>

## 関連クラス

詳細なシグネチャは API リファレンス（`emout.core.backtrace` パッケージ）を参照してください。

- `TraceWrapper` — `data.trace` の実体
- `TraceResult` — 確率・軌跡 payload をまとめる workflow 結果
- `BacktraceWrapper` — `data.backtrace` の実体（低水準 API）
- `BacktraceResult` / `MultiBacktraceResult` — 軌跡コンテナ
- `ProbabilityResult` — 6D 確率グリッドとヒートマップ射影
- `XYData` / `MultiXYData` / `HeatmapData` — 可視化用の軽量コンテナ
