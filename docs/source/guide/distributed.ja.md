# リモート実行 (Dask) — 実験的

リモート実行は、大規模な EMSES シミュレーション出力を HPC の計算ノード側で読み込み・描画し、ログインノードや手元の Jupyter には小さな結果だけを返すための仕組みです。新規コードでは `Emout.remote()`、`remote_scope()`、`remote_figure()` を組み合わせる形を推奨します。

## 最小ワークフロー

### 1. サーバー起動

ターミナルで 1 回だけ emout server を起動します。SLURM パーティション、メモリ、walltime は環境に合わせて指定してください。

```bash
emout server start --partition gr20001a --memory 60G --walltime 03:00:00
```

InfiniBand の IP と TLS 証明書は自動で設定され、active なセッション情報は `~/.emout/server.json` に保存されます。

```text
Session: default
Scheduler running at tls://10.10.64.2:8786
Detected IP: 10.10.64.2
Workers: 1
```

### 2. スクリプトから使う

`Emout.remote()` で worker 側の `Emout` に切り替え、`remote_scope()` で worker 側オブジェクトの寿命を管理します。`remote_figure()` の中の matplotlib 操作は worker 側で再生され、ローカルには PNG 画像だけが返ります。

```python
import matplotlib.pyplot as plt
import emout
from emout.distributed import remote_figure, remote_scope

rdata = emout.Emout("output_dir").remote()

with remote_scope():
    ymid = int(rdata.inp.ny // 2)
    with remote_figure():
        rdata.phisp[-1, :, ymid, :].plot()
        plt.xlabel("x [m]")
        plt.title("Potential")
```

### 3. サーバー停止

使い終わったら server を停止します。

```bash
emout server stop
```

追加の named session は `emout server stop --name <session>` で停止できます。すべて止めたい場合は `emout server stop --all` を使います。

## モードの選び方

| 目的 | 推奨モード | ローカルへ返るもの |
| --- | --- | --- |
| 大きな field を読まずに図だけ見たい | `Emout.remote()` + `remote_scope()` + `remote_figure()` | PNG/SVG 画像 |
| worker 側 object を何度も再利用したい | `Emout.remote()` + `remote_scope()` | `RemoteRef` / 専用 proxy |
| 既存の `plot()` コードを最小変更で使いたい | 互換モード | 2D slice などの小さな配列 |
| matplotlib を自由にカスタマイズしたい | `fetch()` で小さな結果を取得 | ローカル配列 / lightweight data object |

### 推奨モード（`Emout.remote()` + `remote_scope()`）

worker 上の object を `RemoteRef` として保持しながら、ローカルの `emout` / `numpy` に近い記法で処理を書けます。`-ref`, `ref1 + ref2`, `np.abs(ref)`, `int(ref)` のような式も remote のまま評価されます。

```python
import numpy as np
from emout.distributed import remote_scope

rdata = emout.Emout("output_dir").remote()

with remote_scope():
    phi = rdata.phisp[-1, :, 100, :]
    ex = -rdata.exz[-1, :, 100, :]
    peak = np.abs(ex).max()
    small = phi.fetch()  # 必要なときだけローカルへ取り出す
```

`remote_scope()` の内側で作られた remote object は、`with` を抜けると自動で `drop()` されます。ブロック内では中間結果を何度でも再利用しつつ、ワーカー側のメモリ管理は scope に任せられます。

### 画像モード（`remote_figure`）

`remote_figure()` は matplotlib のコマンドを worker 側で再生し、画像だけを返します。ログインノードに大きな field 配列を載せたくない場合の基本形です。

```python
from emout.distributed import remote_figure

with remote_figure(figsize=(8, 5), dpi=200):
    rdata.phisp[-1, :, 100, :].plot()
    plt.axhline(y=50, color="red")
    plt.title("remote rendered figure")
```

`savefilepath` を指定すると、CLI / バッチ実行でも生成画像をそのまま保存できます。拡張子があれば出力フォーマットはそこから推定されます。

```python
with remote_figure(savefilepath="figure.png"):
    rdata.phisp[-1, :, 100, :].plot()
```

### データ転送モード（互換モード）

active/default セッションが保存されている場合、既存の `plot()` コードも互換モードで動作します。Worker が 2D slice などを切り出してローカルへ転送し、matplotlib はローカルで実行されます。

```python
data = emout.Emout("output_dir")
data.phisp[-1, :, 100, :].plot()
plt.title("local matplotlib")
plt.savefig("output.png")
```

転送されるのはスライス済みの小さな配列だけです。ただし新規コードでは、worker 側 object の寿命が明示できる `Emout.remote()` / `remote_scope()` を推奨します。

## 機能別の連携

### backtrace / `data.trace`

重い粒子バックトレースは server で 1 回だけ実行し、結果を worker のメモリに保持できます。新しい解析コードでは高水準 workflow の `data.trace...` / `data.remote().trace...` を優先してください。

```python
with remote_scope():
    rdata = data.remote()

    trace = rdata.trace.forward(
        x=20.0, y=32.0, z=40.0,
        vx=vx_scan,
        vy=0.0,
        vz=vz_scan,
        ispec=0,
        get_trace=True,
        get_probabilities=True,
    )

    with remote_figure():
        trace.plot("vx", "vz")
        trace.plot_traces("x", "z")

    trace.drop()
```

低水準の `data.backtrace...` / `data.remote().backtrace...` も既存コード向けに利用できます。バックトレース API そのもの（`TraceResult`、`ProbabilityResult`、`BacktraceResult` など）は [バックトレースガイド](backtrace.ja.md) を参照してください。

#### `fetch()` によるローカル加工

matplotlib で自由にカスタマイズしたい場合（独自アノテーション、共有カラーバーなど）は、`fetch()` で結果を小さなローカルオブジェクトとして取り出します。

```python
local_trace = trace.fetch()
heatmap = local_trace.probabilities.vxvz
fig, ax = plt.subplots()
heatmap.plot(ax=ax, cmap="plasma")
ax.axhline(y=0, color="red", linestyle="--")
```

### 境界メッシュ

境界形状そのものは軽量なので、`data.boundaries.plot()` はローカルでも使えます。3D field と重ねる場合は、field 側の 3D 配列をローカルへ読ませないように remote 実行と組み合わせます。

```python
data.boundaries.plot()

with remote_scope():
    rdata = data.remote()
    with remote_figure():
        rdata.phisp[-1].plot_surfaces(surfaces=data.boundaries)
```

詳細は [境界メッシュガイド](boundaries.ja.md) を参照してください。

### アニメーション (`gifplot`)

`gifplot()` も worker 側で実行できます。フレーム生成とエンコードは worker で走り、クライアントにはインライン HTML、GIF bytes、または保存済みファイルだけが返ります。

```python
with remote_scope():
    rdata = emout.Emout("output_dir").remote()
    rdata.phisp[:, 100, :, :].gifplot()                                  # inline HTML
    rdata.phisp[:, 100, :, :].gifplot(action="save", filename="out.gif")  # 共有 FS に保存
    gif = rdata.phisp[:, 100, :, :].gifplot(action="bytes")               # bytes を受け取る
```

詳細は [アニメーションガイド](animation.ja.md) のリモート実行節を参照してください。

## 安全設定と lifecycle 詳細

### ローカル field 読み込みを禁止する

ログインノードで大きな field 配列を誤って materialize したくない場合は、ローカル field データアクセスを無効化できます。

```python
import emout

emout.disable_local_data_access()

data = emout.Emout("output_dir")
data.phisp[-1].materialize()  # LocalDataAccessDisabledError
```

この設定は既存の `Emout` インスタンスにも効きます。`Emout(..., local_data_policy="allow")` を渡したインスタンスだけは明示的にローカル読み込みを許可できます。

```python
small = emout.Emout("small_output", local_data_policy="allow")
```

一時的に許可したい場合は context manager を使います。

```python
with emout.local_data_policy("allow"):
    small_slice = data.phisp[-1, :10, :10, :10]
```

シェル全体で強制する場合は環境変数を使えます。

```bash
export EMOUT_LOCAL_DATA_POLICY=remote_required
```

`remote_required` 中でも `data.phisp.shape` や `data.phisp.grid_shape` のようなメタデータ参照は可能です。field の実データを読む解析は `data.remote()` / `RemoteRef` で実行してください。

### `remote_scope()` の lifecycle

`remote_scope()` は worker 側 object の drop をまとめるための scope です。短い処理では `with remote_scope():` が基本です。

#### `open()` / `close()` — Jupyter 向け明示記法

Jupyter でセル全体をインデントしたくない場合は、`open()` / `close()` を直接呼べます。

```python
from emout.distributed import remote_scope

scope = remote_scope()
scope.open()

rdata = data.remote()
ref = rdata.phisp[-1, :, 100, :]
ref.plot()

# ...別のセルでも rdata / ref を使える...

scope.close()   # ここで scope 内の remote object をすべて drop
```

`close()` は何度呼んでも安全（2 回目以降は no-op）です。

#### `clear()` — scope を維持したままの手動 GC

ループで大量に中間 ref を作る場合は、`clear()` で scope を閉じずに溜まった ref だけを解放できます。

```python
scope = remote_scope()
scope.open()
rdata = data.remote()

for t in range(100):
    ref = rdata.phisp[t, :, 100, :]
    arr = ref.fetch()
    # ... 処理 ...
    scope.clear()

scope.close()
```

#### スコープのネスト

`remote_scope` はスタックとして動きます。新しく作られた ref は常に一番内側の scope に登録されるので、内側を閉じればその ref だけが drop され、外側は active のまま残ります。

```python
scope1 = remote_scope()
scope1.open()

with remote_scope():
    ref_inner = rdata.phisp[-1, :, 100, :]

ref_outer = rdata.exz[-1]
scope1.close()
```

> **落とし穴:** 同じ scope インスタンスを `open()` と `with scope:` の両方で使わないでください。`with` を抜けた時点でその scope は閉じるため、後続の ref が tracking されない原因になります。混在が必要なら、内側は新しい `remote_scope()` インスタンスを作ります。

### `remote_figure()` の派生形

#### `as fig` で FigureProxy を受け取る

`remote_figure(...)` は worker 側に作られる `Figure` の `FigureProxy` を yield するため、`as fig` でそのまま受け取って `fig.add_axes(...)` などを呼び出せます。

```python
with remote_figure(figsize=(13, 6), dpi=300) as fig:
    ax = fig.add_axes([0.13, 0.11, 0.57, 0.78], projection="3d")
    cax = fig.add_axes([0.74, 0.12, 0.025, 0.76])
    rdata.phisp[-1].plot_surfaces(ax=ax, surfaces=data.boundaries)
    ax.view_init(elev=36, azim=-110)
    plt.colorbar(cax=cax, label=r"$\phi$ (V)")
```

#### `open()` / `close()` 形式

既存コードに `with` ブロックを追加するのが難しい場合は、`RemoteFigure` の `open()` / `close()` を使えます。

```python
from emout.distributed import RemoteFigure

rf = RemoteFigure()
rf.open()
rdata.phisp[-1, :, 100, :].plot()
plt.xlabel("x [m]")
rf.close()
```

`close()` を呼び忘れると matplotlib がモンキーパッチされたままになるため、できるだけ `with remote_figure():` を使うのが安全です。

#### Jupyter セルマジック（`%%remote_figure`）

セッションで 1 回マジックを登録すれば、セル先頭に `%%remote_figure` と書くだけで使えます。

```python
%load_ext emout.distributed.remote_figure
# または: from emout.distributed import register_magics; register_magics()
```

```python
%%remote_figure --dpi 300 --fmt svg --figsize 12,6
rdata.phisp[-1, :, 100, :].plot()
plt.xlabel("x [m]")
```

| オプション | 短縮 | 説明 | デフォルト |
| --- | --- | --- | --- |
| `--dpi` | `-d` | 出力解像度 | `150` |
| `--fmt` | `-f` | 画像フォーマット（`png`, `svg`, …） | `png` |
| `--figsize` | | `幅,高さ` | matplotlib デフォルト |
| `--emout-dir` | | セッション検索用ディレクトリ | 自動 |

## 仕組み

```text
ログインノード (Jupyter)              計算ノード (SLURM worker)

emout server start              ->     Scheduler + Worker 起動
                                      <-> InfiniBand 高速通信
rdata = emout.Emout("dir").remote()
with remote_scope():
    with remote_figure():
        rdata.phisp[-1,:,100,:].plot()  ->  HDF5 読込 + 描画を worker で実行
                                        <-  PNG bytes だけ返る
```

通常のローカル `plot()` では、HDF5 から読んだスライスをローカル Python に載せてから matplotlib で描画します。`remote_figure()` では HDF5 読み込みと描画を worker 側で完結させ、ローカルには画像だけを返します。

### 共有セッション アーキテクチャ

1 つの `RemoteSession` Dask Actor が、すべての Emout インスタンスを 1 つの worker 上で管理します。異なるシミュレーションのデータにアクセスすると、セッションはその `Emout` を初回使用時に遅延ロードし、以降の呼び出しに備えてキャッシュします。

つまり、異なるシミュレーションの結果を同一の `remote_figure()` ブロック内で自由に混在できます。

```python
data_a = emout.Emout("/path/to/sim_a").remote()
data_b = emout.Emout("/path/to/sim_b").remote()

with remote_scope():
    with remote_figure(figsize=(12, 5)):
        plt.subplot(1, 2, 1)
        data_a.phisp[-1, :, 100, :].plot()
        plt.title("Sim A")

        plt.subplot(1, 2, 2)
        data_b.phisp[-1, :, 100, :].plot()
        plt.title("Sim B")
```

worker job を `scancel` したり walltime で timeout した場合も、次回の `emout server start` / 自動接続時に stale session として扱われ、saved state が自動的に掃除されます。互換モードではローカル実行に戻り、explicit remote では再起動を促すエラーになります。

## 明示的な接続

自動接続ではなく手動で接続する場合:

```python
from emout.distributed import connect

client = connect()                                         # active/default セッション
client = connect(name="batch2")                            # 追加の named session
client = connect("tls://10.10.64.2:8786", name="batch2")   # アドレス指定 + 保存済み資格情報
```

追加セッションを明示的に立ち上げたい場合は、名前を付けて起動します。

```bash
emout server start --allow-multiple --name batch2 --memory 120G
emout server status --all
emout server stop --name batch2
```

## 環境変数

| 変数 | 説明 | デフォルト |
| --- | --- | --- |
| `EMOUT_DASK_SCHED_IP` | スケジューラ IP（自動検出を上書き） | InfiniBand 自動検出 |
| `EMOUT_DASK_SCHED_PORT` | スケジューラポート | `10000 + (UID % 50000)` |
| `EMOUT_DASK_PARTITION` | SLURM パーティション | `gr20001a` |
| `EMOUT_DASK_CORES` | ワーカーコア数 | `60` |
| `EMOUT_DASK_MEMORY` | ワーカーメモリ | `60G` |
| `EMOUT_DASK_WALLTIME` | ジョブ実行時間上限 | `03:00:00` |

### ポート選択

スケジューラポートのデフォルトは `10000 + (UID % 50000)` で、同一ログインノード上の各ユーザーに自動的に異なるポートが割り当てられます（例: UID 36291 → ポート 46291）。そのポートが既に使用中の場合、最大 20 個の連続ポートを探索して空きポートを見つけます。`EMOUT_DASK_SCHED_PORT` を設定すると手動で上書きできます。

## 制限事項

- Python >= 3.10 で `dask` と `distributed` がインストールされている必要があります。
- すべてのシミュレーションディレクトリが worker ノードからアクセス可能でなければなりません（共有ファイルシステムが必要）。
- worker のメモリはロードされた Emout インスタンスごとに増加します。大規模なキャンペーンでは `drop()` や `remote_scope()` でキャッシュされた結果を解放してください。
- PyVista の 3D scene を worker に保持して対話的に操作する用途は対象外です。PyVista については [PyVista 可視化](pyvista.ja.md) を参照してください。
