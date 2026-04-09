Lang: [日本語](distributed.ja.md) | [English](distributed.md)

# リモート実行 (Dask) — 実験的

大規模なシミュレーション出力を HPC の計算ノードで処理し、
ログインノード（手元の Jupyter）にはプロット画像だけを返すリモート実行基盤です。

## 仕組み

```
ログインノード (Jupyter)              計算ノード (SLURM worker)

emout server start              →     Scheduler + Worker 起動
                                      ↕ InfiniBand 高速通信
data = emout.Emout("dir")
data.phisp[-1,:,100,:].plot()   →     HDF5 読込 → 2D スライス → 転送
                                ←     小さい配列 (数 KB)
plt.xlabel("custom")                  ← ローカル matplotlib で描画

with remote_figure():
    data.phisp[-1,:,100,:].plot()  →  全操作をサーバーで実行
    plt.xlabel("custom")           →  (コマンドとして記録)
                                   ←  PNG bytes (~50 KB) だけ返る
```

### 共有セッション アーキテクチャ

1 つの `RemoteSession` Dask Actor が、すべての Emout インスタンスを 1 つの
worker 上で管理します。異なるシミュレーションのデータにアクセスすると、
セッションはその `Emout` を初回使用時に遅延ロードし、以降の呼び出しに備えて
キャッシュします。

つまり、**異なるシミュレーションの結果を同一の `remote_figure()` ブロック内で
自由に混在**できます:

```python
data_a = emout.Emout("/path/to/sim_a")
data_b = emout.Emout("/path/to/sim_b")

result_a = data_a.backtrace.get_probabilities(...)
result_b = data_b.backtrace.get_probabilities(...)

with remote_figure(figsize=(12, 5)):
    plt.subplot(1, 2, 1)
    data_a.phisp[-1, :, 100, :].plot()
    plt.title("Sim A: potential")

    plt.subplot(1, 2, 2)
    result_b.vxvz.plot(cmap="plasma")
    plt.title("Sim B: backtrace")
```

すべてのコマンドは同一の worker 上で再生されます。クライアントにデータは転送されません。

## セットアップ

### 1. サーバー起動（ターミナルで 1 回だけ）

```bash
emout server start --partition gr20001a --memory 60G --walltime 03:00:00
```

InfiniBand の IP が自動検出され、`~/.emout/server.json` に保存されます。

```
Scheduler running at tcp://10.10.64.2:8786
Detected IP: 10.10.64.2
Workers: 1
```

### 2. スクリプトから使う

`server.json` が存在すれば自動接続されるため、**コードの変更は不要**です:

```python
import emout

data = emout.Emout("output_dir")

# ↓ サーバーが起動していれば自動的にリモート実行
#   起動していなければ従来どおりローカル実行
data.phisp[-1, :, 100, :].plot()
```

### 3. サーバー停止

```bash
emout server stop
```

## 使い方

### データ転送モード（デフォルト）

`with remote_figure()` を使わない場合のデフォルト。
Worker がスライスを切り出してローカルに転送し、matplotlib はローカルで実行されます。
**`plt.axhline()` 等のカスタマイズが自由にできます。**

```python
data.phisp[-1, :, 100, :].plot()
plt.axhline(y=50, color="red")    # ← ローカル matplotlib
plt.xlabel("x [m]")
plt.title("カスタムタイトル")
plt.savefig("output.png")
```

転送されるのは 2D スライス（数 KB〜数 MB）だけで、フル 3D 配列はローカルに載りません。

### 画像モード（`remote_figure`）

**全ての matplotlib 操作をサーバー側で実行**し、ローカルには PNG 画像だけが返ります。
メモリ消費を最小限にしたい場合に使います。

```python
from emout.distributed import remote_figure

with remote_figure():
    data.phisp[-1, :, 100, :].plot()
    plt.axhline(y=50, color="red")    # ← サーバー側で実行
    plt.xlabel("x [m]")
    plt.title("カスタムタイトル")
# ← ここで PNG が Jupyter に表示される
```

#### `open()` / `close()` 形式

既存コードに `with` ブロックを追加するのが面倒な場合、`RemoteFigure` の
`open()` / `close()` を使えます:

```python
from emout.distributed import RemoteFigure

rf = RemoteFigure()
rf.open()
data.phisp[-1, :, 100, :].plot()
plt.xlabel("x [m]")
rf.close()   # ← コマンドがサーバーで再生され、PNG が表示される
```

`RemoteFigure` は `with` 文でもそのまま使えます（`with RemoteFigure() as rf: ...`）。

> **注意:** `close()` を呼び忘れると matplotlib がモンキーパッチされたままになり、
> ガベージコレクション時に `ResourceWarning` が発生します。

#### Jupyter セルマジック（`%%remote_figure`）

セッションで 1 回マジックを登録すれば、セル先頭に `%%remote_figure` と書くだけで使えます:

```python
# 登録（1 回だけ）
%load_ext emout.distributed.remote_figure
# または: from emout.distributed import register_magics; register_magics()
```

```python
%%remote_figure
data.phisp[-1, :, 100, :].plot()
plt.xlabel("x [m]")
```

マジック行にオプションを渡すこともできます:

```python
%%remote_figure --dpi 300 --fmt svg --figsize 12,6
data.phisp[-1, :, 100, :].plot()
```

| オプション | 短縮 | 説明 | デフォルト |
| --- | --- | --- | --- |
| `--dpi` | `-d` | 出力解像度 | `150` |
| `--fmt` | `-f` | 画像フォーマット（`png`, `svg`, …） | `png` |
| `--figsize` | | `幅,高さ` | matplotlib デフォルト |
| `--emout-dir` | | セッション検索用ディレクトリ | 自動 |

### backtrace 連携

重い計算はサーバーで 1 回だけ実行し、結果をサーバーメモリに保持。
可視化パラメータを変えて何度でも再レンダリングできます。

```python
# 計算（サーバーで実行、結果はサーバーメモリに保持）
result = data.backtrace.get_probabilities(
    x, y, z, vx_range, vy_center, vz_range, ispec=0,
)

# 可視化を繰り返す（再計算なし）
with remote_figure():
    result.vxvz.plot(cmap="viridis")
    plt.title("速度分布 (vx-vz)")

with remote_figure():
    result.vxvy.plot(cmap="plasma")

with remote_figure():
    result.plot_energy_spectrum(scale="log")
    plt.xlabel("Energy [eV]")

# 不要になったらサーバーメモリを解放
result.drop()
```

#### fetch() によるローカル加工

matplotlib で自由にカスタマイズしたい場合（独自アノテーション、共有カラーバーなど）、
`fetch()` で小さな結果配列をクライアントに転送できます:

```python
heatmap = result.vxvz.fetch()   # → ローカルの HeatmapData
fig, ax = plt.subplots()
heatmap.plot(ax=ax, cmap="plasma")
ax.axhline(y=0, color="red", linestyle="--")
ax.set_title("カスタムアノテーション")
```

### 境界メッシュ

```python
# 境界形状だけ表示（軽量、常にローカル）
data.boundaries.plot()

# フィールド上にオーバーレイ（3D 配列はサーバーからスライス転送）
data.phisp[-1].plot_surfaces(ax=ax, surfaces=data.boundaries)
ax.set_xlabel("x [m]")
```

## 明示的な接続

自動接続ではなく手動で接続する場合:

```python
from emout.distributed import connect
client = connect()                    # ~/.emout/server.json から自動検出
client = connect("tcp://10.10.64.2:8786")  # アドレス指定
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

スケジューラポートのデフォルトは `10000 + (UID % 50000)` で、同一ログインノード上の
各ユーザーに自動的に異なるポートが割り当てられます（例: UID 36291 → ポート 46291）。
そのポートが既に使用中の場合、最大 20 個の連続ポートを探索して空きポートを見つけます。
`EMOUT_DASK_SCHED_PORT` を設定すると手動で上書きできます。

## 制限事項

- Python >= 3.10 で `dask` と `distributed` がインストールされている必要があります。
- すべてのシミュレーションディレクトリが worker ノードからアクセス可能でなければ
  なりません（共有ファイルシステムが必要）。
- worker のメモリはロードされた Emout インスタンスごとに増加します。
  大規模なキャンペーンでは `result.drop()` でキャッシュされた計算結果を解放してください。
