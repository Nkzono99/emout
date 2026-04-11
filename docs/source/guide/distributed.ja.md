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

Python 3.10 以上であれば `pip install emout` で Dask と
`emout server` 用の TLS 依存関係が自動的にインストールされます。
追加のセットアップは不要です。

### 1. サーバー起動（ターミナルで 1 回だけ）

```bash
emout server start --partition gr20001a --memory 60G --walltime 03:00:00
```

InfiniBand の IP が自動検出されます。`emout` は TLS 証明書も自動生成し、
user-only 権限で保存したうえで、active なセッションを
`~/.emout/server.json` にミラーします。

```
Session: default
Scheduler running at tls://10.10.64.2:8786
Detected IP: 10.10.64.2
Workers: 1
```

デフォルトでは 1 ユーザーにつき active なサーバーセッションは 1 つだけです。
追加セッションを明示的に立ち上げたい場合は、名前を付けて起動します:

```bash
emout server start --allow-multiple --name batch2 --memory 120G
emout server status --all
emout server stop --name batch2
```

### 2. スクリプトから使う

active セッションが保存されていれば、既存コードはそのまま互換モードで動作します。
互換モードが追従するのは active/default セッションです。追加の named session を
使う場合は explicit に接続してください。新規コードでは `Emout.remote()` を使う
書き方を推奨します:

```python
import emout
from emout.distributed import remote_figure, remote_scope

data = emout.Emout("output_dir").remote()

with remote_scope():
    ymid = int(data.inp.ny // 2)
    with remote_figure():
        data.phisp[-1, :, ymid, :].plot()
```

### 3. サーバー停止

```bash
emout server stop
```

追加の named session は `emout server stop --name <session>` で停止できます。
すべて止めたい場合は `emout server stop --all` を使います。

worker job を `scancel` したり walltime で timeout した場合も、次回の
`emout server start` / 自動接続時に「worker が失われた stale session」として
扱われ、saved state が自動的に掃除されます。remote 実行は無限待ちせず、
互換モードではローカル実行に戻り、explicit remote では再起動を促すエラーになります。

## 使い方

### 推奨モード（`Emout.remote()`）

worker 上の object を `RemoteRef` として保持しながら、ローカルの `emout` / `numpy`
に近い記法で処理を書けます。`-ref`, `ref1 + ref2`, `np.abs(ref)`, `int(ref)` のような
式も remote のまま評価されます。

```python
import matplotlib.pyplot as plt
import emout
from emout.distributed import remote_figure, remote_scope

rdata = emout.Emout("output_dir").remote()

with remote_scope():
    ymid = int(rdata.inp.ny // 2)

    with remote_figure():
        plt.figure(figsize=(18, 16))
        rdata.phisp[-1, 180:400, ymid, :].plot()
        (-rdata.exz[-1, 180:400, ymid, :]).plot()
        plt.title("remote expression example")
```

`remote_scope()` の内側で作られた remote object は、`with` を抜けると自動で `drop()`
されます。ブロック内では中間結果を何度でも再利用しつつ、ワーカー側のメモリ管理は任せられます。

#### `open()` / `close()` — Jupyter 向け明示記法

`with` でセル全体をインデントしたくない場合は、`open()` / `close()` を直接呼べます。
セルを跨いで scope を持ち続けることもできます:

```python
from emout.distributed import remote_scope

scope = remote_scope()
scope.open()

rdata = data.remote()
ref = rdata.phisp[-1, :, 100, :]
ref.plot()

# ...別のセルや別の処理でそのまま rdata / ref を使える...

scope.close()   # ここで溜まった remote object をすべて drop
```

`close()` は何度呼んでも安全（2 回目以降は no-op）なので、途中で例外が出る
セルの末尾で `try/finally` を組まなくてもよいケースが多いです。

#### `clear()` — scope を維持したままの手動 GC

ループで大量に中間 ref を作るような場合、`clear()` で **scope を閉じずに**
溜まった ref だけを解放できます:

```python
scope = remote_scope()
scope.open()
rdata = data.remote()

for t in range(100):
    ref = rdata.phisp[t, :, 100, :]
    arr = ref.fetch()
    # ... 処理 ...
    scope.clear()   # このイテレーションの ref を drop、scope は継続

scope.close()
```

`scope.clear()` のあとも scope は開いたままなので、**後続で作る ref は
引き続き同じ scope に登録**されます。長時間セッションでワーカー側
メモリが単調増加するのを防ぐのに便利です。

#### スコープのネスト

`remote_scope` はスタックとして動きます。外側の scope を開いたまま、
内側にもう 1 つ scope を作ることができます。新しく作られた ref は
**常に一番内側の scope に登録**されるので、内側を閉じればそれだけが
drop され、外側はそのまま active のまま残ります:

```python
# open/open/close/close パターン
scope1 = remote_scope()
scope1.open()

scope2 = remote_scope()
scope2.open()

ref_inner = rdata.phisp[-1, :, 100, :]   # scope2 に登録
scope2.close()                             # ref_inner だけ drop

ref_outer = rdata.exz[-1]                  # scope1 に登録
scope1.close()                             # ref_outer を drop
```

`with` 文と明示 `open()` を混ぜても問題なくネストできます:

```python
scope1 = remote_scope()
scope1.open()

with remote_scope() as scope2:
    ref_inner = rdata.phisp[-1, :, 100, :]   # scope2 に登録
# with を抜けた時点で scope2 が自動 drop、scope1 は継続

ref_outer = rdata.exz[-1]                     # scope1 に登録
scope1.close()
```

> **落とし穴: 同じ scope インスタンスを `open()` と `with` の両方で使わない。**
> 下のコードは一見正しく見えますが、`with scope:` が中で `__exit__` を呼び出すため、
> `with` を抜けた時点で `scope` は **閉じています**。そのあとに作った ref はどの
> scope にも tracked されず、`scope.close()` も no-op になります:
>
> ```python
> scope = remote_scope()
> scope.open()
> with scope:                    # ← 同じ scope を with に渡してはいけない
>     ref = rdata.phisp[-1]
> # ここで scope はすでに閉じている
> leaked = rdata.phisp[-2]      # ← どの scope にも登録されない！
> scope.close()                  # ← no-op
> ```
>
> 混在が必要なら、**内側は新しい `remote_scope()` インスタンスを作る** のが
> 正解です（上の `with remote_scope() as scope2:` の例）。

### データ転送モード（互換モード）

既存の `plot()` コードをそのまま活かしたい場合の互換モードです。
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

重い粒子バックトレースはサーバーで 1 回だけ実行し、結果をワーカーのメモリに保持します。
可視化パラメータを変えながら何度でも再描画でき、再計算は走りません。

```python
# 計算（サーバーで実行、結果はワーカーにキャッシュ）
result = data.backtrace.get_probabilities(
    x, y, z, vx_range, vy_center, vz_range, ispec=0,
)

# 同じ result を使い回して何度でも再描画
with remote_figure():
    result.vxvz.plot(cmap="viridis")
    plt.title("速度分布 (vx-vz)")

with remote_figure():
    result.plot_energy_spectrum(scale="log")
    plt.xlabel("Energy [eV]")

# 不要になったらワーカーメモリを解放
result.drop()
```

`data.backtrace...` と `data.remote().backtrace...` のどちらを使っても、返り値は
同じ専用 proxy (`RemoteProbabilityResult` / `RemoteBacktraceResult`) です。
既存コードの流れを活かすなら前者、field や境界と同じ explicit-remote のスタイルに
揃えたいなら後者を選びます:

```python
with remote_scope():
    rdata = data.remote()

    bt = rdata.backtrace.get_backtrace(position, velocity, ispec=0)
    result = rdata.backtrace.get_probabilities(
        x, y, z, vx_range, vy_center, vz_range, ispec=0,
    )

    with remote_figure():
        bt.tx.plot()
        result.vxvz.plot(cmap="viridis")
```

バックトレース API そのもの（`BacktraceResult` / `MultiBacktraceResult` /
`ProbabilityResult` のショートハンドや軸一覧）は、専用の
[バックトレースガイド](backtrace.ja.md) を参照してください。

#### fetch() によるローカル加工

matplotlib で自由にカスタマイズしたい場合（独自アノテーション、共有カラーバーなど）、
`fetch()` で結果を小さなローカル配列として取り出せます:

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

### アニメーション (`gifplot`)

`gifplot()` もワーカー側で完結します。フレーム生成とエンコードは全部ワーカー上で走り、
クライアントにはインライン HTML か GIF バイト列だけが返ります:

```python
rdata = emout.Emout("output_dir").remote()

with remote_scope():
    rdata.phisp[:, 100, :, :].gifplot()                                 # inline HTML
    rdata.phisp[:, 100, :, :].gifplot(action="save", filename="out.gif")  # 共有 FS に保存
    gif = rdata.phisp[:, 100, :, :].gifplot(action="bytes")             # bytes を受け取る
```

詳細は [アニメーションガイド](animation.ja.md) の「リモート実行」節を参照。

## 明示的な接続

自動接続ではなく手動で接続する場合:

```python
from emout.distributed import connect
client = connect()                                         # active/default セッション
client = connect(name="batch2")                            # 追加の named session
client = connect("tls://10.10.64.2:8786", name="batch2")   # アドレス指定 + 保存済み資格情報
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
