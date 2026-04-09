Lang: [日本語](animation.ja.md) | [English](animation.md)

# アニメーション (`gifplot`)

`gifplot()` は多次元データから時系列アニメーションを作成します。`plot()` に次いで 2 番目によく使う機能です。

## 基本的な使い方

```python
import emout

data = emout.Emout("output_dir")

# Jupyter Notebook でインライン表示（デフォルト: action='to_html'）
data.phisp[:, 100, :, :].gifplot()
```

スライス `[:, 100, :, :]` は z=100 での全タイムステップを選択し、xy 平面の時間変化アニメーションを生成します。

## 出力アクション

`action` パラメータでフレーム生成後の動作を制御します:

| アクション | 説明 |
| --- | --- |
| `'to_html'` | Jupyter インライン表示用の HTML を返す（デフォルト） |
| `'show'` | matplotlib ウィンドウで表示 |
| `'save'` | ファイルに保存（`filename` が必要） |
| `'return'` | `(fig, animation)` を返す（手動制御用） |
| `'frames'` | マルチパネルレイアウト用の `FrameUpdater` を返す |

```python
# GIF として保存
data.phisp[:, 100, :, :].gifplot(action="save", filename="phisp.gif")

# matplotlib ウィンドウで表示
data.phisp[:, 100, :, :].gifplot(action="show")
```

## 主なオプション

| パラメータ | 型 | 説明 | デフォルト |
| --- | --- | --- | --- |
| `action` | `str` | 出力モード（上表参照） | `'to_html'` |
| `filename` | `str` | `action='save'` 時の保存先 | `None` |
| `axis` | `int` | アニメーション軸 | `0` |
| `interval` | `int` | フレーム間隔 [ミリ秒] | `200` |
| `repeat` | `bool` | ループ再生 | `True` |
| `use_si` | `bool` | SI 単位ラベルを使用 | `True` |
| `vmin` | `float` | カラーバーの最小値 | 自動 |
| `vmax` | `float` | カラーバーの最大値 | 自動 |
| `norm` | `str` | `'log'` で対数スケール | `None` |
| `mode` | `str` | プロットモード（`'cmap'`, `'cont'`, `'stream'`） | 自動 |

## 複数パネルアニメーション

複数データソースをグリッドレイアウトで 1 つのアニメーションにまとめます:

```python
# ステップ1: フレームアップデータを作成
updater0 = data.phisp[:, 100, :, :].gifplot(action="frames", mode="cmap")
updater1 = data.phisp[:, 100, :, :].build_frame_updater(mode="cont")
updater2 = data.nd1p[:, 100, :, :].build_frame_updater(
    mode="cmap", vmin=1e-3, vmax=20, norm="log"
)
updater3 = data.nd2p[:, 100, :, :].build_frame_updater(
    mode="cmap", vmin=1e-3, vmax=20, norm="log"
)
updater4 = data.j2xy[:, 100, :, :].build_frame_updater(mode="stream")

# ステップ2: レイアウトを3重リストで定義 [行][列][重ね合わせ]
layout = [
    [
        [updater0, updater1],   # 行0, 列0: カラーマップ + 等高線重ね合わせ
        [updater2],             # 行0, 列1: 密度（対数スケール）
        [updater3, updater4],   # 行0, 列2: 密度 + ストリームライン重ね合わせ
    ]
]

# ステップ3: Animator を作成して表示
animator = updater0.to_animator(layout=layout)
animator.plot(action="to_html")
```

### レイアウト構造

レイアウトは **3 重のネストされたリスト** です:

- **レベル 1（外側）:** 行
- **レベル 2:** 行内の列
- **レベル 3（内側）:** 同じサブプロットに重ね描きする updater

最内リスト内の各 updater は同じ Axes に描画されるため、異なる可視化モードの重ね合わせ（例: カラーマップ + 等高線、密度 + ストリームライン）が可能です。

### `build_frame_updater` と `gifplot(action='frames')` の違い

どちらも `FrameUpdater` オブジェクトを生成します。違いは:

- `gifplot(action='frames')` は簡易ショートカット
- `build_frame_updater()` は `mode`, `vmin`, `vmax` 等を明示的に制御可能

パネルごとにカスタマイズが必要な場合は `build_frame_updater()` を使います。
