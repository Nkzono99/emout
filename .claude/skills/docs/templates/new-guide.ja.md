<!--
  新しいガイドページの骨組み（canonical、日本語版）。
  `docs/source/guide/<name>.ja.md` にコピーしてから `<...>` を埋めること。
  英語ミラーは `.claude/skills/docs/templates/new-guide.md` を参照。

  既存ガイド（quickstart / plotting / animation / inp / units / boundaries
  / backtrace / distributed）の節の粒度・長さをまず眺めてから書き始めると、
  文量が揃いやすい。迷ったら backtrace.ja.md が一番構造のお手本になる。
-->

# <機能名>（`<公開 API の典型的な書き方>`）<— 実験的なら末尾に「— 実験的」>

<1〜3 文の導入。この機能で何ができるか、どういう読者を想定しているかを
明示する。コード片は挟まず、平文で。>

> **要件（あれば）:** 外部依存（例: `vdist-solver-fortran`）、optional
> extras（例: `pip install "emout[pyvista]"`）、対応 Python バージョン
> 条件など。未インストール時の挙動（例: `ImportError`）も書く。

## いつ使うか

- <ユースケース 1>
- <ユースケース 2>
- <ユースケース 3>

<もし計算コストや副作用が大きい場合、その注意書きをここに 1 段落で。
例: 「大きな max_step を渡すと時間がかかるので HPC で走らせるのが推奨」>

## クイックスタート

```python
import emout

data = emout.Emout("output_dir")

# <最短の使い方 1 — 典型的な 1 行>
<code>

# <最短の使い方 2 — 別の入口があれば>
<code>
```

## <主要な API 1>：`<method_name>`

`<method_name>(arg1, arg2, ...)` は <1 行説明>。戻り値は
:class:`<ReturnType>` で、<ひとこと補足>。

```python
<code example>
```

| 属性 / 引数 | 形状 / 型 | 意味 |
| --- | --- | --- |
| `<attr>` | `(N,)` | <説明> |
| `<attr>` | `(N, 3)` | <説明> |

### ショートハンド / 便利記法（あれば）

<`.vxvz.plot()` のような省略記法や、`__getattr__` で拾える pair 名
一覧などを説明。既存 guide の backtrace.ja.md が参考になる。>

```python
<code>
```

## <主要な API 2>：`<method_name>`

<同じ形式で繰り返す。2〜4 個を目安に。>

## リモート実行との連携（あれば）

`data.remote()` 経由でも同じ API が使えます。<どのルートが
専用プロキシで、どれが汎用 RemoteRef 経由かを明記>:

```python
from emout.distributed import remote_figure, remote_scope

rdata = emout.Emout("output_dir").remote()

with remote_scope():
    <remote usage example>
```

詳細は [リモート実行ガイド](distributed.ja.md) を参照してください。

## よくあるハマりどころ

- <ハマりどころ 1>
- <ハマりどころ 2>
- <ハマりどころ 3>

## 関連クラス

詳細なシグネチャは API リファレンス（`<emout.sub.package>` パッケージ）を
参照してください。

- `<ClassName1>` — <役割>
- `<ClassName2>` — <役割>
