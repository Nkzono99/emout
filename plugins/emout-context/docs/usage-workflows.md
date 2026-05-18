Lang: [日本語](usage-workflows.md) | [English](usage-workflows.en.md)

# emout Usage Workflows

この文書は、plugin skill が利用者に提示する標準的な作業導線をまとめます。実際の回答では、ユーザーのデータパス、物理量、環境に合わせて短く調整します。

## 出力ディレクトリを読み込む

```python
import emout

data = emout.Emout("output_dir")
```

次に、目的の物理量を EMSES のファイル名由来の属性で参照します。

```python
data.phisp
data.nd1p
data.j1xy
```

グリッドデータのスライス軸順序は `(t, z, y, x)` です。断面を作るときは、目的の平面と固定する軸を明示します。

## 断面を可視化する

```python
ymid = data.inp.ny // 2
data.phisp[-1, :, ymid, :].plot()
```

大きな出力では、`val_si` で全体を読む前に `[-1, :, ymid, :]` のようにスライスします。密度や粒子分布など桁が広い量では `norm="log"` を検討します。

## 単位を確認する

```python
data.phisp[-1, :, ymid, :].val_si
data.unit.v.reverse(1.0)
```

SI 変換が期待通りでない場合は、入力ファイルの `!!key dx=...,to_c=...` または `plasma.toml` の `[meta.unit_conversion]` を確認します。

## 境界を重ねる

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
data.phisp[-1, :, ymid, :].plot(ax=ax)
data.phisp[-1, :, ymid, :].plot_surfaces(ax=ax, surfaces=data.boundaries)
```

境界が空の場合は、入力ファイル側で finbound が設定されているか、emout が対応している形状かを確認します。

## リモート実行を使う

```bash
emout server start --partition gr20001a --memory 60G
```

```python
from emout.distributed import remote_figure, remote_scope

data = emout.Emout("output_dir").remote()

with remote_scope():
    ymid = int(data.inp.ny // 2)
    with remote_figure(savefilepath="figures/phisp.png"):
        data.phisp[-1, :, ymid, :].plot()
```

リモート実行は Python 3.10+ の環境で使います。サーバー状態は `emout server status` で確認します。

`RemoteSession` は worker 側で `Emout` インスタンスや中間結果を共有する内部 Actor です。script では直接作らず、`Emout.remote()`、`remote_scope()`、`remote_figure()`、既存コードを包む `RemoteFigure` を使います。

## 可視化 script を作る

自然言語の依頼から script を作る場合は、目的の物理量、断面、時刻、保存先を先に決めます。大規模出力では、script の中でサーバー起動までは行わず、実行前の手順として `emout server start ...` を案内します。

```python
import argparse

import emout
from emout.distributed import remote_figure, remote_scope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--output", default="phisp.png")
    args = parser.parse_args()

    rdata = emout.Emout(args.output_dir).remote()
    with remote_scope():
        ymid = int(rdata.inp.ny // 2)
        with remote_figure(savefilepath=args.output):
            rdata.phisp[-1, :, ymid, :].plot()


if __name__ == "__main__":
    main()
```

## 問題を切り分ける

1. 最小 script と traceback を用意する。
2. `python -m pip show emout`、インストール方法、Python バージョンを確認する。
3. 出力ディレクトリのファイル一覧を要約する。
4. 入力ファイルの格子サイズ、単位変換、境界設定を確認する。
5. 軸順序 `(t, z, y, x)` と full-array load の有無を確認する。
