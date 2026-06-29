Lang: [日本語](analysis-pitfalls.md) | [English](analysis-pitfalls.en.md)

# emout Analysis Pitfalls

emout 利用者の質問では、同じ種類の誤りが繰り返し現れます。診断時は、traceback だけでなく出力ディレクトリ、入力ファイル、データ規模、実行環境を確認します。

## よくある原因

| 症状 | 確認点 | 対処 |
| --- | --- | --- |
| 想定と違う断面が出る | 軸順序を `(t, z, y, x)` としてスライスしているか | xz 平面なら `data.phisp[-1, :, y_index, :]` の形に直す |
| メモリ不足、処理が遅い | 4D 全体を `.val` / `.val_si` で読んでいないか | 先に時刻・平面・範囲でスライスする |
| `val_si` や単位表示が不自然 | `!!key` または `[meta.unit_conversion]` があるか | 入力ファイルの単位変換情報を確認し、ない場合は EMSES 単位として扱う |
| 変数属性が見つからない | HDF5 ファイル名と EMSES 変数名が一致しているか | `ls *.h5` や `data.__dict__` ではなく guide の変数解決ルールを確認する |
| ベクトルプロットが期待と違う | `j1xy` / `j1xyz` の成分とスライス面が合っているか | 表示したい平面に合うベクトル属性を選ぶ |
| 3D plot が import error になる | emout / PyVista の依存関係が更新済みか | emout 2.20.0+ へ更新し、editable install なら再インストールする |
| 境界が表示されない | `data.boundaries` が空でないか、finbound が対応形状か | 入力ファイルの境界設定と `boundaries` guide を確認する |
| remote plot が動かない | Python 3.10+、Dask server、TLS 設定、session 名 | `emout server status` と `distributed` guide を確認する |
| 継続出力がつながらない | appended output を明示しているか | `emout.Emout("output_dir", ad="auto")` を試す |

## 追加で集める情報

- `python -m pip show emout` または `python -c "import importlib.metadata as m; print(m.version('emout'))"` の結果
- インストール方法
- `ls` などで要約した出力ファイル一覧
- `plasma.inp` / `plasma.toml` の単位変換と格子サイズに関係する部分
- 最小 Python script と完全な traceback
- HPC / remote execution を使う場合は `emout server status` の要約

## issue 化前の注意

個人パス、ホスト名、ジョブ ID、アクセストークン、未公開データ名はそのまま貼らないでください。再現に必要な範囲だけを残し、出力データそのものが必要な場合は最小化したサンプルを作ります。
