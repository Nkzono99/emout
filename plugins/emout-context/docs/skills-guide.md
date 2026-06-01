Lang: [日本語](skills-guide.md) | [English](skills-guide.en.md)

# emout Context Skill Guide

この plugin の skill は、emout を pip 経由で導入した利用者でも使えるように、`references/` の同梱ドキュメントを一次参照として使います。repo 全体が見える開発環境では、同名の root docs がより新しい可能性があるため、必要に応じて root docs も確認します。

## 共通方針

- ユーザーの言語に合わせて応答する。コード識別子、ファイル名、コマンド、EMSES 変数名は英語表記を維持する。
- emout の公開入口は `emout.Emout` とし、グリッドデータのスライス軸順序は常に `(t, z, y, x)` と明示する。
- 単位変換は `plasma.inp` の `!!key dx=...,to_c=...` または `plasma.toml` の `[meta.unit_conversion]` がある場合に使える。根拠がない場合は SI 変換できると断定しない。
- 大きな HDF5 データでは、全 4D 配列を読み込む前に時刻・平面・範囲でスライスする例を優先する。
- 入力ファイル、ログ、出力パスには個人パス、ホスト名、ジョブ ID、秘密情報が含まれることがあるため、外部 issue 化の前に要約・マスクする。

## Skill 一覧

| Skill | 使う場面 | 主な入力 | 主な出力 | 主な reference |
| --- | --- | --- | --- | --- |
| `emout-usage-guide` | emout の基本的な使い方、変数アクセス、単位変換、パラメータ参照を案内する | 出力ディレクトリ、目的の物理量、入力ファイル形式 | 最小コード例、軸順序、単位変換、次に読む guide | `README.md`, `quickstart.ja.md`, `inp.ja.md`, `units.ja.md` |
| `emout-article-publication` | 論文・公開データ用の article record/replay、環境変数、archive、平均データ保存を案内する | 可視化 script、公開データ要件、records path、複数 simulation、平均範囲 | record/replay 実行方法、環境変数、保存粒度、注意点 | `article-publication.md`, `usage-workflows.md` |
| `emout-visualization-workflow` | 1D/2D/3D プロット、アニメーション、境界オーバーレイを設計する | 物理量、スライス条件、表示形式、出力先 | プロット手順、Python 例、依存関係、保存方法 | `plotting.ja.md`, `animation.ja.md`, `boundaries.ja.md`, `distributed.ja.md` |
| `emout-visualization-script` | 自然言語の依頼や既存 script から可視化 script を作成・改善する | 目的、出力ディレクトリ、物理量、既存 script、HPC 制約 | runnable script、remote 実行版、実行手順、前提条件 | `quickstart.ja.md`, `plotting.ja.md`, `animation.ja.md`, `distributed.ja.md` |
| `emout-output-diagnose` | 読み込み失敗、plot エラー、単位変換、remote execution の問題を切り分ける | traceback、出力一覧、入力ファイル、実行環境 | 原因候補、確認コマンド、最小対処、追加で必要な情報 | `quickstart.ja.md`, `inp.ja.md`, `units.ja.md`, `analysis-pitfalls.md` |
| `emout-script-review` | emout を使う analysis script をレビューする | Python script、目的、データ規模、出力例 | 修正必須、改善推奨、リスク、修正版 snippet | `library-context.md`, `analysis-pitfalls.md`, `plotting.ja.md`, `distributed.ja.md` |
| `emout-feedback-report` | バグ、改善要望、docs 不足、解析導線の摩擦を GitHub Issue 下書きにする | 現象、期待、影響、script、traceback、環境、privacy 制約 | 分類、Issue 下書き、不足情報、ラベル候補 | `analysis-pitfalls.md`, `library-context.md`, `README.md` |
| `emout-issue-report` | バグ報告、質問、改善要望を GitHub Issue にまとめる | 現象、期待動作、再現手順、環境、traceback | Issue タイトル、本文、ラベル候補、投稿前チェック | `analysis-pitfalls.md`, `README.md`, `quickstart.ja.md` |

## 代表的な依頼

```text
emout で output_dir の phisp を xz 平面で描画したい。
論文公開用に data.phisp[-20:].mean().plot_surfaces(..., bounds=...) の record/replay 方法を教えて。
output_dir の phisp と nd1p を 2 パネルで保存する script を作って。大きいので remote_figure を使って。
この traceback から emout の読み込み失敗を診断して。
この analysis.py の軸順序と SI 変換が正しいかレビューして。
emout の remote_figure が使いづらかった点を maintainer に送る feedback にして。
emout の issue に投げる本文を作って。
```

MPIEMSES3D の入力パラメータそのものや実行失敗を調べる場合は、MPIEMSES3D context plugin の skill を併用します。emout 側では、生成済み出力をどう読むか、どう可視化するか、解析 script が妥当かに集中します。`RemoteSession` は内部の共有 Dask Actor なので、利用者向け script では通常 `Emout.remote()`、`remote_scope()`、`remote_figure()`、`RemoteFigure` を使います。
