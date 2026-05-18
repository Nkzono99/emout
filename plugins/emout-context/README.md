Lang: [日本語](README.md) | [English](README.en.md)

# emout Context Plugin

emout を使う利用者が、EMSES シミュレーション出力を読み込み、解析し、可視化し、問題を切り分けるための repo-local plugin です。各 skill はユーザーの言語に合わせて応答し、コード識別子、ファイル名、変数名、コマンドは英語表記を維持します。

利用者が `pip install emout` だけで利用しており、repo 全体を読めない状況を前提にしています。そのため、この plugin は `references/` に README とユーザーガイドのスナップショットを同梱しています。repo 全体を読める開発環境では、同名の最新 docs を優先して確認してください。

## 導入

emout をインストール済みなら、次のコマンドで marketplace を登録できます:

```bash
emout codex install-plugin
```

このコマンドは Codex CLI を使って marketplace を登録します。Codex CLI が見つからない場合は、`npm install -g @openai/codex` と `codex --login` を含む導入手順を表示します。

この時点では marketplace が登録されるだけで、`emout Context` plugin はまだ有効化されていません。Codex を起動し、`/plugins` から `emout Context` を install してください。

```bash
codex
# Codex 内で /plugins を開く
```

install 後に Codex を再起動すると、repo 外の作業ディレクトリでもこの plugin の skill が利用できます。

登録済み marketplace を更新する場合:

```bash
emout codex upgrade-plugin
```

ローカル checkout を marketplace として使う場合:

```bash
codex plugin marketplace add /path/to/emout
```

この場合も、登録後に `/plugins` から `emout Context` を install します。

## Repo-local skill との違い

emout repo の `.claude/skills/` は開発者向けの project-local skill です。emout repo 配下で Codex を起動したときだけ読み込まれます。

この plugin の `skills/` は利用者向け skill です。`/plugins` で install して Codex を再起動すると、`~` やシミュレーション出力ディレクトリなど repo 外で Codex を起動しても利用できます。

## 同梱 skill

- `emout-usage-guide`: `emout.Emout` の読み込み、変数アクセス、スライス、単位変換、パラメータ参照の案内
- `emout-visualization-workflow`: 1D/2D/3D プロット、アニメーション、境界オーバーレイ、リモート描画の設計
- `emout-visualization-script`: 自然言語の可視化依頼や既存 script をもとに、`remote_scope` / `remote_figure` を含む emout 可視化 script を作成・改善
- `emout-output-diagnose`: 読み込み失敗、HDF5 / 入力ファイル不整合、単位変換、optional 依存、リモート実行の診断
- `emout-script-review`: emout analysis script の軸順序、単位変換、メモリ使用、可視化 API のレビュー
- `emout-feedback-report`: バグ、改善要望、docs 不足、解析導線の摩擦を分類し、GitHub Issue 下書きに整形
- `emout-issue-report`: バグ報告、質問、改善要望の GitHub Issue 化

各 skill の詳しい使い分けは [docs/skills-guide.md](docs/skills-guide.md) を参照してください。

## 同梱 reference

`references/` には、plugin 単体でも利用者支援ができるように以下を同梱しています。

- `README.md` / `README.en.md`
- `quickstart.ja.md` / `quickstart.md`
- `plotting.ja.md` / `plotting.md`
- `animation.ja.md` / `animation.md`
- `inp.ja.md` / `inp.md`
- `units.ja.md` / `units.md`
- `boundaries.ja.md` / `boundaries.md`
- `backtrace.ja.md` / `backtrace.md`
- `distributed.ja.md` / `distributed.md`

## 配布方針

emout 固有の使い方と解析時の注意点はこの plugin に置きます。MPIEMSES3D の入力設計や実行診断は、必要に応じて MPIEMSES3D 側の context plugin と併用してください。
