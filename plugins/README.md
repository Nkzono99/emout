Lang: [日本語](README.md) | [English](README.en.md)

# emout Codex Plugins

このディレクトリには、emout を使った EMSES 出力解析に必要な文脈を Codex に配布するための plugin を置きます。

## 利用できる plugin

| Plugin | 内容 |
| --- | --- |
| [emout-context](emout-context/README.md) | 出力読み込み、可視化 script 作成・改善、単位変換、境界、`remote_scope` / `remote_figure` を使う大規模可視化、トラブルシュート、feedback / issue 作成補助用の skill と同梱 reference |

## 導入

emout をインストール済みなら、次のコマンドで Codex marketplace を登録できます:

```bash
emout codex install-plugin
```

このコマンドは内部で `codex plugin marketplace add` を実行します。Codex CLI が見つからない場合は、Codex CLI の導入手順を表示します。

登録後は Codex を起動して `/plugins` を開き、`emout Context` を install してください。

```bash
codex
# Codex 内で /plugins を開く
```

install 後に Codex を再起動すると、repo 外の作業ディレクトリでも `emout-context` の skill が利用できるようになります。

GitHub から marketplace と plugin だけを手動で sparse install する場合:

```bash
codex plugin marketplace add Nkzono99/emout \
  --ref main \
  --sparse .agents/plugins \
  --sparse plugins/emout-context
```

このコマンドは marketplace を Codex に登録します。plugin 本体はまだ有効化されていないため、続けて Codex を起動して `/plugins` を開き、`emout Context` を install してください。

既に登録済みの marketplace を更新する場合:

```bash
emout codex upgrade-plugin
```

ローカル checkout を marketplace として使う場合:

```bash
codex plugin marketplace add /path/to/emout
```

この場合も、登録後に `/plugins` から `emout Context` を install します。

## Skill の見え方

repo root の `.claude/skills/` は emout 開発者向けの project-local skill です。emout repo 配下で Codex を起動した場合だけ読み込まれます。

一方、`plugins/emout-context/skills/` は利用者向け plugin skill です。`/plugins` で install して Codex を再起動すると、`~` やシミュレーション出力ディレクトリなど repo 外で Codex を起動しても利用できます。

## 配置方針

emout 固有の API、軸順序、単位変換、可視化、境界、リモート実行、解析時の失敗モードはこの plugin に置きます。ライブラリ開発、リリース、境界型追加などの保守作業は repo root の `.claude/skills/` に寄せます。
