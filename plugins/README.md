Lang: [日本語](README.md) | [English](README.en.md)

# emout Agent Plugins

このディレクトリには、emout を使った EMSES 出力解析に必要な文脈を Codex と Claude Code に配布するための plugin を置きます。

## 利用できる plugin

| Plugin | 内容 |
| --- | --- |
| [emout-context](emout-context/README.md) | 出力読み込み、PyVista 3D 可視化、可視化 script 作成・改善、単位変換、境界、`remote_scope` / `remote_figure` を使う大規模可視化、トラブルシュート、feedback / issue 作成補助用の skill と同梱 reference |

## Codex での導入

Codex 標準の plugin CLI で、GitHub から marketplace と plugin 本体だけを sparse install します。

```bash
codex plugin marketplace add Nkzono99/emout \
  --ref main \
  --sparse .agents/plugins \
  --sparse plugins/emout-context
codex plugin add emout-context@emout
```

`codex plugin marketplace add` は marketplace を Codex に登録し、`codex plugin add` が `emout Context` plugin を install します。install 後に Codex を再起動すると、repo 外の作業ディレクトリでも `emout-context` の skill が利用できるようになります。

Codex app で install する場合は、Codex を起動して `/plugins` から `emout Context` を install しても同じです。

```bash
codex
# Codex 内で /plugins を開く
```

既に登録済みの marketplace を更新する場合:

```bash
codex plugin marketplace upgrade emout
codex plugin add emout-context@emout
```

更新後は Codex を再起動するか、新しい thread で確認してください。

emout CLI をインストール済みなら、次のショートカットも使えます。内部では Codex CLI を呼び出します。

```bash
emout codex install-plugin
emout codex upgrade-plugin
```

ローカル checkout を marketplace として使う場合:

```bash
codex plugin marketplace add /path/to/emout
codex plugin add emout-context@emout
```

この場合も、Codex app の `/plugins` から install できます。

## Claude Code での導入

Claude Code 標準の plugin marketplace として、この repo を追加してから `emout-context` を install します。

```bash
claude plugin marketplace add Nkzono99/emout \
  --sparse .claude-plugin plugins/emout-context
claude plugin install emout-context@emout
```

Claude Code の対話 UI から導入する場合:

```text
/plugin marketplace add Nkzono99/emout
/plugin install emout-context@emout
/reload-plugins
```

登録済み marketplace を更新する場合:

```bash
claude plugin marketplace update emout
claude plugin update emout-context@emout
```

ローカル checkout を marketplace として使う場合:

```bash
claude plugin marketplace add /path/to/emout
claude plugin install emout-context@emout
```

emout CLI をインストール済みなら、次のショートカットも使えます。内部では Claude Code CLI を呼び出します。

```bash
emout claude install-plugin
emout claude upgrade-plugin
```

開発中の plugin を一時的に試すだけなら、install せずに直接読み込めます。

```bash
claude --plugin-dir ./plugins/emout-context
```

## Skill の見え方

repo root の `.claude/skills/` は emout 開発者向けの project-local skill です。emout repo 配下で agent を起動した場合だけ読み込まれます。

一方、`plugins/emout-context/skills/` は利用者向け plugin skill です。Codex の `/plugins` または Claude Code の `/plugin` で install すると、`~` やシミュレーション出力ディレクトリなど repo 外で agent を起動しても利用できます。Claude Code では `/emout-context:<skill-name>` のように plugin 名で namespace されます。

## 配置方針

emout 固有の API、軸順序、単位変換、可視化、境界、リモート実行、解析時の失敗モードはこの plugin に置きます。ライブラリ開発、リリース、境界型追加などの保守作業は repo root の `.claude/skills/` に寄せます。
