Lang: [English](README.en.md) | [日本語](README.md)

# emout Agent Plugins

This directory contains Codex and Claude Code plugins that package the context needed to analyze EMSES outputs with emout.

## Available Plugins

| Plugin | Contents |
| --- | --- |
| [emout-context](emout-context/README.en.md) | Skills and bundled references for loading outputs, creating and improving visualization scripts, unit conversion, boundaries, large-scale visualization with `remote_scope` / `remote_figure`, troubleshooting, feedback, and issue reporting |

## Codex Installation

Use the standard Codex plugin CLI to sparse-install only the marketplace metadata and plugin from GitHub.

```bash
codex plugin marketplace add Nkzono99/emout \
  --ref main \
  --sparse .agents/plugins \
  --sparse plugins/emout-context
codex plugin add emout-context@emout
```

`codex plugin marketplace add` registers the marketplace with Codex, and `codex plugin add` installs the `emout Context` plugin. After installing the plugin, restart Codex. The `emout-context` skills will then be available even when Codex is started outside the repository.

To install from the Codex app instead, start Codex, open `/plugins`, and install `emout Context`.

```bash
codex
# Open /plugins inside Codex
```

To update an already registered marketplace:

```bash
codex plugin marketplace upgrade emout
codex plugin add emout-context@emout
```

After upgrading, restart Codex or verify in a new thread.

If emout is already installed, these shortcut commands are also available. They call the Codex CLI internally.

```bash
emout codex install-plugin
emout codex upgrade-plugin
```

To use a local checkout as the marketplace:

```bash
codex plugin marketplace add /path/to/emout
codex plugin add emout-context@emout
```

In this case too, you can install from `/plugins` in the Codex app.

## Claude Code Installation

Add this repository as a Claude Code plugin marketplace, then install `emout-context`.

```bash
claude plugin marketplace add Nkzono99/emout \
  --sparse .claude-plugin plugins/emout-context
claude plugin install emout-context@emout
```

To install from the interactive Claude Code UI:

```text
/plugin marketplace add Nkzono99/emout
/plugin install emout-context@emout
/reload-plugins
```

To update an already registered marketplace:

```bash
claude plugin marketplace update emout
claude plugin update emout-context@emout
```

To use a local checkout as the marketplace:

```bash
claude plugin marketplace add /path/to/emout
claude plugin install emout-context@emout
```

If emout is already installed, these shortcut commands are also available. They call the Claude Code CLI internally.

```bash
emout claude install-plugin
emout claude upgrade-plugin
```

For temporary development testing without installing the marketplace:

```bash
claude --plugin-dir ./plugins/emout-context
```

## Skill Visibility

The repo-root `.claude/skills/` directory contains project-local skills for emout developers. They are loaded only when an agent is started inside the emout repository.

The `plugins/emout-context/skills/` directory contains user-facing plugin skills. After installing the plugin from Codex `/plugins` or Claude Code `/plugin`, these skills are available from other working directories such as `~` or simulation output directories. In Claude Code, skills are namespaced as `/emout-context:<skill-name>`.

## Placement Policy

emout-specific APIs, axis order, unit conversion, visualization, boundaries, remote execution, and analysis failure modes live in this plugin. Library development, releases, and boundary-type maintenance workflows should stay in the repo-root `.claude/skills/` directory.
