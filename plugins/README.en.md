Lang: [English](README.en.md) | [日本語](README.md)

# emout Codex Plugins

This directory contains Codex plugins that package the context needed to analyze EMSES outputs with emout.

## Available Plugins

| Plugin | Contents |
| --- | --- |
| [emout-context](emout-context/README.en.md) | Skills and bundled references for loading outputs, creating and improving visualization scripts, unit conversion, boundaries, large-scale visualization with `remote_scope` / `remote_figure`, troubleshooting, feedback, and issue reporting |

## Installation

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

## Skill Visibility

The repo-root `.claude/skills/` directory contains project-local skills for emout developers. They are loaded only when Codex is started inside the emout repository.

The `plugins/emout-context/skills/` directory contains user-facing plugin skills. After installing the plugin from `/plugins` and restarting Codex, these skills are available from other working directories such as `~` or simulation output directories.

## Placement Policy

emout-specific APIs, axis order, unit conversion, visualization, boundaries, remote execution, and analysis failure modes live in this plugin. Library development, releases, and boundary-type maintenance workflows should stay in the repo-root `.claude/skills/` directory.
