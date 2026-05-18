Lang: [English](README.en.md) | [日本語](README.md)

# emout Codex Plugins

This directory contains Codex plugins that package the context needed to analyze EMSES outputs with emout.

## Available Plugins

| Plugin | Contents |
| --- | --- |
| [emout-context](emout-context/README.en.md) | Skills and bundled references for loading outputs, creating and improving visualization scripts, unit conversion, boundaries, large-scale visualization with `remote_scope` / `remote_figure`, troubleshooting, feedback, and issue reporting |

## Installation

If emout is already installed, register the Codex marketplace with:

```bash
emout codex install-plugin
```

This command runs `codex plugin marketplace add` internally. If the Codex CLI is not found, it prints Codex CLI installation instructions.

After registration, start Codex, open `/plugins`, and install `emout Context`.

```bash
codex
# Open /plugins inside Codex
```

After installing the plugin, restart Codex. The `emout-context` skills will then be available even when Codex is started outside the repository.

To sparse-install only the marketplace metadata and plugin from GitHub manually:

```bash
codex plugin marketplace add Nkzono99/emout \
  --ref main \
  --sparse .agents/plugins \
  --sparse plugins/emout-context
```

This command registers the marketplace with Codex. It does not enable the plugin yet. Next, start Codex, open `/plugins`, and install `emout Context`.

To update an already registered marketplace:

```bash
emout codex upgrade-plugin
```

To use a local checkout as the marketplace:

```bash
codex plugin marketplace add /path/to/emout
```

In this case too, install `emout Context` from `/plugins` after registering the marketplace.

## Skill Visibility

The repo-root `.claude/skills/` directory contains project-local skills for emout developers. They are loaded only when Codex is started inside the emout repository.

The `plugins/emout-context/skills/` directory contains user-facing plugin skills. After installing the plugin from `/plugins` and restarting Codex, these skills are available from other working directories such as `~` or simulation output directories.

## Placement Policy

emout-specific APIs, axis order, unit conversion, visualization, boundaries, remote execution, and analysis failure modes live in this plugin. Library development, releases, and boundary-type maintenance workflows should stay in the repo-root `.claude/skills/` directory.
