Lang: [English](README.en.md) | [日本語](README.md)

# emout Context Plugin

This repo-local plugin helps emout users load, analyze, visualize, and troubleshoot EMSES simulation outputs. Each skill responds in the user's language while keeping code identifiers, filenames, variable names, and commands in English.

The plugin assumes users may have installed emout with `pip install emout` and may not have the full repository available. For that reason, `references/` bundles snapshots of the README and user guides. In a development checkout where the full repository is available, prefer the current root docs with the same names.

## Installation

If emout is already installed, register the marketplace with:

```bash
emout codex install-plugin
```

This command uses the Codex CLI to register the marketplace. If the Codex CLI is not found, it prints setup instructions including `npm install -g @openai/codex` and `codex --login`.

At this point only the marketplace is registered; the `emout Context` plugin is not enabled yet. Start Codex, open `/plugins`, and install `emout Context`.

```bash
codex
# Open /plugins inside Codex
```

After installing the plugin, restart Codex. The plugin skills will then be available even when Codex is started outside the repository.

To update an already registered marketplace:

```bash
emout codex upgrade-plugin
```

To use a local checkout as the marketplace:

```bash
codex plugin marketplace add /path/to/emout
```

In this case too, install `emout Context` from `/plugins` after registering the marketplace.

## Difference From Repo-local Skills

The emout repository's `.claude/skills/` directory contains project-local skills for developers. They are loaded only when Codex is started inside the emout repository.

This plugin's `skills/` directory contains user-facing plugin skills. After installing the plugin from `/plugins` and restarting Codex, these skills are available from other working directories such as `~` or simulation output directories.

## Bundled Skills

- `emout-usage-guide`: guidance for `emout.Emout` loading, variable access, slicing, unit conversion, and parameter inspection
- `emout-visualization-workflow`: plot design for 1D/2D/3D views, animations, boundary overlays, and remote rendering
- `emout-visualization-script`: create or improve emout visualization scripts from natural-language plotting requests or existing scripts, including `remote_scope` / `remote_figure`
- `emout-output-diagnose`: diagnosis for loading failures, HDF5/input mismatches, unit conversion, optional dependencies, and remote execution
- `emout-script-review`: review of emout analysis scripts for axis order, unit conversion, memory use, and visualization APIs
- `emout-feedback-report`: classify bugs, improvement requests, documentation gaps, and analysis workflow friction into GitHub Issue drafts
- `emout-issue-report`: GitHub issue drafting for bugs, questions, and improvement requests

See [docs/skills-guide.en.md](docs/skills-guide.en.md) for detailed skill selection.

## Bundled References

`references/` includes the following files so the plugin can help users on its own:

- `README.md` / `README.en.md`
- `quickstart.ja.md` / `quickstart.md`
- `plotting.ja.md` / `plotting.md`
- `animation.ja.md` / `animation.md`
- `inp.ja.md` / `inp.md`
- `units.ja.md` / `units.md`
- `boundaries.ja.md` / `boundaries.md`
- `backtrace.ja.md` / `backtrace.md`
- `distributed.ja.md` / `distributed.md`

## Distribution Policy

emout-specific usage guidance and analysis pitfalls live in this plugin. For MPIEMSES3D input design or run diagnosis, use the MPIEMSES3D context plugin alongside this one when needed.
