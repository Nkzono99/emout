Lang: [English](README.en.md) | [日本語](README.md)

# emout Context Plugin

This repo-local plugin helps emout users load, analyze, visualize, and troubleshoot EMSES simulation outputs. Each skill responds in the user's language while keeping code identifiers, filenames, variable names, and commands in English.

The plugin assumes users may have installed emout with `pip install emout` and may not have the full repository available. For that reason, `references/` bundles snapshots of the README and user guides. In a development checkout where the full repository is available, prefer the current root docs with the same names.

## Installation

Use the standard Codex plugin CLI to sparse-install only the marketplace metadata and plugin from GitHub.

```bash
codex plugin marketplace add Nkzono99/emout \
  --ref main \
  --sparse .agents/plugins \
  --sparse plugins/emout-context
codex plugin add emout-context@emout
```

`codex plugin marketplace add` registers the marketplace with Codex, and `codex plugin add` installs the `emout Context` plugin. After installing the plugin, restart Codex. The plugin skills will then be available even when Codex is started outside the repository.

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

## Difference From Repo-local Skills

The emout repository's `.claude/skills/` directory contains project-local skills for developers. They are loaded only when Codex is started inside the emout repository.

This plugin's `skills/` directory contains user-facing plugin skills. After installing the plugin from `/plugins` and restarting Codex, these skills are available from other working directories such as `~` or simulation output directories.

## Layout Policy

This plugin uses the repo marketplace + plugin subdirectory layout. The marketplace lives at `.agents/plugins/marketplace.json`, and the plugin source lives under `plugins/emout-context/`.

- `.codex-plugin/plugin.json`: plugin manifest. It points at bundled entry points such as `skills` with plugin-root-relative paths
- `skills/`: compact workflow definitions used for implicit selection. Do not put long explanations here
- `references/`: user-guide snapshots that remain available outside the source checkout. Put long explanations and API examples here
- `docs/`: shared support material, decision criteria, and short workflow notes used by multiple skills

Add `hooks/`, `.mcp.json`, `.app.json`, or `assets/` at the plugin root only when they are actually needed. Do not reference missing companion files from `plugin.json`.

## Bundled Skills

- `emout-usage-guide`: guidance for `emout.Emout` loading, variable access, slicing, unit conversion, and parameter inspection
- `emout-article-publication`: guidance for article record/replay, environment variables, archives, and averaged data for paper/publication bundles
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
- `article.ja.md` / `article.md`
- `inp.ja.md` / `inp.md`
- `units.ja.md` / `units.md`
- `boundaries.ja.md` / `boundaries.md`
- `backtrace.ja.md` / `backtrace.md`
- `distributed.ja.md` / `distributed.md`

`docs/` also includes `article-publication.md` / `article-publication.en.md` as shared support material for skills. Detailed usage stays in `references/article.*.md` to keep skill bodies small.

## Distribution Policy

emout-specific usage guidance and analysis pitfalls live in this plugin. For MPIEMSES3D input design or run diagnosis, use the MPIEMSES3D context plugin alongside this one when needed.
