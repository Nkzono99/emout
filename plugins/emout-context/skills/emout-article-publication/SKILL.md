---
name: emout-article-publication
description: Guide emout article record/replay workflows for paper publication data, including environment variables, records path, article_name/source_name, archives, time-mean data, and plot_surfaces(bounds) ROI recording.
---

Use this skill when the user asks how to publish minimal emout visualization data, use article record/replay mode, reproduce paper figures, create public data bundles, configure `EMOUT_ARTICLE_*` environment variables, archive article records, or record averaged data such as `data.phisp[-20:].mean().plot_surfaces(..., bounds=...)`.

## Response Language

- Respond in the user's language.
- Keep code identifiers, filenames, environment variables, commands, and EMSES variable names unchanged.
- If language is unclear, default to Japanese for this repository.

## Context Sources

- Primary bundled references: `../../references/article.ja.md`, `../../references/article.md`.
- Primary bundled docs: `../../docs/article-publication.md`, `../../docs/article-publication.en.md`.
- Related bundled docs: `../../docs/usage-workflows.md`, `../../docs/usage-workflows.en.md`, `../../docs/library-context.md`, `../../docs/library-context.en.md`.
- Related references only when needed: `../../references/plotting.ja.md`, `../../references/plotting.md`, `../../references/boundaries.ja.md`, `../../references/boundaries.md`, `../../references/distributed.ja.md`, `../../references/distributed.md`.
- Repo root docs only when the full checkout is available and may be newer, especially `docs/source/guide/article.ja.md` and `docs/source/guide/article.md`.

## Workflow

- Start from environment-variable usage when the user wants the same script to run in normal, record, and replay modes.
- Recommend `EMOUT_ARTICLE_RECORDS_PATH` explicitly for both record and replay.
- Use `EMOUT_ARTICLE_NAME` only when the user wants separate bundles per figure; otherwise explain that `default` can collect a notebook or script.
- Use `EMOUT_ARTICLE_SOURCE_NAME` for multiple simulation outputs or path changes after publication.
- Use `EMOUT_ARTICLE_ARCHIVE=zip` or `tar.gz` when upload size or packaging matters.
- For 2D figures, recommend slicing before plotting or `to_numpy()`.
- For 3D `plot_surfaces()`, recommend passing `bounds` so article data records the bounded ROI instead of a full 3D field.
- For time averages, recommend `data.phisp[-20:].mean()`; explain that article recording saves the averaged result, and with `plot_surfaces(bounds=...)` saves only the averaged ROI.
- Mention that the field returned by `mean()` exposes `inp`, `unit`, and `boundaries`, so existing helper functions can often accept it.
- State that article replay does not provide particles, backtrace, or remote execution itself; record the grid data needed for visualization.

## Output

Use concise sections when helpful:

```text
## 環境変数での使い方
...

## 平均・3D surface の記録
...

## 注意点
...
```

Keep examples short and adapt paths, `article_name`, source names, and variables to the user's script.
