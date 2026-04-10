---
name: run-tests
description: Run emout's full pytest baseline. Use whenever you need a clean signal that your changes did not introduce regressions.
---

# run-tests

Run the project's test baseline. The suite is clean, no ignores needed.

## What to run

```bash
python -m pytest tests/ -q
```

`test_toml_integration.py` は `toml2inp` が PATH 上で動作可能な場合のみ実行される（`skipif` 付き）。
`toml2inp` 未導入でもテスト全体はグリーンを維持する。

## How to use the result

- The expected baseline (as of 2026-04-10) is `1321 passed`. On machines
  without `toml2inp`, expect some tests skipped instead; both are
  clean signals. Any *failure* or lower passed-count means a regression.
- If you added tests, update the baseline number in `CLAUDE.md` / `AGENTS.md`
  and in the paragraph above.
- If you only touched surface_cut / boundaries, the narrower form is
  faster and equally informative:

  ```bash
  python -m pytest tests/test_boundaries.py tests/plot/test_surface_cut_mesh.py -q
  ```

- If you touched `Data3d`, also include `tests/data/test_data.py`.
- Do **not** silence failures by adding `--ignore` flags. If a test
  breaks, fix the code or fix the test.

## When not to use

- If you are in the middle of a refactor and expect breakage — wait
  until you reach a natural checkpoint, then run.
- If the user has explicitly asked for speed and the change is tiny and
  localized, running the single relevant test file is enough.
