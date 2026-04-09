---
name: run-tests
description: Run emout's full pytest baseline. Use whenever you need a clean signal that your changes did not introduce regressions. Also the canonical way to refresh the test-count number quoted in AGENTS.md §10.
---

# run-tests

Run the project's test baseline. As of 2026-04-09 the suite is clean, no ignores needed.

## What to run

```bash
python -m pytest tests/ -q
```

No `--ignore` flags are required — the previously broken
`tests/utils/test_toml_converter.py` and `tests/utils/test_toml_integration.py`
were repaired on 2026-04-09:

- `test_toml_converter.py` now only tests `TomlData` / `load_toml` (the
  `_convert_v*` / `load_toml_as_namelist` helpers were removed upstream
  and the tests that imported them were deleted).
- `test_toml_integration.py` carries a `pytestmark = pytest.mark.skipif(...)`
  that skips the file when `toml2inp` is not on PATH, so the suite stays
  green even without MPIEMSES3D installed.

## How to use the result

- The expected baseline (as of 2026-04-09) is `177 passed`. On machines
  without `toml2inp`, expect `153 passed, 19 skipped` instead; both are
  clean signals. Any *failure* or lower passed-count means a regression.
- If you added tests, update the `172` number in `AGENTS.md §10` and in
  the paragraph above.
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
