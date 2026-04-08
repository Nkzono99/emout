---
name: run-tests
description: Run emout's test baseline with the known-broken TOML integration tests excluded. Use whenever you need a clean signal that your changes did not introduce regressions. Also the canonical way to refresh the test-count number quoted in AGENTS.md §10.
---

# run-tests

Run the project's test baseline with the pre-existing failures excluded so the signal is actionable.

## What to run

```bash
python -m pytest tests/ -q \
  --ignore=tests/utils/test_toml_converter.py \
  --ignore=tests/utils/test_toml_integration.py
```

Why those ignores:

- `tests/utils/test_toml_converter.py` fails at *collection time* because it imports `load_toml_as_namelist`, which was removed. That single collection error aborts the whole suite if you do not ignore the file.
- `tests/utils/test_toml_integration.py` depends on the external `toml2inp` binary. If the binary is not installed, Emout cannot parse `plasma.toml` fixtures and the tests fail with `'NoneType' object has no attribute 'nx'`. These failures are pre-existing and unrelated to every change we have made in this session.

Both of those are tracked in `AGENTS.md §10` — update that section if either situation changes.

## How to use the result

- The expected baseline (as of 2026-04-08) is `103 passed`. Any lower number means you introduced a regression; any higher number means you added tests and should update `AGENTS.md §10`.
- If you only touched surface_cut / boundaries, the narrower form is faster and equally informative:

  ```bash
  python -m pytest tests/test_boundaries.py tests/plot/test_surface_cut_mesh.py -q
  ```

- If you touched `Data3d`, also include `tests/data/test_data.py`.
- Do **not** silence failures by adding more `--ignore` flags. If a test breaks, fix the code or fix the test.

## When not to use

- If you are in the middle of a refactor and expect breakage — wait until you reach a natural checkpoint, then run.
- If the user has explicitly asked for speed and the change is tiny and localized, running the single relevant test file is enough.
