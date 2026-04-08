# Testing and Regression Framework

This repository now includes a lightweight test framework to support ongoing regression checks.

## Test Layers

- **Unit tests** (`tests/test_decisioning.py`)
  - Validate decision-layer logic in `src/decisioning.py`:
    - risk tier mapping
    - approve/review/decline mapping
    - threshold sweep math
    - simple capital helper functions

- **Smoke test** (`tests/test_model_smoke.py`)
  - Verifies the local ML toolchain behaves correctly with a deterministic reference model run.
  - Uses `sklearn.datasets.load_breast_cancer` (no external data dependency).

- **Regression test** (`tests/test_model_regression.py`)
  - Compares deterministic model quality to baseline minimums in:
    - `tests/baselines/model_quality_baseline.json`
  - Fails if quality drops below floor values.

## Run Tests

```bash
pytest
```

Run only smoke tests:

```bash
pytest -m smoke
```

Run only regression checks:

```bash
pytest -m regression
```

## Updating Baselines Intentionally

If you intentionally change baseline expectations:

1. edit `tests/baselines/model_quality_baseline.json`
2. run `pytest -m regression`
3. include a commit note explaining why the floor changed

## Notes

- These tests validate **decisioning code correctness** and **environment/model health**.
- They do **not** execute the full notebook end-to-end.
- For notebook-level validation, run the notebook portfolio decisioning section after model cells have executed.
