# Runbook

## Environment

- Python: `3.13.x` tested locally
- Install: `pip install -r requirements.txt`
- Deterministic test seeds are enforced in `tests/conftest.py`.

## Data snapshot (bundled sample)

- File: `data/loans.csv`
- Source: public Lending Club sample mirror (documented in commit history)
- Rows: `6305`
- Example issue date in sample: `Dec-2014`

If you use a different CSV, set `LENDING_CLUB_DATA_PATH` before running the notebook.

## Standard execution path

1. Fast checks:
   - `pytest`
2. Full notebook execution check:
   - `pytest --run-notebook`
3. Interactive notebook:
   - Open `Credit_Underwriting_Decisioning-Lending_Club.ipynb`
   - Run all cells from a clean kernel

## Policy-driven decisioning outside Jupyter

Use the CLI with a score file that contains `p_good`:

`python scripts/run_decisioning.py --scores-csv <scores.csv> --policy config/policy.default.yaml`

Optional: include `y_true` in `scores.csv` to compute default-rate and expected-loss summary in CLI output.
