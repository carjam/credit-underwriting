# Model card (prototype)

## Purpose

Demonstrate **credit risk decisioning** on top of classifier outputs: tiers, approve/review/decline, threshold tradeoffs, SHAP views, and a directional loss lens. **Not** a production underwriting or capital system.

## Scope

| In scope | Out of scope |
|----------|----------------|
| Notebook-based training/eval on a fixed Lending Club–style sample | Live scoring API, model registry, or governance sign-off |
| Policy logic in `src/decisioning.py` + `config/policy.default.yaml` | Calibrated PD as regulatory default probability |
| Reproducibility via pinned `requirements.txt` and tests | Out-of-time validation on current vintages |

## Data

- **Default artifact:** `data/loans.csv` (public sample mirror; on the order of **~6.3k** rows, **2014-era** issue dates in the bundled file).
- **Override:** set `LENDING_CLUB_DATA_PATH` before the notebook load cell.
- **Label:** binary `loan_status` mapped to Fully Paid vs Charged Off after filtering (see notebook).

## Model outputs (notebook)

- **Classification:** `P(Fully Paid)` from the selected `best_model` (e.g. Random Forest / XGBoost after the notebook run).
- **Regression (secondary):** interest-rate linear model for pricing context.
- **Metrics cited in README** (e.g. ~0.66 ROC-AUC for logistic baseline, ~92% accuracy for tree models, high R² on rate): refer to **that notebook run on that data slice**; numbers move if data, seed, or preprocessing change.

## Validation snapshot (offline)

- Train/test split and CV as implemented in the notebook (not purged for leakage; prototype scope).
- `pytest` checks: decisioning math, optional full notebook execution in CI (`docs/TESTING.md`).

## Known limitations

- Scores are used for **ranking and policy illustration** unless you add explicit probability calibration.
- **No** CECL/IFRS 9/regulatory capital treatment.
- Bundled data is **historical sample**, not a claim about current Lending Club or any live portfolio.

## Reproduction

1. `pip install -r requirements.txt`
2. Run `Credit_Underwriting_Decisioning-Lending_Club.ipynb` end-to-end, or `pytest --run-notebook` (slow).
3. Record **git commit** and **data file** when reporting metrics.
