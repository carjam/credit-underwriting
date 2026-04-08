# Portfolio Credit Risk Decisioning

This document describes how this repository elevates an existing underwriting ML notebook into a **portfolio-oriented credit risk decisioning narrative**—without retraining models.

## What Was Added

| Layer | Purpose |
|--------|---------|
| **Decision rules** | Map model scores to **approve / review / decline** using configurable thresholds. |
| **Risk tiers** | Label population as **prime**, **near-prime**, or **subprime** using P(good) bands. |
| **Threshold simulation** | Sweep approval cutoffs; report **approval rate** and **default rate among approved** (directional). |
| **SHAP** | **Global** feature importance (beeswarm/summary) and **one-loan** waterfall-style explanation. |
| **Capital framing** | Illustrative **expected loss per approved loan** using assumed **average loan size** and **LGD** (loss given default). |

## Where to Run It

Open `Credit_Underwriting_Decisioning-Lending_Club.ipynb` and run through the existing modeling cells so `best_model`, `X_test`, and `y_test` exist. Then execute the **Portfolio Credit Risk Decisioning Layer** section at the end.

## Reuse, Not Rebuild

- Classifiers and preprocessing are **unchanged** from the original assignment workflow.
- New logic lives in `src/decisioning.py` (pure functions) and the notebook cells that call it.

## Interpretation

Metrics on the SMOTE-balanced test split are **illustrative** of precision/recall and approval tradeoffs. For regulatory or capital reporting, rerun the same decision layer on an **unbiased holdout** or production cohort with documented as-of policy.
