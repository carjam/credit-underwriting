# End-to-End Credit Risk Decisioning and Capital Allocation System

## Executive Summary

This project reframes an existing Lending Club underwriting ML workflow into a **portfolio-grade credit risk decisioning story**. The original notebook still trains and compares classifiers **without redesign**; new work layers **approval thresholds**, **risk tiers**, **mapped decisions** (approve / review / decline), a **threshold simulation** (approval vs. default-rate tradeoff), **SHAP** explainability, and a **simple capital / loss-given-default** illustration—so stakeholders see **ML → decisions → business outcomes**.

## Business Context

Lenders must grow responsibly: approve enough credit to meet volume goals while containing default losses and using capital efficiently. Raw model scores are not decisions—teams need **explicit policy rules**, **tiering**, and **directional economics** to align risk appetite, underwriting operations, and portfolio strategy.

## Problem Statement

- Turn calibrated risk scores into **actionable underwriting decisions**.
- Make **tradeoffs visible**: stricter approval rules vs. volume and default experience among approved loans.
- Communicate **why** the model ranks risk (global and case-level explainability).
- Anchor discussion with **simple, transparent capital framing** (clarity over precision).

## Solution Overview:

### Predictive modeling

- Existing end-to-end notebook: ingestion, cleaning, encoding, scaling, **SMOTE**, and comparison of classifiers (e.g. Logistic Regression, KNN, Random Forest, XGBoost, boosted variants).
- Secondary **interest-rate** regression track for pricing context.
- **No retraining or architecture changes** as part of this elevation—models are reused as-is.

### Decision layer

- **`src/decisioning.py`**: maps **P(Fully Paid)** to **prime / near-prime / subprime** tiers and to **approve / review / decline** using configurable thresholds.
- Notebook section applies these rules on top of **`best_model`** scores on the test split.

### Evaluation framework

- Original metrics: confusion matrix, precision, recall, F1, ROC-AUC, ROC / PR plots.
- **Added**: threshold sweep with **approval rate** and **default rate among approved**; optional **expected loss per approved loan** using assumed **average loan amount** and **LGD**.
- **SHAP**: global importance (beeswarm/summary) and one **individual** explanation (waterfall-style where supported).

## Technical Implementation

| Artifact | Role |
|----------|------|
| `Credit_Underwriting_Decisioning-Lending_Club.ipynb` | Full pipeline + new **Portfolio Credit Risk Decisioning Layer** cells |
| `src/decisioning.py` | Decision tiers, actions, threshold sweep, simple capital helpers |
| `docs/PORTFOLIO_DECISIONING.md` | Stakeholder-oriented description of the decisioning add-on |
| `requirements.txt` | Dependencies including `shap` |

**Environment:** Python 3.9.x recommended (per notebook metadata). Configure dataset path via `DATA_PATH` in the notebook.

## Business Impact (Modeled)

Directional, offline illustration only:

- Clearer **policy levers** (thresholds and tiers) tied to model scores.
- **Tradeoff curves** linking approval volume to default experience among approved loans.
- **Explainability** for internal review and future governance conversations.
- **Illustrative P&L / loss** view via EL ≈ default_rate × LGD × average exposure per approved loan—not a substitute for full CECL/IFRS 9 or regulatory capital models.

## Extension Opportunities

- Deploy decision rules as a **policy engine** (config-driven thresholds, A/B testing).
- **Calibration** and **unbiased holdout** for reported default rates.
- **Survival / time-to-default** for multi-period capital and pricing.
- Full **capital allocation** optimization (constraints + objectives) once metrics are production-grade.
- **Monitoring**: drift, stability, and override analytics.

## Key Takeaways

- Demonstrates **ML → decisions → business translation** without rebuilding models.
- Suitable for **technical and non-technical** audiences: README + `docs/PORTFOLIO_DECISIONING.md` + notebook narrative.
- **Scope stays tight**: clarity, decisioning, and portfolio thinking—not a new modeling competition.

## Skills Demonstrated

Credit Risk, Underwriting Policy, Machine Learning, Decision Systems, Explainable AI (SHAP), Portfolio Simulation, Python, Product-Oriented Analytics
