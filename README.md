# Credit Underwriting Modeling

## Executive Summary
This project provides a business-ready, offline underwriting analytics prototype using Lending Club data. It benchmarks multiple machine learning approaches to estimate repayment risk (`loan_status`) and includes a secondary pricing-oriented regression lens (`int_rate`) to support more informed credit policy conversations.

## Business Context
Lending organizations must balance growth, loss control, and decision speed. Manual or static-rule underwriting can be inconsistent and slow to adapt. A model-led workflow improves consistency, supports portfolio risk calibration, and creates a stronger analytical foundation for policy updates.

## Problem Statement
- Predict default-related outcomes with usable discrimination power.
- Translate model outputs into practical risk signals for underwriting decisions.
- Evaluate model tradeoffs (precision, recall, ROC-AUC) for directional policy insight.

## Solution Overview:
### Predictive modeling
- End-to-end notebook workflow for ingestion, cleaning, encoding, scaling, and class balancing (`SMOTE`).
- Classification models include Logistic Regression, KNN, Random Forest, XGBoost, and boosted tree variants.
- A separate linear regression track models `int_rate` as a pricing/risk proxy.

### Decision layer
- Risk is represented through model outputs and class predictions for `loan_status`.
- Comparative model results support directional threshold and segment strategy discussions.
- Current repo scope is analytical; no production policy engine or API is implemented.

### Evaluation framework
- Performance reviewed with confusion matrix, classification report, precision, recall, F1, and ROC-AUC.
- ROC and precision-recall plots used for model comparison and tradeoff interpretation.
- Results are based on offline notebook experiments, not live production monitoring.

## Technical Implementation
- Primary artifact: `Credit_Underwriting_Decisioning-Lending_Club.ipynb`.
- Environment: Python 3.9.x notebook runtime.
- Core libraries: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, statsmodels, matplotlib, seaborn.
- Dataset path is configurable in notebook via `DATA_PATH`.

## Business Impact (Modeled)
- Directional improvement in identifying higher-risk applications versus simpler baseline approaches.
- More explicit precision/recall tradeoff visibility to inform approval-threshold and exception-policy discussions.
- Improved stakeholder alignment by translating model metrics into underwriting-relevant decision signals.
- Reusable experimentation baseline for future model governance and policy design iterations.

## Extension Opportunities
- Add explainability (SHAP/LIME) for model governance and stakeholder transparency.
- Introduce survival/time-to-default modeling for richer risk horizon analysis.
- Formalize train/validation/test protocols and model acceptance thresholds.
- Package notebook logic into modular code with reproducible pipelines and serving endpoints.

## Key Takeaways
- The repository represents a solid analytics prototype for underwriting strategy evaluation, not a production decisioning stack.
- Multi-model benchmarking enables practical, business-facing model selection and tradeoff discussions.
- The current implementation is suitable for directional policy insight and roadmap planning toward productionization.

## Skills Demonstrated
Credit Risk Modeling, Machine Learning, Underwriting Analytics, Feature Engineering, Model Evaluation, Python, Product-Oriented Data Science