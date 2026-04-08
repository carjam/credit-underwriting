# Credit Underwriting Modeling (Lending Club Notebook)

## Overview
This repository currently contains a single, notebook-based machine learning workflow for credit underwriting analysis using Lending Club data.

The project focuses on:
- predicting `loan_status` (Charged Off vs Fully Paid), and
- modeling `int_rate` in a separate regression track.

It is an experimental/offline analysis project, not a deployed production decisioning service.

---

## Repository Contents
- `Credit_Underwriting_Decisioning-Lending_Club.ipynb`: end-to-end notebook for data prep, EDA, feature processing, model training, and evaluation.
- `README.md`: project documentation.

---

## What Is Implemented
### Data and preprocessing
- Lending Club CSV ingestion
- data cleaning and column filtering
- categorical encoding and scaling
- class imbalance handling with `SMOTE`

### Classification workflow (`loan_status`)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest (including tuning/cross-validation blocks)
- XGBoost (including tuning blocks)
- Decision Tree variants with AdaBoost / Gradient Boosting sections

### Regression workflow (`int_rate`)
- Linear Regression
- OLS/statistical summary output

### Evaluation artifacts
- confusion matrix / classification report
- precision, recall, F1, ROC-AUC
- ROC and precision-recall visualizations

---

## Setup and Reproducibility
### Environment
The notebook metadata indicates Python 3.9.x (`3.9.7`).

At minimum, install:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`
- `xgboost`
- `imbalanced-learn`

### Data path
The notebook currently reads data from a local path:
- `~/Desktop/IK/LendingClub/loans.csv`

Update this path in the notebook to your local dataset location before running.

### Recommended next reproducibility improvements
- add `requirements.txt` or `environment.yml`
- parameterize input/output paths
- save trained model artifacts and run metadata

---

## Interpreting Results
All model performance and impact statements should be interpreted as offline experimental results on the provided dataset, not live production outcomes.

---

## Known Limitations
- single-notebook workflow with no packaging or automated tests
- no deployment/API layer in this repository
- no model registry, experiment tracker, or versioned artifacts
- external local data dependency path in current notebook
- potential leakage/generalization risks should be explicitly re-validated before operational use

---

## Future Enhancements
- add SHAP/LIME explainability
- add survival/time-to-default modeling
- formalize train/validation/test protocols and acceptance thresholds
- add a model card and data dictionary
- package notebook logic into reusable modules + CLI/API

---

## Skills Demonstrated
Credit Risk Modeling, Machine Learning, Exploratory Data Analysis, Feature Engineering, Model Evaluation, Python, Risk Analytics