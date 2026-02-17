# CLI Scripts

This directory contains the entry-point scripts for the Credit Risk pipeline. These scripts are thin wrappers that call the core logic in `src/credit_risk/`.

## 🚀 Execution Order

To build the project from scratch, run the scripts in the following order:

### 1. Preprocessing & Target Engineering
```bash
python -m scripts.run_preprocessing
```
- **Inputs:** `data/raw/data.csv`
- **Outputs:** `data/processed/customer_features_with_target.csv`
- **Description:** Performs cleaning, RFM aggregation, and K-Means clustering to generate the `is_high_risk` target variable.

### 2. Model Training
```bash
python -m scripts.run_training
```
- **Inputs:** `data/processed/customer_features_with_target.csv`
- **Outputs:** Trained model artifacts logged to MLflow.
- **Description:** Runs a GridSearchCV for Logistic Regression and Random Forest. Metrics and models are tracked in the `Credit_Risk_Production_Training` experiment.

### 3. Explainability (SHAP)
```bash
python -m scripts.run_explainability
```
- **Inputs:** Best model from MLflow + processed data.
- **Outputs:** `reports/figures/shap_summary.png` and `reports/figures/shap_local_high_risk.png`.
- **Description:** Generates global and local feature importance plots to satisfy financial transparency requirements.

---

## 🛠 Usage Notes
- **Environment:** Ensure the virtual environment is activated before running.
- **Paths:** All scripts use relative paths defined in `src/credit_risk/config.py`.
- **Module Execution:** Always run scripts using the `-m` flag from the root directory to ensure `src` is correctly added to the Python path.
