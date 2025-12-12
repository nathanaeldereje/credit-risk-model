# Credit Risk Probability Model for Alternative Data
**Bati Bank – Buy-Now-Pay-Later Credit Scoring Project**  
*December 2025*

An end-to-end machine learning project to build, deploy, and automate a credit risk model using alternative transactional data from an eCommerce partner. The model estimates default probability for BNPL customers using RFM-derived features and behavioral signals.

---
## Business Goal
Bati Bank is launching a Buy-Now-Pay-Later (BNPL) service in partnership with a leading eCommerce platform. To approve loans responsibly while expanding financial inclusion, we need a credit scoring model that:

- **Predicts the probability of default** using only alternative data (no traditional credit bureau scores).
- **Transforms customer behavioral patterns** (Recency, Frequency, Monetary) into predictive risk signals.
- **Outputs an interpretable credit score** and recommends optimal loan amount/duration.
- **Meets regulatory expectations** for risk measurement, interpretability, and documentation.
## Project Overview
We engineer a **proxy target variable** for credit risk (high-risk vs. low-risk customers) based on disengagement patterns, then train supervised models to predict this risk. The final product is a production-ready FastAPI service that serves real-time risk probabilities.

### Key Deliverables
- [ ] Comprehensive EDA with actionable insights
- [ ] Automated feature engineering pipeline (RFM + WoE/IV)
- [ ] Proxy target creation via RFM + K-Means clustering
- [ ] Multiple models (Logistic Regression, Random Forest, XGBoost/LightGBM) with MLflow tracking
- [ ] Best model registered and deployed via FastAPI + Docker
- [ ] CI/CD pipeline with linting and unit tests
- [ ] Full documentation and reproducible environment

---
## Credit Scoring Business Understanding
*(Task 1 Analysis)*

### 1. Basel II Implications
**Basel II Capital Accord** emphasizes robust, risk-sensitive measurement of credit risk, requiring banks to hold capital proportional to the actual risk of their portfolios (Pillar 1). This drives the need for **interpretable and well-documented models** that allow regulators to verify risk calculations and ensure fair treatment of borrowers.

### 2. The Need for Proxy Variables
Since the dataset lacks a direct "default" label, we create a **proxy variable based on RFM behavioral patterns** (low engagement → high risk of future default). 
*   **Business Risk:** Using a proxy introduces the risk of misclassifying truly good customers or under-estimating risk in edge cases. This could potentially lead to higher defaults (financial loss) or lost revenue from overly conservative approvals (opportunity cost).

### 3. Model Trade-offs
In a regulated environment, we must balance:
*   **Simple & Interpretable (e.g., Logistic Regression with WoE):** Preferred for easier regulatory approval, explainability, and auditability.
*   **Complex High-Performance (e.g., Gradient Boosting):** Offers better predictive accuracy but is harder to explain ("black box") and validate.

*Strategy: We prioritize interpretability while using ensemble methods where performance gains are significant, ensuring all model decisions are thoroughly documented.*

---

## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml          # GitHub Actions CI/CD
├── data/                             # Raw & processed data (add to .gitignore)
│   ├── raw/                          # Original Xente dataset
│   └── processed/                    # Feature-engineered data
├── notebooks/
│   └── eda.ipynb                     # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # Feature engineering + WoE/IV
│   ├── train.py                      # Model training & MLflow logging
│   ├── predict.py                    # Inference script
│   └── api/
│       ├── main.py                   # FastAPI application
│       └── pydantic_models.py        # Request/response validation
├── tests/
│   └── test_data_processing.py       # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```
## Tech Stack
- Core: Python 3.10+, pandas, numpy
- ML: scikit-learn, xgboost, lightgbm
- Tracking: mlflow (Experiment tracking & Model Registry)
- API: fastapi, uvicorn
- DevOps: Docker, GitHub Actions
- Quality: pytest, flake8/black
- Optional: xverse or scorecardpy (for WoE/IV)

## Quick Start
```bash
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download raw data to data/raw/ (from Kaggle or provided link)
# Then run feature engineering
python src/data_processing.py

# Train models and log to MLflow
python src/train.py

# Start the API locally
uvicorn src.api.main:app --reload
```
## Current Progress
| Task | Status | Notes |
| :--- | :--- | :--- |
| **Task 1 – Business Understanding** | ✅ Completed | README section written |
| **Task 2 – EDA** | 🟡 In Progress | Notebook ready, key insights being finalized |
| **Task 3 – Feature Engineering** | 📅 Planned | RFM + WoE/IV pipeline next |
| **Task 4 – Proxy Target** | 📅 Planned | RFM clustering & high-risk label |
| **Task 5 – Model Training** | 📅 Planned | MLflow setup + multiple models |
| **Task 6 – Deployment & CI/CD** | 📅 Planned | FastAPI, Docker, GitHub Actions |

Challenge completed – Dec _ 2025
Built by Nathanael Dereje