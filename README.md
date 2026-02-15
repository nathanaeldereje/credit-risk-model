# Credit Risk Probability Model for Alternative Data
**Bati Bank – Buy-Now-Pay-Later Credit Scoring Project**  
*December 2025*

An end-to-end machine learning project to build, deploy, and automate a credit risk model using alternative transactional data from an eCommerce partner. The model estimates default probability for BNPL customers using RFM-derived features and behavioral signals.    
Deployed Output: https://credit-risk-model-3zqyyrjpsbg66xzfpgf8nv.streamlit.app/
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

## 🚀 Live Demo

**Interactive Dashboard:**  
https://credit-risk-model-3zqyyrjpsbg66xzfpgf8nv.streamlit.app/


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
## Top 5 Key Insights from EDA (Task 2 Deliverable)

1. **Severe Class Imbalance in Target**  
   Only 193 fraud cases out of 95,662 transactions → **0.20% fraud rate**. This confirms the need for a proxy target (RFM-based disengagement) instead of using FraudResult directly.

2. **Amount vs Value Relationship & Hidden Fees**  
   - 39.9% of transactions have negative Amount (refunds/cash-ins)  
   - In 2.68% of cases, `Value > |Amount|` → this difference is a **transaction fee/commission** (0.55–5,400 UGX)  
   → **Strong behavioral signal**: customers paying higher fees may be higher-risk or higher-value

3. **Extreme Right-Skew in Transaction Amounts**  
   Most transactions are small (< 10,000 UGX), but a long tail exists up to 98.8 million UGX. Log transformation clearly needed for modeling.

4. **Dominant Categories & Channels**  
   - Top 2 ProductCategories (financial_services + airtime) = 94% of volume  
   - ChannelId_3 (likely USSD) dominates with 59%, ChannelId_2 (web/app) with 39%  
   → Clear customer segments emerging

5. **Single Country & Currency**  
   All transactions are in Uganda (CountryCode 256, Currency UGX) → we can safely drop these columns.

**Bonus Insight**: Average customer has ~25 transactions over ~19 active days — perfect for RFM segmentation.
---

## Proxy Target Creation (Task 4 Summary)

Since the dataset contains no direct "default" or "credit risk" label, we engineered a **behavioral proxy target** called `is_high_risk` to enable supervised learning.

### Approach
- Calculated **RFM metrics** for each customer:
  - **Recency**: Days since last transaction (snapshot date = last transaction date + 1 day)
  - **Frequency**: Total number of transactions
  - **Monetary**: Total absolute transaction amount (`total_amount`)
- Applied **log transformation** (`log1p`) to handle extreme skew in Frequency and Monetary
- **Standardized** the transformed RFM features
- Performed **K-Means clustering** (k=3, `random_state=42`) on scaled RFM data
- Analyzed cluster profiles and identified the **high-risk cluster** as the one with the **lowest Monetary value** (strongly correlated with low Frequency and high Recency — classic disengagement pattern)
- Assigned `is_high_risk = 1` to customers in the high-risk cluster, `0` otherwise

### Result
- High-risk customers represent approximately **XX%** of the population (exact % shown when script is run)
- Final dataset saved as `data/processed/customer_features_with_target.csv` (or .parquet), ready for model training

This proxy enables us to train predictive models that estimate default probability based on behavioral disengagement — a standard and effective approach in alternative credit scoring when true default labels are unavailable.
---
## Project Structure

```text
credit-risk-model/
├── .github/workflows/ci.yml   # GitHub Actions (Linting, Testing, Coverage)
├── data/                      # Data directory (Git-ignored)
│   ├── raw/                   # Raw transaction data
│   └── processed/             # Aggregated customer features & targets
├── notebooks/                 # Exploratory Research
│   └── eda.ipynb              # Initial Data Discovery
├── reports/                   # Model Explainability (SHAP) reports
│   └── figures/               # Global & Local explanation plots
├── scripts/                   # CLI Entry points for the pipeline
│   ├── run_preprocessing.py   # Full data pipeline (Loading -> RFM -> Target)
│   ├── run_training.py        # MLflow-tracked training & GridSearch
│   └── run_explainability.py  # SHAP value generation
├── src/                       # Core Source Code
│   ├── credit_risk/           # Core Logic Package (The "Engine")
│   │   ├── config.py          # Dataclasses & centralized constants
│   │   ├── processing.py      # Modular transaction logic
│   │   ├── features.py        # RFM & Clustering (Target Engineering)
│   │   ├── model.py           # Training logic & validation metrics
│   │   ├── explainability.py  # SHAP logic for pipeline models
│   │   └── utils.py           # Logging & I/O helpers
│   ├── api/                   # Backend: FastAPI Application
│   │   ├── main.py            # API routes & model loading logic
│   │   └── pydantic_models.py # Pydantic V2 data validation
│   └── dashboard/             # NEW: Frontend: Streamlit Application
│       └── app.py             # Interactive UI & Risk Simulator
├── tests/                     # 14+ Unit & Integration Tests
└── requirements.txt           # Project dependencies
```
## Tech Stack
- **Core:** Python 3.12, Pandas, NumPy
- **ML & Explainability:** Scikit-learn, **SHAP** (SHapley Additive exPlanations)
- **Tracking:** MLflow (Experiment tracking with Model Signatures)
- **API (Backend):** FastAPI (Pydantic V2), Uvicorn
- **Dashboard (Frontend):** **Streamlit**, Requests
- **DevOps:** Docker, GitHub Actions, Pytest-cov
- **Quality:** Flake8, Type Hinting, Dataclasses


## Quick Start
```bash   
# Clone the repository
git clone https://github.com/nathanaeldereje/credit-risk-model.git
cd credit-risk-model

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # On Linux/Mac
# .venv\Scripts\activate           # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Execution Pipeline

**Step 0: Exploratory Data Analysis (EDA)**  
Review `notebooks/eda.ipynb` for initial data validation, distributions, and outlier detection.

**Step 1: Data Pipeline & Target Engineering**
```bash 
python -m scripts.run_preprocessing
```

**2. Production Training & MLflow Tracking**
```bash 
python -m scripts.run_training
```

**3. Generate Explainability Reports (SHAP)**
```bash 
python -m scripts.run_explainability
```

**4. Start the Backend API**
```bash 
uvicorn src.api.main:app --reload
```

**5. Launch the Interactive Dashboard**
```bash 
# Ensure the API is running in another terminal
streamlit run src/dashboard/app.py
```
---

## Model Training Results (Production Upgrade)

After refactoring the codebase and incorporating tenure-based features (`active_days`), the model achieved superior performance compared to initial baselines.

### Final Performance Comparison

| Metric          | Random Forest (New) | Logistic Regression | Improvement (vs Prev) |
|-----------------|---------------------|---------------------|-----------------------|
| **Accuracy**    | **0.9799**          | 0.9559              | +0.93%                |
| **Precision**   | **0.9665**          | 0.9020              | **+2.03%**            |
| **Recall**      | **0.9705**          | 0.9664              | +0.84%                |
| **F1 Score**    | **0.9685**          | 0.9331              | +1.44%                |
| **ROC-AUC**     | **0.9986**          | 0.9933              | Stable (Excellence)   |

### Best Model: Random Forest
- **Key Discovery:** Feature engineering revealed that **Monetary Volume** and **Account Tenure** are the strongest predictors of repayment behavior.
- **Explainability:** Integrated SHAP Waterfall plots provide "Reason Codes" for every prediction, ensuring regulatory compliance and model transparency.
- **Deployment:** Best model is automatically queried from MLflow via the API based on the highest ROC-AUC.

**Task 5: Fully Complete** — models trained, tuned, evaluated, and tracked.
---
## Current Progress
| Task | Status | Notes |
| :--- | :--- | :--- |
| **Task 1 – Business Understanding** | ✅ Completed | README section written |
| **Task 2 – EDA** | ✅ Completed | Notebook ready, key insights finalized |
| **Task 3 – Feature Engineering** | ✅ Completed | Robust pipeline with aggregates, WoE/IV, logging |
| **Task 4 – Proxy Target** | ✅ Completed | RFM + K-Means → `is_high_risk` created                  |
| **Task 5 – Model Training** | ✅ Completed | Logistic Regression + Random Forest, full MLflow tracking, hyperparameter tuning |
| **Task 6 – Deployment & CI/CD** | ✅ Completed | FastAPI, Docker, GitHub Actions |

<!-- Challenge completed – Dec 16 2025 -->
Built by Nathanael Dereje