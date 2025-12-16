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
- Core: Python 3.12, pandas, numpy
- ML: scikit-learn, xgboost, lightgbm
- Tracking: mlflow (Experiment tracking & Model Registry)
- API: fastapi, uvicorn
- DevOps: Docker, GitHub Actions
- Quality: pytest, flake8/black
- Optional: xverse or scorecardpy (for WoE/IV)

## Quick Start
```bash   
# Clone the repository
git clone https://github.com/your-username/credit-risk-model.git
cd credit-risk-model

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # On Linux/Mac
# .venv\Scripts\activate           # On Windows

# Install dependencies
pip install -r requirements.txt
```


Place the raw Xente dataset in data/raw/data.csv
(Download from Kaggle: https://www.kaggle.com/competitions/xente-fraud-detection/data or provided link)
- Step 1: Run feature engineering (Task 3)
```bash 
   python src/data_processing.py  # → Generates data/processed/customer_features.csv
```


- Step 2: Create proxy target via RFM clustering (Task 4)
```bash 
   python src/create_target.py # → Generates data/processed/customer_features_with_target.csv (or .parquet)
#    with the binary target 'is_high_risk'
```



- Step 3: Train models with MLflow tracking (Task 5)
```bash 
   python src/train.py
```
```bash 
   mlflow ui
```


- Step 4: Start the FastAPI service locally (Task 6)
```bash 
   uvicorn src.api.main:app --reload
   # → API available at http://127.0.0.1:8000/docs
#    Test: POST to /predict with customer features
   
# Alternative: Run with Docker (Task 6)
docker-compose up --build
# → API available at http://localhost:8000
```

---
## Model Training Results (Task 5 Summary)

Trained and compared two models using GridSearchCV and full MLflow tracking on the RFM-derived proxy target `is_high_risk`.

### Final Performance Comparison

| Metric          | Random Forest | Logistic Regression | Winner          |
|-----------------|---------------|---------------------|-----------------|
| Accuracy        | **0.9786**    | 0.9559              | Random Forest   |
| Precision       | **0.9625**    | 0.9020              | Random Forest   |
| Recall          | **0.9706**    | 0.9664              | Random Forest   |
| F1 Score        | **0.9665**    | 0.9331              | Random Forest   |
| ROC-AUC         | **0.9988**    | 0.9933              | Random Forest   |

### Best Model: Random Forest
- Parameters: `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `class_weight='balanced'`
- **ROC-AUC = 0.9988** — outstanding discrimination
- High precision and recall balance risk control with customer inclusion
- Logged to MLflow experiment "Credit_Risk_Model_Experiment"
- Best model artifact saved and ready for registry/deployment

**Task 5: Fully Complete** — models trained, tuned, evaluated, and tracked.
---
## Current Progress(as of December 15, 2025)
| Task | Status | Notes |
| :--- | :--- | :--- |
| **Task 1 – Business Understanding** | ✅ Completed | README section written |
| **Task 2 – EDA** | ✅ Completed | Notebook ready, key insights finalized |
| **Task 3 – Feature Engineering** | ✅ Completed | Robust pipeline with aggregates, WoE/IV, logging |
| **Task 4 – Proxy Target** | ✅ Completed | RFM + K-Means → `is_high_risk` created                  |
| **Task 5 – Model Training** | ✅ Completed | Logistic Regression + Random Forest, full MLflow tracking, hyperparameter tuning |
| **Task 6 – Deployment & CI/CD** | ✅ Completed | FastAPI, Docker, GitHub Actions |

Challenge completed – Dec 16 2025
Built by Nathanael Dereje