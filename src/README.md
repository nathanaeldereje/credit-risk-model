# src – Production Pipeline
Clean, modular, well-documented Python scripts designed to be run sequentially or orchestrated.

| Script                        | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| **`data_processing.py`**      | Loads raw data, performs cleaning, generates customer-level aggregates (total/avg/std transaction amount, fees, refunds, time stats), adds most frequent categorical modes per customer, and saves processed dataset for RFM and modeling. |
| **`create_proxy_target.py`**  | **New** – Loads aggregated customer features, calculates RFM metrics (Recency, Frequency, Monetary), applies log transformation and standardization, performs K-Means clustering (k=3), identifies the high-risk (disengaged) cluster, creates the binary proxy target `is_high_risk`, and saves the final modeling dataset. |
| **`train.py`**                | Trains multiple models (Logistic Regression, Random Forest, XGBoost/LightGBM), logs experiments to MLflow, and registers the best model. |
| **`predict.py`**              | Loads the registered model and performs inference on new customer data.     |
| **`api/main.py`**             | FastAPI application that serves real-time risk probability predictions.    |
| **`api/pydantic_models.py`**  | Pydantic schemas for request/response data validation.                      |

---

### Usage Order
For a fresh run, execute the scripts in the following order:

1. **Feature Engineering:** `python data_processing.py`
2. **Model Training:** `python train.py`
3. **Inference (Batch):** `python predict.py`
4. **Inference (Real-time API):** `uvicorn api.main:app --reload`