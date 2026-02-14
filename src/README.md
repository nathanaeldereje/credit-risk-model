# Source Code (`src/`)

This directory contains the core logic and library code for the Credit Risk Model.

## 📦 Internal Packages

- **`credit_risk/`**: The core engine.
  - `config.py`: Centralized project settings and constants.
  - `processing.py`: Data cleaning and aggregation logic.
  - `features.py`: RFM and Target engineering math.
  - `model.py`: Model training and metric calculation.
  - `explainability.py`: SHAP implementation for model transparency.
  - `utils.py`: Logging and I/O helpers.

- **`api/`**: Deployment layer.
  - `main.py`: FastAPI routes and model serving logic.
  - `pydantic_models.py`: Data validation schemas (Pydantic V2).

## 🛠 Development Guidelines
- **Modularity**: Keep logic in this directory and execution in `scripts/`.
- **Type Safety**: All new functions should include PEP 484 type hints.
- **Testing**: Corresponding tests must be added to the `tests/` root folder for any changes made here.