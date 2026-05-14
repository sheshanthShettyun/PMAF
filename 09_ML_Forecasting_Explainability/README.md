# Assignment 09: ML Forecasting and Explainability

## Files
- `09_ml_forecasting_explainability.py`
- `sales.csv`
- `shared_drive_09_sales_models.py`

## Packages Required
```bash
python -m pip install pandas numpy matplotlib scikit-learn
```

Optional packages for the XGBoost and SHAP sections:
```bash
python -m pip install xgboost shap
```

The script includes Linear Regression, Decision Tree Regressor, Random Forest Regressor, MAE, R2 Score, feature importance, and SHAP explainability when SHAP is installed. It also runs without optional packages by using scikit-learn fallbacks.

Or install the common project requirements from the project root:
```bash
python -m pip install -r requirements.txt
```

## How to Run
From the project root:
```bash
python 09_ML_Forecasting_Explainability/09_ml_forecasting_explainability.py
```

If using VS Code, select the interpreter from your local `.venv` folder.

## Shared Drive Version
The shared Drive version is `shared_drive_09_sales_models.py`.

Run it from the project root:
```bash
python 09_ML_Forecasting_Explainability/shared_drive_09_sales_models.py
```
