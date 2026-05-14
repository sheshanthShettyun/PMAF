
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ModuleNotFoundError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ModuleNotFoundError:
    shap = None
    SHAP_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(SCRIPT_DIR, "sales.csv")
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    print(f"WARNING: '{csv_path}' not found. Using sample data for demonstration.")
    data = pd.DataFrame({
        'TV': [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8],
        'Radio': [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6],
        'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 1.0, 21.2],
        'Sales': [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6]
    })

print("First 5 Rows:")
print(data.head())

data.fillna(data.mean(numeric_only=True), inplace=True)

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLinear Regression R2 Score:",
      r2_score(y_test, lr_pred))
print("Linear Regression MAE:",
      mean_absolute_error(y_test, lr_pred))

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\nDecision Tree R2 Score:",
      r2_score(y_test, dt_pred))
print("Decision Tree MAE:",
      mean_absolute_error(y_test, dt_pred))


rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Random Forest R2 Score:",
      r2_score(y_test, rf_pred))
print("Random Forest MAE:",
      mean_absolute_error(y_test, rf_pred))


if XGBOOST_AVAILABLE:
    xgb_model = XGBRegressor(random_state=42)
    xgb_label = "XGBoost"
else:
    print("\nXGBoost is not installed. Using GradientBoostingRegressor fallback.")
    xgb_model = GradientBoostingRegressor(random_state=42)
    xgb_label = "Gradient Boosting"

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print(f"{xgb_label} R2 Score:",
      r2_score(y_test, xgb_pred))
print(f"{xgb_label} MAE:",
      mean_absolute_error(y_test, xgb_pred))


models = ['Linear Regression',
          'Decision Tree',
          'Random Forest',
          xgb_label]

scores = [
    r2_score(y_test, lr_pred),
    r2_score(y_test, dt_pred),
    r2_score(y_test, rf_pred),
    r2_score(y_test, xgb_pred)
]

plt.bar(models, scores)

plt.ylabel("R2 Score")
plt.title("Model Comparison")

plt.show()


importance = rf_model.feature_importances_

plt.bar(X.columns, importance)

plt.title("Feature Importance")
plt.ylabel("Importance")

plt.show()


if SHAP_AVAILABLE:
    explainer = shap.Explainer(rf_model)

    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test)
else:
    print("\nSHAP is not installed. Showing Random Forest feature importances instead.")
    fallback_importance = getattr(rf_model, "feature_importances_", None)
    if fallback_importance is not None:
        for feature, value in zip(X.columns, fallback_importance):
            print(f"{feature}: {value:.4f}")
    else:
        print("Feature importance is not available for this fallback model.")

print("\nExplainability Completed Successfully!")
