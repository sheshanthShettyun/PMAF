# Serial mapped from current codebase: Machine Learning Models for Forecasting & Explainability.py
# Matched against pmafdhuv assignment order.

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor
import shap

# Load dataset
csv_path = "sales.csv"
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

# Display first rows
print("First 5 Rows:")
print(data.head())

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Select features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 1. Linear Regression
# -------------------------------

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLinear Regression R2 Score:",
      r2_score(y_test, lr_pred))

# -------------------------------
# 2. Random Forest Regressor
# -------------------------------

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Random Forest R2 Score:",
      r2_score(y_test, rf_pred))

# -------------------------------
# 3. XGBoost Regressor
# -------------------------------

xgb_model = XGBRegressor()

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("XGBoost R2 Score:",
      r2_score(y_test, xgb_pred))

# -------------------------------
# Compare Models
# -------------------------------

models = ['Linear Regression',
          'Random Forest',
          'XGBoost']

scores = [
    r2_score(y_test, lr_pred),
    r2_score(y_test, rf_pred),
    r2_score(y_test, xgb_pred)
]

plt.bar(models, scores)

plt.ylabel("R2 Score")
plt.title("Model Comparison")

plt.show()

# -------------------------------
# Feature Importance
# -------------------------------

importance = rf_model.feature_importances_

plt.bar(X.columns, importance)

plt.title("Feature Importance")
plt.ylabel("Importance")

plt.show()

# -------------------------------
# SHAP Explainability
# -------------------------------

explainer = shap.Explainer(xgb_model)

shap_values = explainer(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test)

print("\nExplainability Completed Successfully!")