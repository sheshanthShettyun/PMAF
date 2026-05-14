# Serial mapped from current codebase: Linear Regression Model.py
# Matched against pmafdhuv assignment order.

# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
csv_path = "cereal.csv"
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    print(f"WARNING: '{csv_path}' not found. Using a sample dataset for demonstration.")
    data = pd.DataFrame([
        [50, 1, 123, 0.5, 10, 8, 33, 110],
        [90, 3, 260, 1.2, 2, 25, 100, 170],
        [70, 2, 150, 1.5, 8, 12, 50, 140],
        [120, 4, 300, 2.0, 4, 10, 65, 200],
        [80, 2, 180, 1.0, 12, 5, 44, 150],
        [100, 3, 200, 0.8, 14, 6, 45, 180],
        [110, 5, 330, 1.7, 3, 30, 110, 190],
        [60, 1, 110, 0.3, 15, 4, 35, 120]
    ], columns=["calories", "protein", "fat", "sodium", "fiber", "carbo", "sugars", "potass"])

# Display first 5 rows
print("First 5 Rows:")
print(data.head())

# Select numerical features
features = ['protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass']

# Target variable
target = 'calories'

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Define X and y
X = data[features]
y = data[target]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Display predictions
print("\nPredicted Calories:")
print(y_pred[:10])

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# Display coefficients
print("\nModel Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef}")

print("\nIntercept:", model.intercept_)
