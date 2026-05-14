# Converted from PMAF_3/Assignment 3.ipynb
# Original notebook code cells exported as a normal Python script.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("cereal.csv")

df.head()

df.columns

target = "calories"

X = df.drop(columns=["name", target])

y = df[target]

categorical_cols = ["mfr", "type"]

numeric_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# Extract trained regressor
regressor = model.named_steps["regressor"]

# Get feature names after encoding
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# Combine into DataFrame
coeff_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": regressor.coef_
})

print(coeff_df.sort_values(by="Coefficient", ascending=False))

sample = X.iloc[[0]]
prediction = model.predict(sample)

print("Actual:", y.iloc[0])
print("Predicted:", prediction[0])
