
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(SCRIPT_DIR, "Social_Network_Ads.csv")
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    print(f"WARNING: '{csv_path}' not found. Using a sample dataset for demonstration.")
    data = pd.DataFrame([
        [19, 19000, 0],
        [35, 20000, 0],
        [26, 43000, 0],
        [27, 57000, 0],
        [19, 76000, 0],
        [27, 48000, 1],
        [27, 23000, 0],
        [32, 18000, 1],
        [25, 77000, 0],
        [35, 68000, 1]
    ], columns=["Age", "EstimatedSalary", "Purchased"])

print("First 5 Rows:")
print(data.head())

X = data[['Age', 'EstimatedSalary']]

y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
