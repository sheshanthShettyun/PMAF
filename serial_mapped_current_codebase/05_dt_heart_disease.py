# Serial mapped from current codebase: Decision Tree Classification.py
# PMAF 5B: Decision Tree Classification on Heart Disease.

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

csv_path = "heart_disease.csv"
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    print(f"WARNING: '{csv_path}' not found. Using sample heart disease data.")
    data = pd.DataFrame({
        "age": [63, 37, 41, 56, 57, 57, 56, 44],
        "sex": [1, 1, 0, 1, 0, 1, 0, 1],
        "cp": [1, 2, 1, 1, 0, 0, 1, 2],
        "trestbps": [145, 130, 130, 120, 120, 140, 140, 120],
        "chol": [233, 250, 204, 236, 354, 192, 294, 263],
        "target": [1, 1, 1, 1, 1, 1, 1, 1],
    })

print("First 5 Rows:")
print(data.head())
print("\nMissing Values:")
print(data.isnull().sum())

data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(18, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=["No Disease", "Disease"])
plt.title("Decision Tree - Heart Disease Dataset")
plt.show()
