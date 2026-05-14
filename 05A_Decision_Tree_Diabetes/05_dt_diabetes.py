
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(SCRIPT_DIR, "diabetes.csv")
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    print(f"WARNING: '{csv_path}' not found. Using sample diabetes data.")
    data = pd.DataFrame({
        "Pregnancies": [6, 1, 8, 1, 0, 5, 3, 10],
        "Glucose": [148, 85, 183, 89, 137, 116, 78, 115],
        "BloodPressure": [72, 66, 64, 66, 40, 74, 50, 0],
        "SkinThickness": [35, 29, 0, 23, 35, 0, 32, 0],
        "Insulin": [0, 0, 0, 94, 168, 0, 88, 0],
        "BMI": [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3],
        "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134],
        "Age": [50, 31, 32, 21, 33, 30, 26, 29],
        "Outcome": [1, 0, 1, 0, 1, 0, 1, 0],
    })

cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zero:
    data[col] = data[col].replace(0, np.nan)
    data[col] = data[col].fillna(data[col].median())

print("First 5 Rows:")
print(data.head())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(15, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=["No Diabetes", "Diabetes"])
plt.title("Decision Tree - Diabetes Dataset")
plt.show()
