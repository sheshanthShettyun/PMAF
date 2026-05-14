# Converted from PMAF_5/Heart Disease.ipynb
# Original notebook code cells exported as a normal Python script.

# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('heart_disease.csv')

# First 5 rows
df.head()

df.info()

# Handle missing values
print(df.isnull().sum())
df = df.dropna()

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train decision tree model
model = DecisionTreeClassifier(
    criterion='entropy',   # or 'gini'
    max_depth=5,
    random_state=42
)

# Fit model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy =", accuracy_score(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Tree visualization
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(18,10))
plot_tree(model, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
plt.show()
