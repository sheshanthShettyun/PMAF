# Serial mapped from current codebase: Data Cleaning and Preprocessing.py
# Cleaned current-codebase serial file.

# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
# Download dataset from Kaggle or UCI repository
csv_path = os.path.join(SCRIPT_DIR, "diabetes.csv")
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    print(f"WARNING: '{csv_path}' not found. Using a small sample dataset for demonstration.")
    data = pd.DataFrame([
        [6,148,72,35,0,33.6,0.627,50,1],
        [1,85,66,29,0,26.6,0.351,31,0],
        [8,183,64,0,0,23.3,0.672,32,1],
        [1,89,66,23,94,28.1,0.167,21,0],
        [0,137,40,35,168,43.1,2.288,33,1],
        [5,116,74,0,0,25.6,0.201,30,0],
        [3,78,50,32,88,31.0,0.248,26,1],
        [10,115,0,0,0,35.3,0.134,29,0]
    ], columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(data.head())

# Check dataset information
print("\nDataset Information:")
print(data.info())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Replace invalid zero values with NaN
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns:
    data[col] = data[col].replace(0, np.nan)

# Check missing values after replacement
print("\nMissing Values After Replacing 0 with NaN:")
print(data.isnull().sum())

# Fill missing values using mean
data.fillna(data.mean(), inplace=True)

# Verify missing values removed
print("\nMissing Values After Filling:")
print(data.isnull().sum())

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Feature Scaling
scaler = StandardScaler()

features = data.drop("Outcome", axis=1)

scaled_features = scaler.fit_transform(features)

# Convert scaled data into DataFrame
scaled_data = pd.DataFrame(scaled_features, columns=features.columns)

# Add target column back
scaled_data["Outcome"] = data["Outcome"]

# Display cleaned dataset
print("\nCleaned and Scaled Dataset:")
print(scaled_data.head())

# Save cleaned dataset
scaled_data.to_csv(os.path.join(SCRIPT_DIR, "cleaned_diabetes.csv"), index=False)

print("\nData Cleaning and Preprocessing Completed Successfully!")
