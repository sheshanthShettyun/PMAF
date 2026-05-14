
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

print("First 5 Rows of Dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns:
    data[col] = data[col].replace(0, np.nan)

print("\nMissing Values After Replacing 0 with NaN:")
print(data.isnull().sum())

data.fillna(data.mean(), inplace=True)

print("\nMissing Values After Filling:")
print(data.isnull().sum())

data.drop_duplicates(inplace=True)

scaler = StandardScaler()

features = data.drop("Outcome", axis=1)

scaled_features = scaler.fit_transform(features)

scaled_data = pd.DataFrame(scaled_features, columns=features.columns)

scaled_data["Outcome"] = data["Outcome"]

print("\nCleaned and Scaled Dataset:")
print(scaled_data.head())

scaled_data.to_csv(os.path.join(SCRIPT_DIR, "cleaned_diabetes.csv"), index=False)

print("\nData Cleaning and Preprocessing Completed Successfully!")

loan_data = pd.DataFrame({
    "ApplicantIncome": [5000, 3200, 4100, np.nan, 6200, 3200],
    "LoanAmount": [150, 95, np.nan, 120, 180, 95],
    "Gender": ["Male", "Female", "Male", "Female", np.nan, "Female"],
    "Married": ["Yes", "No", "Yes", np.nan, "Yes", "No"],
    "Credit_History": [1, 1, 0, 1, np.nan, 1],
    "Loan_Status": ["Y", "N", "Y", "Y", "N", "N"],
})

print("\nLoan Preprocessing Sample:")
print(loan_data)

print("\nLoan Missing Values:")
print(loan_data.isnull().sum())

loan_data.drop_duplicates(inplace=True)

numeric_columns = loan_data.select_dtypes(include=np.number).columns
categorical_columns = loan_data.select_dtypes(exclude=np.number).columns

for col in numeric_columns:
    loan_data[col] = loan_data[col].fillna(loan_data[col].mean())

for col in categorical_columns:
    loan_data[col] = loan_data[col].fillna(loan_data[col].mode()[0])

encoded_loan_data = pd.get_dummies(loan_data, drop_first=True)

loan_scaler = StandardScaler()
scaled_loan_data = encoded_loan_data.copy()
scaled_loan_data[numeric_columns] = loan_scaler.fit_transform(encoded_loan_data[numeric_columns])

print("\nEncoded and Scaled Loan Data:")
print(scaled_loan_data.head())

scaled_loan_data.to_csv(os.path.join(SCRIPT_DIR, "preprocessed_loan_sample.csv"), index=False)
