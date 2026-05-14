# Converted from PMAF_2/Assignment 2.ipynb
# Original notebook code cells exported as a normal Python script.

import pandas as pd
import numpy as np

df = pd.read_csv("diabetes.csv")

df.describe()

cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

for col in cols_with_zero:
    df[col].fillna(df[col].median(), inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt

for col in df.columns[:-1]:
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
