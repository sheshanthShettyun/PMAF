import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).with_name("student_performance.csv")
df = pd.read_csv(DATA_PATH)

print("Missing Values:\n", df.isnull().sum())

df.drop_duplicates(inplace=True)

if 'student_id' in df.columns:
    df.drop('student_id', axis=1, inplace=True)

print("\nData Types:\n", df.dtypes)

df = pd.get_dummies(df, drop_first=True)

print("\nCleaned Data Preview:\n", df.head())
