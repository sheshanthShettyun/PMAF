#pip install pandas
import pandas as pd
from pathlib import Path

# Read dataset
DATA_PATH = Path(__file__).with_name("student_performance.csv")
df = pd.read_csv(DATA_PATH)

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop 'student_id' column only if it exists
if 'student_id' in df.columns:
    df.drop('student_id', axis=1, inplace=True)

# Check data types
print("\nData Types:\n", df.dtypes)

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Show cleaned data
print("\nCleaned Data Preview:\n", df.head())
