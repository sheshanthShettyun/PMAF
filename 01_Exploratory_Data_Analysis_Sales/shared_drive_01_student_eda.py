import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Read CSV file
DATA_PATH = Path(__file__).with_name("cleaned_student_data.csv")
df = pd.read_csv(DATA_PATH)

# Basic Information
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df["pass"].value_counts())

# Seaborn style
sns.set()

# Pass/Fail Distribution
sns.countplot(x="pass", data=df)
plt.title("Pass/Fail Distribution")
plt.show()

# Study Hours vs Result
sns.boxplot(x="pass", y="study_hours", data=df)
plt.title("Study Hours vs Result")
plt.show()

# Attendance vs Result
sns.boxplot(x="pass", y="attendance", data=df)
plt.title("Attendance vs Result")
plt.show()

# Final Grade Distribution
sns.histplot(df["final_grade"], kde=True)
plt.title("Final Grade Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
