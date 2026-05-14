import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path(__file__).with_name("cleaned_student_data.csv")
df = pd.read_csv(DATA_PATH)

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df["pass"].value_counts())

sns.set()

sns.countplot(x="pass", data=df)
plt.title("Pass/Fail Distribution")
plt.show()

sns.boxplot(x="pass", y="study_hours", data=df)
plt.title("Study Hours vs Result")
plt.show()

sns.boxplot(x="pass", y="attendance", data=df)
plt.title("Attendance vs Result")
plt.show()

sns.histplot(df["final_grade"], kde=True)
plt.title("Final Grade Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
