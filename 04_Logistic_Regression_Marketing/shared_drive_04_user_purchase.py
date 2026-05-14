import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_PATH = Path(__file__).with_name("user_purchase.csv")
df = pd.read_csv(DATA_PATH)

df.columns = ["Age", "Salary", "Purchased"]

print(df.head())

X = df[["Age", "Salary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.scatter(df["Age"], df["Salary"], c=df["Purchased"])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Purchase Prediction")
plt.show()
