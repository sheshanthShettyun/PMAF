# Converted from PMAF_4/Assignment 4.ipynb
# Original notebook code cells exported as a normal Python script.

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Encode Gender
dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})

# Features and target
X = dataset[['Gender', 'Age', 'EstimatedSalary']]
y = dataset['Purchased']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy =", accuracy_score(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
