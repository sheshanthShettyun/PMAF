import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

data = {
    'Advertising': [100,150,200,250,300,350,400,450,500,550],
    'Price': [10,12,11,13,12,14,13,15,14,16],
    'Sales': [200,250,300,350,400,450,500,550,600,650]
}

df = pd.DataFrame(data)

X = df[['Advertising', 'Price']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)

print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("Decision Tree R2:", r2_score(y_test, dt_pred))

importance = dt.feature_importances_

print("\nFeature Importance:")
for i, col in enumerate(X.columns):
    print(col, ":", importance[i])