# Serial mapped from current codebase: Deep Learning Forecasting & Model Deployment.py
# Matched against pmafdhuv assignment order.

# =========================================================
# Deep Learning Forecasting using LSTM + Flask Deployment
# =========================================================

# Install Required Libraries
# pip install pandas numpy matplotlib scikit-learn tensorflow flask

# =========================================================
# Import Libraries
# =========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

from flask import Flask, request

# =========================================================
# Load Dataset
# =========================================================

# Dataset should contain a 'Close' column
csv_path = "stock_data.csv"
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    print(f"WARNING: '{csv_path}' not found. Using sample stock price data.")
    close_prices = np.linspace(150.0, 250.0, 100)
    data = pd.DataFrame({'Close': close_prices})

print("First 5 Rows:")
print(data.head())

# =========================================================
# Select Closing Price
# =========================================================

dataset = data['Close'].values

# Reshape data
dataset = dataset.reshape(-1, 1)

# =========================================================
# Feature Scaling
# =========================================================

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(dataset)

# =========================================================
# Prepare Training Data
# =========================================================

X_train = []
y_train = []

# Using previous 60 values to predict next value

for i in range(60, len(scaled_data)):

    X_train.append(scaled_data[i-60:i, 0])

    y_train.append(scaled_data[i, 0])

# Convert into NumPy arrays

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape for LSTM

X_train = np.reshape(
    X_train,
    (X_train.shape[0], X_train.shape[1], 1)
)

print("\nTraining Data Shape:")
print(X_train.shape)

# =========================================================
# Build LSTM Model
# =========================================================

model = Sequential()

# First LSTM Layer
model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
    )
)

# Second LSTM Layer
model.add(LSTM(units=50))

# Output Layer
model.add(Dense(units=1))

# =========================================================
# Compile Model
# =========================================================

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# =========================================================
# Train Model
# =========================================================

print("\nTraining Model...")

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
)

# =========================================================
# Save Model
# =========================================================

model.save("lstm_model.h5")

print("\nModel Saved Successfully!")

# =========================================================
# Predict Future Value
# =========================================================

# Take last 60 values
test_data = scaled_data[-60:]

X_test = []

X_test.append(test_data[:,0])

X_test = np.array(X_test)

# Reshape for prediction

X_test = np.reshape(
    X_test,
    (X_test.shape[0], X_test.shape[1], 1)
)

# Predict

predicted_price = model.predict(X_test)

# Convert back to original values

predicted_price = scaler.inverse_transform(predicted_price)

print("\nPredicted Future Price:")
print(predicted_price)

# =========================================================
# Plot Graph
# =========================================================

plt.figure(figsize=(10,5))

plt.plot(dataset, label='Original Prices')

plt.scatter(
    len(dataset),
    predicted_price[0][0],
    label='Predicted Price'
)

plt.title("LSTM Stock Price Prediction")

plt.xlabel("Time")
plt.ylabel("Stock Price")

plt.legend()

plt.show()

# =========================================================
# Flask Deployment
# =========================================================

app = Flask(__name__)

# Load saved model
loaded_model = load_model("lstm_model.h5")

# Home Route

@app.route('/')

def home():

    return '''
    <h2>LSTM Stock Price Prediction</h2>

    <form action="/predict" method="post">

        <input type="text" name="values"
        placeholder="Enter 60 comma-separated values">

        <button type="submit">Predict</button>

    </form>
    '''

# Prediction Route

@app.route('/predict', methods=['POST'])

def predict():

    # Get input values
    values = request.form['values']

    # Convert into list
    values = [float(x) for x in values.split(',')]

    # Convert into array
    input_data = np.array(values)

    # Reshape
    input_data = input_data.reshape(1, 60, 1)

    # Predict
    prediction = loaded_model.predict(input_data)

    # Convert back
    prediction = scaler.inverse_transform(prediction)

    return f'''
    <h3>Predicted Future Stock Price:</h3>

    <h2>{prediction[0][0]}</h2>
    '''

# =========================================================
# Run Flask App
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)