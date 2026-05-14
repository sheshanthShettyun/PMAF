



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM
    TENSORFLOW_AVAILABLE = True
except ModuleNotFoundError:
    Sequential = load_model = Dense = LSTM = None
    TENSORFLOW_AVAILABLE = False

from flask import Flask, request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "lstm_model.h5")


csv_path = os.path.join(SCRIPT_DIR, "stock_data.csv")
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    print(f"WARNING: '{csv_path}' not found. Using sample stock price data.")
    close_prices = np.linspace(150.0, 250.0, 100)
    data = pd.DataFrame({'Close': close_prices})

print("First 5 Rows:")
print(data.head())


dataset = data['Close'].values

dataset = dataset.reshape(-1, 1)


scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(dataset)


X_train = []
y_train = []


for i in range(60, len(scaled_data)):

    X_train.append(scaled_data[i-60:i, 0])

    y_train.append(scaled_data[i, 0])


X_train = np.array(X_train)
y_train = np.array(y_train)

print("\nTraining Data Shape:")
print(X_train.shape)


if TENSORFLOW_AVAILABLE:
    X_train_model = np.reshape(
        X_train,
        (X_train.shape[0], X_train.shape[1], 1)
    )

    model = Sequential()

    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(X_train_model.shape[1], 1)
        )
    )

    model.add(LSTM(units=50))

    model.add(Dense(units=1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
else:
    print("\nTensorFlow is not installed. Using LinearRegression fallback.")
    X_train_model = X_train
    model = LinearRegression()

print("\nTraining Model...")

if TENSORFLOW_AVAILABLE:
    model.fit(
        X_train_model,
        y_train,
        epochs=3,
        batch_size=32,
        verbose=0
    )
    model.save(MODEL_PATH)
    print("\nTensorFlow model saved successfully!")
else:
    model.fit(X_train_model, y_train)

print("\nModel Ready Successfully!")


test_data = scaled_data[-60:]

X_test = np.array([test_data[:, 0]])

if TENSORFLOW_AVAILABLE:
    X_test_model = np.reshape(
        X_test,
        (X_test.shape[0], X_test.shape[1], 1)
    )
else:
    X_test_model = X_test

def predict_scaled_window(values):
    values = np.array(values, dtype=float).reshape(-1, 1)
    scaled_values = scaler.transform(values).reshape(1, 60)
    if TENSORFLOW_AVAILABLE:
        scaled_values = scaled_values.reshape(1, 60, 1)
        return model.predict(scaled_values, verbose=0)
    return model.predict(scaled_values).reshape(-1, 1)


if TENSORFLOW_AVAILABLE:
    predicted_price = model.predict(X_test_model, verbose=0)
else:
    predicted_price = model.predict(X_test_model).reshape(-1, 1)


predicted_price = scaler.inverse_transform(predicted_price)

print("\nPredicted Future Price:")
print(predicted_price)


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


app = Flask(__name__)

if TENSORFLOW_AVAILABLE and os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)


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


@app.route('/predict', methods=['POST'])

def predict():

    values = request.form['values']

    values = [float(x) for x in values.split(',')]

    prediction = predict_scaled_window(values)

    prediction = scaler.inverse_transform(prediction)

    return f'''
    <h3>Predicted Future Stock Price:</h3>

    <h2>{prediction[0][0]}</h2>
    '''


if __name__ == "__main__":

    if os.environ.get("RUN_FLASK") == "1":
        app.run(debug=True)
    else:
        print("\nFlask app is ready. Set RUN_FLASK=1 to start the web server.")
