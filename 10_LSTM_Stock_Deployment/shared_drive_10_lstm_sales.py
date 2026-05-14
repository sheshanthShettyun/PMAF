#pip install numpy pandas matplotlib scikit-learn tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except ModuleNotFoundError:
    Sequential = LSTM = Dense = None
    TENSORFLOW_AVAILABLE = False

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Create dataset
data = [200,220,250,270,300,320,350,370,400,420,450,480,
        500,520,550,580,600,630,650,680,700,720,750,780]

df = pd.DataFrame(data, columns=['Sales'])

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

# Create sequences (window size = 3)
X, y = [], []

for i in range(len(scaled_data) - 3):
    X.append(scaled_data[i:i+3])
    y.append(scaled_data[i+3])

X = np.array(X)
y = np.array(y)

# Model
if TENSORFLOW_AVAILABLE:
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(X, y, epochs=50, verbose=0)

    # Predict
    predictions = model.predict(X)
else:
    print("TensorFlow is not installed. Using LinearRegression fallback.")
    model = LinearRegression()
    X_flat = X.reshape(X.shape[0], X.shape[1])
    model.fit(X_flat, y.ravel())
    predictions = model.predict(X_flat).reshape(-1, 1)

predictions = scaler.inverse_transform(predictions)

# Plot
plt.plot(df.values, label='Original')
plt.plot(range(3, len(predictions)+3), predictions, label='Predicted')
plt.legend()
plt.show()
