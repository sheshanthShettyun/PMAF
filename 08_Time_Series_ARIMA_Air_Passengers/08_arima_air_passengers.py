# Serial mapped from current codebase: Time-Series Forecasting using ARIMA & Prophet.py
# Cleaned current-codebase serial file.

# Install libraries
# pip install pandas matplotlib statsmodels

# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
csv_path = os.path.join(SCRIPT_DIR, "AirPassengers.csv")
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
else:
    print(f"WARNING: '{csv_path}' not found. Using sample data for demonstration.")
    data = pd.DataFrame({
        'Month': pd.date_range(start='1949-01-01', periods=24, freq='ME'),
        'Passengers': [112,118,132,129,121,135,148,148,136,119,104,118,
                       115,126,141,135,125,149,170,170,158,133,114,140]
    })

# Display first rows
print(data.head())

# Rename columns
data.columns = ['Month', 'Passengers']

# Convert Month column to datetime
data['Month'] = pd.to_datetime(data['Month'])

# Set index
data.set_index('Month', inplace=True)

# Plot dataset
data.plot(figsize=(10,5))
plt.title("Airline Passengers Dataset")
plt.show()

# Build ARIMA model
model = ARIMA(data['Passengers'], order=(2,1,2))

# Train model
model_fit = model.fit()

# Forecast next 12 months
forecast = model_fit.forecast(steps=12)

print("\nForecasted Values:")
print(forecast)

# Plot forecast
plt.figure(figsize=(10,5))

plt.plot(data.index, data['Passengers'], label='Original Data')

future_dates = pd.date_range(
    start=data.index[-1],
    periods=12,
    freq='ME'
)

plt.plot(future_dates, forecast, label='Forecast')

plt.legend()
plt.title("ARIMA Forecast")

plt.show()
