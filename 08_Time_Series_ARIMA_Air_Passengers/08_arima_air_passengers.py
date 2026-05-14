

import os
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ModuleNotFoundError:
    Prophet = None
    PROPHET_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

print(data.head())

data.columns = ['Month', 'Passengers']

data['Month'] = pd.to_datetime(data['Month'])

data.set_index('Month', inplace=True)

data.plot(figsize=(10,5))
plt.title("Airline Passengers Dataset")
plt.show()

model = ARIMA(data['Passengers'], order=(2,1,2))

model_fit = model.fit()

forecast = model_fit.forecast(steps=12)

print("\nForecasted Values:")
print(forecast)

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

train = data.iloc[:-6]
test = data.iloc[-6:]

sarima_model = SARIMAX(
    train["Passengers"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

sarima_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_fit.forecast(steps=len(test))
sarima_rmse = mean_squared_error(test["Passengers"], sarima_forecast) ** 0.5

print("\nSARIMA Forecasted Values:")
print(sarima_forecast)
print("\nSARIMA RMSE:", sarima_rmse)

plt.figure(figsize=(10, 5))
plt.plot(train.index, train["Passengers"], label="Training Data")
plt.plot(test.index, test["Passengers"], label="Actual Data")
plt.plot(test.index, sarima_forecast, label="SARIMA Forecast")
plt.legend()
plt.title("SARIMA Forecast")
plt.show()

if PROPHET_AVAILABLE:
    prophet_data = data.reset_index().rename(columns={"Month": "ds", "Passengers": "y"})
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)
    future = prophet_model.make_future_dataframe(periods=12, freq="MS")
    prophet_forecast = prophet_model.predict(future)
    print("\nProphet Forecast:")
    print(prophet_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12))
    prophet_model.plot(prophet_forecast)
    plt.title("Prophet Forecast")
    plt.show()
else:
    print("\nProphet is not installed. Install it with: python -m pip install prophet")
