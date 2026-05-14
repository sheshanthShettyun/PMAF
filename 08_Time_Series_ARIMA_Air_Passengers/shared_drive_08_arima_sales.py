import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

data = {
    'Month': pd.date_range(start='2023-01-01', periods=24, freq='ME'),
    'Sales': [200,220,250,270,300,320,350,370,400,420,450,480,
              500,520,550,580,600,630,650,680,700,720,750,780]
}

df = pd.DataFrame(data)

df.set_index('Month', inplace=True)

plt.plot(df['Sales'])
plt.title("Sales Over Time")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

model = ARIMA(df['Sales'], order=(1,1,1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=6)

print("Forecasted Values:\n", forecast)

plt.plot(df['Sales'], label="Original")
plt.plot(forecast, label="Forecast", color='red')
plt.legend()
plt.show()

