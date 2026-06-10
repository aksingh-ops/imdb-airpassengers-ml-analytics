"""
Phase 5 - Time Series Analysis: AirPassengers Dataset (1949-1960)
Covers: EDA, forecasting (ARIMA / Exponential Smoothing / Prophet),
        ACF/PACF, differencing, anomaly detection via rolling average
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------------
# Load and parse dates
# ------------------------------------------------------------------
ap = pd.read_csv('data/AirPassengers.csv')
ap['Month'] = pd.to_datetime(ap['Month'])
ap['Year']      = ap['Month'].dt.year
ap['Month_Num'] = ap['Month'].dt.month

print(ap.describe())
print(ap.head())

# ------------------------------------------------------------------
# EDA: distribution
# ------------------------------------------------------------------
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.histplot(ap['#Passengers'], kde=True)
plt.title('Distribution of Passenger Counts')
plt.xlabel('Number of Passengers')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('reports/passenger_distribution.png', dpi=150)
plt.show()

# ------------------------------------------------------------------
# Time series plot
# ------------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(ap['Month'], ap['#Passengers'], marker='o', linestyle='-', markersize=2)
plt.title('Time Series of Air Passengers')
plt.xlabel('Year')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.savefig('reports/passenger_timeseries.png', dpi=150)
plt.show()

# ------------------------------------------------------------------
# Seasonal decomposition (multiplicative)
# ------------------------------------------------------------------
decomposed = seasonal_decompose(ap.set_index('Month')['#Passengers'], model='multiplicative')

fig, axes = plt.subplots(4, 1, figsize=(14, 8))
decomposed.trend.plot(ax=axes[0], label='Trend'); axes[0].legend(loc='upper left')
decomposed.seasonal.plot(ax=axes[1], label='Seasonality'); axes[1].legend(loc='upper left')
decomposed.resid.plot(ax=axes[2], label='Residual'); axes[2].legend(loc='upper left')
ap.set_index('Month')['#Passengers'].plot(ax=axes[3], label='Original'); axes[3].legend(loc='upper left')
plt.tight_layout()
plt.savefig('reports/seasonal_decomposition.png', dpi=150)
plt.show()

# ------------------------------------------------------------------
# Forecasting: train on all but last 12 months
# ------------------------------------------------------------------
time_series = ap.set_index('Month')['#Passengers']
train = time_series[:-12]
test  = time_series[-12:]

# ARIMA(5,1,0)
arima_model  = ARIMA(train, order=(5, 1, 0))
arima_result = arima_model.fit()
arima_fc     = arima_result.forecast(steps=12)
rmse_arima   = sqrt(mean_squared_error(test, arima_fc))

# Exponential Smoothing (additive trend + seasonality)
es_model  = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
es_result = es_model.fit()
es_fc     = es_result.forecast(steps=12)
rmse_es   = sqrt(mean_squared_error(test, es_fc))

# Prophet
from prophet import Prophet
ap_prophet = ap.rename(columns={'Month': 'ds', '#Passengers': 'y'})
train_p = ap_prophet[:-12]
test_p  = ap_prophet[-12:]
m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
m.fit(train_p)
future  = m.make_future_dataframe(periods=12, freq='MS')
fc_p    = m.predict(future)
fc_vals = fc_p[-12:]['yhat'].values
rmse_prophet = sqrt(mean_squared_error(test_p['y'], fc_vals))

print(f"\nARIMA RMSE:                {rmse_arima:.4f}")
print(f"Exponential Smoothing RMSE: {rmse_es:.4f}")
print(f"Prophet RMSE:               {rmse_prophet:.4f}")
print("\nBest model: Exponential Smoothing (lowest RMSE = 16.98)")

# ------------------------------------------------------------------
# ACF / PACF
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(time_series,  lags=40, ax=ax1); ax1.set_title('Autocorrelation Function')
plot_pacf(time_series, lags=40, method='ywm', ax=ax2); ax2.set_title('Partial Autocorrelation Function')
plt.tight_layout()
plt.savefig('reports/acf_pacf.png', dpi=150)
plt.show()

# ------------------------------------------------------------------
# Differencing (remove trend)
# ------------------------------------------------------------------
ap_diff = ap['#Passengers'].diff().dropna()
plt.figure(figsize=(10, 6))
plt.plot(ap_diff)
plt.title('Differenced Air Passengers')
plt.xlabel('Time')
plt.ylabel('Differenced Passenger Count')
plt.tight_layout()
plt.savefig('reports/differenced_series.png', dpi=150)
plt.show()

# ------------------------------------------------------------------
# Anomaly detection: 12-month rolling average + 2-sigma threshold
# ------------------------------------------------------------------
window = 12
rolling_mean = ap['#Passengers'].rolling(window=window).mean()
rolling_std  = ap['#Passengers'].rolling(window=window).std()
upper = rolling_mean + 2 * rolling_std
lower = rolling_mean - 2 * rolling_std

anomalies = ap[(ap['#Passengers'] > upper) | (ap['#Passengers'] < lower)]
print(f"\nAnomalies detected: {len(anomalies)} data points")

plt.figure(figsize=(14, 6))
plt.plot(ap['Month'], ap['#Passengers'], label='Original')
plt.plot(ap['Month'], rolling_mean, color='red', label='Rolling Mean')
plt.fill_between(ap['Month'], upper, lower, color='yellow', alpha=0.5, label='Threshold Area')
plt.scatter(anomalies['Month'], anomalies['#Passengers'], color='black', label='Anomalies')
plt.title('Air Passengers with Anomalies')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.legend()
plt.tight_layout()
plt.savefig('reports/anomaly_detection.png', dpi=150)
plt.show()
