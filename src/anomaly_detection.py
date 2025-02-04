# src/anomaly_detection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def load_actual_and_forecast(actual_file: str, forecast_file: str):
    """
    Load actual data and forecast data from CSV files.
    """
    actual = pd.read_csv(actual_file, index_col='Datetime', parse_dates=True)['Global_active_power']
    forecast = pd.read_csv(forecast_file, index_col='Datetime', parse_dates=True)['Global_active_power']
    
    # Align the actual data with forecast index
    actual = actual[forecast.index]
    return actual, forecast


def detect_anomalies_residual(actual: pd.Series, forecast: pd.Series, threshold: float = 3.0):
    """
    Detect anomalies where the residual (actual - forecast) exceeds a threshold multiplier.
    """
    residual = actual - forecast
    resid_std = residual.std()
    anomaly_mask = np.abs(residual) > threshold * resid_std
    anomalies = residual[anomaly_mask]
    return anomalies

def detect_anomalies_isolation_forest(actual: pd.Series, forecast: pd.Series, contamination: float = 0.01):
    """
    Detect anomalies using Isolation Forest on the residuals.
    """
    residual = (actual - forecast).to_frame(name='residual')
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(residual)
    anomaly_flags = iso_forest.predict(residual)
    anomalies = residual[anomaly_flags == -1]
    return anomalies

def plot_anomalies(actual: pd.Series, forecast: pd.Series, anomalies: pd.Series, method: str = "Residual"):
    """
    Plot actual and forecast data, highlighting detected anomalies.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actual.index, actual, label='Actual', color='blue')
    plt.plot(forecast.index, forecast, label='Forecast', color='orange')
    plt.scatter(anomalies.index, actual[anomalies.index], color='red', label=f'Anomalies ({method})', marker='o')
    plt.xlabel('Datetime')
    plt.ylabel('Global Active Power')
    plt.title(f'Anomaly Detection using {method} Method')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    actual_file = "data/processed/household_power_consumption_enhanced.csv"
    forecast_file = "data/processed/arima_forecast.csv"
    actual, forecast = load_actual_and_forecast(actual_file, forecast_file)
    
    anomalies_res = detect_anomalies_residual(actual, forecast, threshold=3.0)
    anomalies_if = detect_anomalies_isolation_forest(actual, forecast, contamination=0.01)
    
    plot_anomalies(actual, forecast, anomalies_res, method="Residual")
    plot_anomalies(actual, forecast, anomalies_if, method="Isolation Forest")
    
    print(f"Residual method found {len(anomalies_res)} anomalies.")
    print(f"Isolation Forest found {len(anomalies_if)} anomalies.")
