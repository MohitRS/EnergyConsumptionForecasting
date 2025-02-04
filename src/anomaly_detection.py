import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_actual_and_forecast(actual_file: str, forecast_file: str):
    """
    Load the actual data and forecasted data from CSV files.
    """
    actual = pd.read_csv(actual_file, index_col='Datetime', parse_dates=True)['Global_active_power']
    forecast = pd.read_csv(forecast_file, index_col='Datetime', parse_dates=True)['Global_active_power']
    
    # Align the actual data with forecast index
    actual = actual[forecast.index]
    return actual, forecast

def detect_anomalies(actual: pd.Series, forecast: pd.Series, threshold: float = 3.0):
    """
    Identify anomalies where the absolute difference between actual and forecast
    exceeds a specified multiple of the residuals' standard deviation.
    """
    residual = actual - forecast
    resid_std = residual.std()
    # Flag points where the absolute residual is greater than threshold * std
    anomaly_mask = np.abs(residual) > threshold * resid_std
    anomalies = residual[anomaly_mask]
    return anomalies

def plot_anomalies(actual: pd.Series, forecast: pd.Series, anomalies: pd.Series):
    """
    Plot actual vs. forecasted values and highlight anomalies.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actual.index, actual, label='Actual', color='blue')
    plt.plot(forecast.index, forecast, label='Forecast', color='orange')
    plt.scatter(anomalies.index, actual[anomalies.index], color='red', label='Anomalies', marker='o')
    plt.xlabel('Datetime')
    plt.ylabel('Global Active Power')
    plt.title('Anomaly Detection via Residual Analysis')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # File paths for the actual processed data and the forecast data
    actual_file = "data/processed/household_power_consumption_processed.csv"
    forecast_file = "data/processed/arima_forecast.csv"
    
    # Load the actual and forecast data
    actual, forecast = load_actual_and_forecast(actual_file, forecast_file)
    
    # Detect anomalies based on residuals
    anomalies = detect_anomalies(actual, forecast)
    
    # Plot the anomalies
    plot_anomalies(actual, forecast, anomalies)
    
    print(f"Anomaly detection complete. Found {len(anomalies)} anomalies.")
