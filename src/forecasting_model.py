import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def load_processed_data(file_path: str):
    """
    Load the processed CSV data with a datetime index.
    """
    return pd.read_csv(file_path, index_col='Datetime', parse_dates=True)

def train_arima_model(series: pd.Series, order=(1, 1, 1)):
    """
    Fit an ARIMA model to the given time series data.
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def plot_forecast(train, test, forecast):
    """
    Plot the training data, test data, and forecasted values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Test Data')
    plt.plot(test.index, forecast, label='Forecast')
    plt.xlabel('Datetime')
    plt.ylabel('Global Active Power')
    plt.title('ARIMA Forecasting of Energy Consumption')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the processed dataset
    processed_file = "data/processed/household_power_consumption_processed.csv"
    df = load_processed_data(processed_file)
    
    # Use 'Global_active_power' as the target for forecasting
    series = df['Global_active_power']
    
    # Split the data: 80% for training and 20% for testing
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # Train the ARIMA model
    model_fit = train_arima_model(train)
    
    # Forecast for the duration of the test set
    forecast = model_fit.forecast(steps=len(test))
    
    # Plot the results
    plot_forecast(train, test, forecast)
    
    # Save the forecasted data for anomaly detection later
    forecast_df = forecast.to_frame(name='Global_active_power')
    forecast_df.to_csv("data/processed/arima_forecast.csv")
    
    print("Forecasting complete. Forecast data saved to data/processed/arima_forecast.csv")
