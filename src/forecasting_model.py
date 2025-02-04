# src/forecasting_model.py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import itertools

def load_processed_data(file_path: str):
    """
    Load the processed CSV data with a datetime index and enforce an hourly frequency.
    """
    df = pd.read_csv(file_path, index_col='Datetime', parse_dates=True)
    df = df.asfreq('H')  # Reassign frequency since CSV does not store freq metadata
    return df


def grid_search_arima(series: pd.Series, p_values: list, d_values: list, q_values: list):
    """
    Perform a grid search to find the best ARIMA parameters based on AIC.
    """
    best_score, best_cfg = float("inf"), None
    for p, d, q in itertools.product(p_values, d_values, q_values):
        order = (p, d, q)
        try:
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_score:
                best_score, best_cfg = aic, order
        except Exception:
            continue
    print(f'Best ARIMA order: {best_cfg} with AIC {best_score}')
    return best_cfg

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
    plt.title('ARIMA Forecasting')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    processed_file = "data/processed/household_power_consumption_enhanced.csv"
    df = load_processed_data(processed_file)
    series = df['Global_active_power']
    
    # Split data: 80% training, 20% testing
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    best_order = grid_search_arima(train, p_values=[0, 1, 2], d_values=[0, 1], q_values=[0, 1, 2])
    model_fit = train_arima_model(train, order=best_order)
    forecast = model_fit.forecast(steps=len(test))
    
    plot_forecast(train, test, forecast)
    
    forecast_df = forecast.to_frame(name='Global_active_power')
    forecast_df.index.name = 'Datetime'  # Ensure the index is named
    # Save with index_label to include the index header in the CSV file
    forecast_df.to_csv("data/processed/arima_forecast.csv", index_label='Datetime')
    print("Forecasting complete and saved.")

