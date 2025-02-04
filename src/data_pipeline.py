# src/data_pipeline.py
import os
import pandas as pd
from data_preprocessing import download_dataset, load_and_preprocess_data
from feature_engineering import add_lag_features, add_rolling_features, add_fourier_features
from forecasting_model import load_processed_data, train_arima_model, grid_search_arima
from anomaly_detection import load_actual_and_forecast, detect_anomalies_residual

def run_pipeline():
    # Directories
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Download and preprocess data
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.txt"
    dataset_path = os.path.join(raw_dir, "household_power_consumption.txt")
    download_dataset(dataset_url, dataset_path)
    df_preprocessed = load_and_preprocess_data(dataset_path)
    
    # Save the initial processed data
    processed_file = os.path.join(processed_dir, "household_power_consumption_processed.csv")
    df_preprocessed.to_csv(processed_file)
    
    # Advanced feature engineering
    df = pd.read_csv(processed_file, index_col="Datetime", parse_dates=True)
    df = add_lag_features(df, "Global_active_power")
    df = add_rolling_features(df, "Global_active_power")
    df = add_fourier_features(df, period=24, order=3)
    enhanced_file = os.path.join(processed_dir, "household_power_consumption_enhanced.csv")
    df.to_csv(enhanced_file)
    
    # Forecasting using enhanced data
    series = df["Global_active_power"]
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    best_order = grid_search_arima(train, p_values=[0, 1, 2], d_values=[0, 1], q_values=[0, 1, 2])
    model_fit = train_arima_model(train, order=best_order)
    forecast = model_fit.forecast(steps=len(test))
    forecast_df = forecast.to_frame(name='Global_active_power')
    forecast_file = os.path.join(processed_dir, "arima_forecast.csv")
    forecast_df.to_csv(forecast_file)
    
    # Anomaly detection using the residual method
    actual, forecast_loaded = load_actual_and_forecast(enhanced_file, forecast_file)
    anomalies = detect_anomalies_residual(actual, forecast_loaded, threshold=3.0)
    print(f"Pipeline detected {len(anomalies)} anomalies using the residual method.")

if __name__ == "__main__":
    run_pipeline()
    print("Data pipeline execution complete.")
