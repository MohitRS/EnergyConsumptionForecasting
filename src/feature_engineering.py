# src/feature_engineering.py
import pandas as pd
import numpy as np

def add_lag_features(df: pd.DataFrame, column: str, lags: list = [1, 2, 3, 24]):
    """
    Create lag features for a given column.
    :param df: DataFrame with datetime index.
    :param column: Column for which to create lag features.
    :param lags: List of lag intervals (e.g., hours).
    """
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, column: str, windows: list = [3, 6, 24]):
    """
    Create rolling mean features for a given column.
    :param df: DataFrame with datetime index.
    :param column: Column for which to create rolling features.
    :param windows: List of window sizes (in number of periods).
    """
    for window in windows:
        df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
    return df

def add_fourier_features(df: pd.DataFrame, period: int = 24, order: int = 3):
    """
    Add Fourier series features to capture seasonality.
    :param df: DataFrame with datetime index.
    :param period: The period of the seasonality (e.g., 24 for hourly data).
    :param order: Number of sine/cosine pairs to add.
    """
    t = np.arange(len(df))
    for i in range(1, order + 1):
        df[f'sin_{i}'] = np.sin(2 * np.pi * i * t / period)
        df[f'cos_{i}'] = np.cos(2 * np.pi * i * t / period)
    return df

if __name__ == "__main__":
    # Example usage: load processed data, add features, and save the updated file.
    processed_path = "data/processed/household_power_consumption_processed.csv"
    df = pd.read_csv(processed_path, index_col="Datetime", parse_dates=True)
    
    df = add_lag_features(df, "Global_active_power")
    df = add_rolling_features(df, "Global_active_power")
    df = add_fourier_features(df, period=24, order=3)
    
    enhanced_path = "data/processed/household_power_consumption_enhanced.csv"
    df.to_csv(enhanced_path)
    print(f"Enhanced features saved to {enhanced_path}")
