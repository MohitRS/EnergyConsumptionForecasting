import os
import pandas as pd
import requests

def download_dataset(url: str, dest: str):
    """
    Download the dataset from the specified URL if it does not already exist.
    """
    if not os.path.exists(dest):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(dest, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Dataset already exists.")

def load_and_preprocess_data(file_path: str):
    """
    Load the dataset, parse datetime columns, handle missing values,
    and perform resampling and feature engineering.
    """
    print("Loading dataset...")
    # Load the dataset; note that missing values are denoted by '?' in this dataset
    df = pd.read_csv(file_path, sep=';', na_values='?', low_memory=False)
    
    print("Parsing datetime...")
    # Merge 'Date' and 'Time' into a single datetime column
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    
    print("Handling missing values...")
    # Forward-fill missing values; other strategies can be used based on your analysis
    df.fillna(method='ffill', inplace=True)
    
    print("Resampling data to hourly frequency...")
    # Resample the data to hourly intervals; this can be adjusted (e.g., to daily) if needed
    df_hourly = df.resample('H').mean()
    df_hourly.index.freq = 'H'  # Explicitly set the frequency
    
    print("Creating additional features...")
    # Create some basic temporal features for further modeling
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['day_of_week'] = df_hourly.index.dayofweek

    return df_hourly

if __name__ == "__main__":
    # Define directories for raw and processed data
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # URL for the UCI dataset
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.txt"
    dataset_path = os.path.join(raw_data_dir, "household_power_consumption.txt")
    
    # Download the dataset if necessary
    download_dataset(dataset_url, dataset_path)
    
    # Load and preprocess the dataset
    df_preprocessed = load_and_preprocess_data(dataset_path)
    
    # Save the processed data for future use
    processed_file_path = os.path.join(processed_data_dir, "household_power_consumption_processed.csv")
    df_preprocessed.to_csv(processed_file_path)
    
    print("Data preprocessing complete. Processed data saved to:", processed_file_path)
