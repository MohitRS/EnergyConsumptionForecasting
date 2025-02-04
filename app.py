import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Cache the data loading for performance
@st.cache_data
def load_data(path):
    # Load CSV file, parse dates, and set 'Datetime' as index
    try:
        df = pd.read_csv(path, index_col='Datetime', parse_dates=True)
        return df.asfreq('H')
    except Exception as e:
        st.error(f"Error loading data from {path}: {e}")
        return pd.DataFrame()

def plot_time_series(data, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data.iloc[:, 0], label=data.columns[0])
    ax.set_title(title)
    ax.set_xlabel('Datetime')
    ax.set_ylabel(ylabel)
    ax.legend()
    st.pyplot(fig)

def detect_anomalies(actual, forecast, threshold_multiplier):
    residual = actual - forecast
    std_resid = residual.std()
    anomaly_mask = np.abs(residual) > threshold_multiplier * std_resid
    anomalies = actual[anomaly_mask]
    return anomalies

def plot_seasonal_decomposition(series, model='additive'):
    decomposition = sm.tsa.seasonal_decompose(series, model=model, period=24)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    decomposition.observed.plot(ax=axes[0], title="Observed")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonal")
    decomposition.resid.plot(ax=axes[3], title="Residual")

    for ax in axes:
        ax.set_xlabel("Datetime")
    plt.tight_layout()
    st.pyplot(fig)

def display_footer(description: str):
    """Display a footer with a short description summarizing insights from the page."""
    st.markdown("---")
    st.markdown("### Observations & Insights")
    st.write(description)

# App Title and Description
st.title("Energy Consumption Forecast & Anomaly Detection Dashboard")
st.markdown("""
This dashboard allows users to explore energy consumption data through various analyses:
- **Historical & Forecasted Data:** View trends and predictions.
- **Anomaly Detection:** Identify periods with unusual consumption.
- **Rolling Averages:** Smooth out short-term fluctuations.
- **Seasonal Decomposition:** Separate time series into trend, seasonality, and noise.
""")

# File paths for data
processed_file = "data/processed/household_power_consumption_enhanced.csv"
forecast_file  = "data/processed/arima_forecast.csv"

# Load the datasets
data = load_data(processed_file)
forecast = load_data(forecast_file)

# Sidebar Navigation and Filters
st.sidebar.header("Navigation & Filters")
analysis_type = st.sidebar.radio("Select Analysis Type", 
                                 ["Historical & Forecast", "Anomaly Detection", "Rolling Average", "Seasonal Decomposition"])

if not data.empty:
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date   = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
else:
    st.sidebar.error("Historical data is not available.")
    start_date = end_date = None

filtered_data = data.loc[str(start_date):str(end_date)] if start_date and end_date else data
filtered_forecast = forecast.loc[str(start_date):str(end_date)] if start_date and end_date else forecast

# Analysis Sections
if analysis_type == "Historical & Forecast":
    st.header("Historical & Forecasted Energy Consumption")
    
    st.subheader("Historical Energy Consumption")
    if not filtered_data.empty:
        plot_time_series(filtered_data[['Global_active_power']], "Historical Global Active Power", "Global Active Power")
    else:
        st.write("No historical data available.")

    st.subheader("Forecasted Energy Consumption")
    if not filtered_forecast.empty:
        plot_time_series(filtered_forecast, "Forecasted Global Active Power", "Global Active Power")
    else:
        st.write("No forecast data available.")
    
    description = (
        "This section compares historical data with forecasted energy consumption. "
        "By observing the trends, you can assess how well the forecast aligns with past behavior and identify potential shifts in consumption patterns."
    )
    display_footer(description)

elif analysis_type == "Anomaly Detection":
    st.header("Anomaly Detection")
    anomaly_threshold = st.sidebar.slider("Anomaly Threshold (Multiplier)", 2.0, 5.0, 3.0)
    st.write(f"Using anomaly threshold multiplier: {anomaly_threshold}")

    common_index = filtered_data.index.intersection(filtered_forecast.index)
    if not common_index.empty:
        actual_aligned = filtered_data.loc[common_index, 'Global_active_power']
        forecast_aligned = filtered_forecast.loc[common_index, 'Global_active_power']
        anomalies = detect_anomalies(actual_aligned, forecast_aligned, anomaly_threshold)
        st.write(f"Number of anomalies detected: {len(anomalies)}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(common_index, actual_aligned, label="Actual", color='blue')
        ax.plot(common_index, forecast_aligned, label="Forecast", linestyle="--", color='green')
        if not anomalies.empty:
            ax.scatter(anomalies.index, anomalies, color="red", label="Anomalies", zorder=5)
        ax.set_title("Anomaly Detection: Actual vs Forecast")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Global Active Power")
        ax.legend()
        st.pyplot(fig)
        
        description = (
            "Anomalies are highlighted as red points where the actual energy consumption deviates significantly "
            "from the forecast. This analysis helps pinpoint unusual events or potential data issues that may require further investigation."
        )
    else:
        st.write("No overlapping data points for anomaly detection.")
        description = (
            "The anomaly detection section requires overlapping historical and forecast data. "
            "Please adjust the date range or ensure that the data is properly loaded."
        )
    display_footer(description)

elif analysis_type == "Rolling Average":
    st.header("Rolling Average Analysis")
    rolling_window = st.sidebar.slider("Select Rolling Window (Hours)", 3, 48, 24)
    if not filtered_data.empty:
        rolling_avg = filtered_data['Global_active_power'].rolling(window=rolling_window).mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(filtered_data.index, filtered_data['Global_active_power'], label="Original Data", alpha=0.5)
        ax.plot(filtered_data.index, rolling_avg, label=f"Rolling Average ({rolling_window} hrs)", color="red")
        ax.set_title(f"Rolling Average with {rolling_window}-hour Window")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Global Active Power")
        ax.legend()
        st.pyplot(fig)
        description = (
            "The rolling average smooths short-term fluctuations and emphasizes longer-term trends in energy consumption. "
            "This view helps you better understand underlying patterns, reducing the noise present in the raw data."
        )
    else:
        st.write("No data available for rolling average analysis.")
        description = "No data was available to compute the rolling average."
    display_footer(description)

elif analysis_type == "Seasonal Decomposition":
    st.header("Seasonal Decomposition Analysis")
    model_type = st.sidebar.radio("Select Decomposition Model", ["Additive", "Multiplicative"])
    if not filtered_data.empty:
        plot_seasonal_decomposition(filtered_data['Global_active_power'].dropna(), model=model_type.lower())
        description = (
            "Seasonal decomposition splits the time series into its constituent components: trend, seasonality, and residuals. "
            "This analysis reveals underlying patterns and irregularities, aiding in understanding the dynamics of energy consumption."
        )
    else:
        st.write("No data available for seasonal decomposition.")
        description = "No data was available for seasonal decomposition analysis."
    display_footer(description)

