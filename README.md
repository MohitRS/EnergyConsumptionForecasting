# Energy Consumption Forecast & Anomaly Detection Dashboard

## Overview

This project provides an interactive dashboard built with Streamlit for exploring and forecasting energy consumption data. It combines historical data visualization, forecasting, anomaly detection, rolling averages, and seasonal decomposition to offer comprehensive insights into energy consumption trends. The dashboard helps users identify patterns, detect anomalies, and understand underlying trends in energy usage.

## Features

- **Historical & Forecasted Data Visualization:**  
  Compare historical energy consumption with forecasted values side-by-side.

- **Anomaly Detection:**  
  Detect unusual deviations between actual and forecasted data using a configurable threshold multiplier.

- **Rolling Average Analysis:**  
  Smooth the time series data with a rolling average to highlight long-term trends over various time windows.

- **Seasonal Decomposition:**  
  Decompose the time series into trend, seasonality, and residual components using both additive and multiplicative models.

- **Interactive Controls:**  
  Use the sidebar to select analysis types, adjust date ranges, and configure parameters like anomaly thresholds and rolling window sizes.

- **Footer Descriptions:**  
  Each analysis section includes a footer that provides a summary of the insights observed and explains the significance of the displayed data.

## Project Structure

```
/project-root
├── app.py                         # Main Streamlit application
├── data/
│   ├── processed/
│   │   ├── household_power_consumption_enhanced.csv   # Historical data
│   │   └── arima_forecast.csv                         # Forecast data generated using ARIMA
├── README.md                      # Project documentation (this file)
└── requirements.txt               # Python dependencies list
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Dependencies

Install the required dependencies using pip. For example, if you have a `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```


## Usage

To run the dashboard locally, navigate to your project directory and execute:

```bash
streamlit run app.py
```

Once the app is running, your default web browser will open the dashboard. Use the sidebar to navigate through the different analysis sections and adjust parameters such as date ranges, anomaly thresholds, and rolling window sizes.

## Data Description

The dashboard uses two main CSV files located in the `data/processed` directory:

- **household_power_consumption_enhanced.csv:**  
  Contains historical energy consumption data with a datetime index (parsed from a 'Datetime' column) and recorded at an hourly frequency.

- **arima_forecast.csv:**  
  Contains forecasted energy consumption data generated using an ARIMA model. This file is used to compare against historical data for anomaly detection.

## Future Enhancements

- **Machine Learning-Based Forecasting:**  
  Explore and integrate ML-based models for improved forecasting accuracy.

- **Dynamic Data Updates:**  
  Implement real-time or near-real-time data ingestion to keep the dashboard up-to-date.

- **Energy Consumption Recommendations:**  
  Offer actionable recommendations based on detected anomalies and consumption patterns.

- **Enhanced Visualizations:**  
  Further enhance visual interactivity and provide additional insights through more advanced charts.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch with your changes:  
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes with a descriptive message.
4. Push your branch and open a pull request.

For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgements

- Built with [Streamlit](https://streamlit.io/)
- Data handling using [pandas](https://pandas.pydata.org/)
- Visualization powered by [matplotlib](https://matplotlib.org/)
- Time series decomposition with [statsmodels](https://www.statsmodels.org/)
- Thanks to the community for valuable feedback and contributions.
```

This README file outlines the project's purpose, features, installation steps, usage instructions, and potential future enhancements, offering clear guidance for both users and contributors.
