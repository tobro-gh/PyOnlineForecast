#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
np.random.seed(42)

date_range = pd.date_range(start='2130-01-01', end='2133-12-31 23:00:00', freq='H')


# Simulate temperature data
base_temperature = 15

daily_temp_seasonality = 5 * np.sin(2 * np.pi * date_range.hour / 24)
yearly_temp_seasonality = 10 * np.sin(2 * np.pi * date_range.dayofyear / 365)

# Random noise
temp_noise = np.random.normal(0, 2, len(date_range))

# Total temperature
temperature = base_temperature + daily_temp_seasonality + yearly_temp_seasonality + temp_noise

# Simulate seasonal energy data
# Base load
base_load = 50

# Simulate energy consumption data based on temperature
# Energy consumption increases when temperature is lower (heating) or higher (cooling)
energy_consumption = base_load + 0.5 * (20 - temperature) + np.random.normal(0, 5, len(date_range))

# Create a DataFrame
data = pd.DataFrame({'t': date_range, 'energy': energy_consumption, 'Taobs': temperature})

#%%
# Simulate temperature forecasts
forecast_horizons = list(range(1, 37))

for horizon in forecast_horizons:
    forecast_column = f'Ta.k{horizon}'
    data[forecast_column] = data['Taobs'].shift(-horizon) + np.random.normal(0, 5, len(data))

# Smooth the forecasts using a rolling mean
for horizon in forecast_horizons:
    forecast_column = f'Ta.k{horizon}'
    data[forecast_column] = data[forecast_column].rolling(window=10, min_periods=1).mean()

# Drop the rows with NaN values due to shifting
data.dropna(inplace=True)

#%%
#
# Save the data as CSV
output_dir = os.path.join(os.path.dirname(__file__), '../py_online_forecast/data')
os.makedirs(output_dir, exist_ok=True)
data.to_csv(os.path.join(output_dir, 'simulated_data.csv'), index=False)

# Set the pair of dates for plotting
if __name__ == "__main__":
    start_date = '2130-01-01'
    end_date = '2130-01-07'

    # Subset of horizons to plot
    subset_horizons = [24]

    # Filter data for the specified date range
    plot_data = data[(data['t'] >= start_date) & (data['t'] <= end_date)]

    # Plot the actual temperature and forecasts
    plt.figure(figsize=(14, 8))
    plt.plot(plot_data['t'], plot_data['Taobs'], label='Actual Temperature', color='black')

    for horizon in subset_horizons:
        forecast_column = f'Ta.k{horizon}'
        shifted_plot_data = plot_data.copy()
        shifted_plot_data['t'] = shifted_plot_data['t'] + pd.to_timedelta(horizon, unit='h')
        plt.plot(shifted_plot_data['t'], shifted_plot_data[forecast_column], label=f'Forecast {horizon}h')

    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Temperature and Forecasts')
    plt.legend()
    plt.grid(True)
    plt.show()
# %%
