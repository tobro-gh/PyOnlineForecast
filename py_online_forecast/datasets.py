#%%
from .core import *
import numpy as np
import pandas as pd
#import os
import matplotlib.pyplot as plt
np.random.seed(42)

date_range = pd.date_range(start='2130-01-01', end='2133-12-31 23:00:00', freq='h')


# Simulate temperature data
base_temperature = 15

daily_temp_seasonality = 5 * np.sin(2 * np.pi * date_range.hour / 24)
yearly_temp_seasonality = 10 * np.sin(2 * np.pi * date_range.dayofyear / 365)

# Random noise
temp_noise = np.random.normal(0, 1, len(date_range))

# Total temperature
temperature = base_temperature + daily_temp_seasonality + yearly_temp_seasonality + temp_noise

# Simulate seasonal energy data
# Base load
base_load = 50

# Simulate energy consumption data based on temperature
# Energy consumption increases when temperature is lower (heating) or higher (cooling)
energy_consumption = base_load + 0.5 * (20 - temperature) + np.random.normal(0, 5, len(date_range))

# Create a DataFrame with 't' as the index
sample_data = pd.DataFrame({'energy': energy_consumption, 'Taobs': temperature}, index=date_range).fc.convert()

#%%
# Simulate temperature forecasts
forecast_horizons = list(range(1, 37))

for horizon in forecast_horizons:
    sample_data["Ta", horizon] = sample_data['Taobs', 0].shift(-horizon) + np.random.normal(0, 5, len(sample_data))

# Smooth the forecasts using a rolling mean
for horizon in forecast_horizons:
    sample_data["Ta", horizon] = sample_data["Ta", horizon].rolling(window=10, min_periods=1).mean()

# Drop the rows with NaN values due to shifting
sample_data.dropna(inplace=True)

#%%
#
# Save the data as CSV
#output_dir = os.path.join(os.path.dirname(__file__), '../py_online_forecast/data')
#os.makedirs(output_dir, exist_ok=True)
#sample_data.to_csv(os.path.join(output_dir, 'simulated_data.csv'), index=False)

# Set the pair of dates for plotting
if __name__ == "__main__":
    start_date = '2130-01-01'
    end_date = '2130-01-07'

    # Subset of horizons to plot
    subset_horizons = [24]

    # Filter data for the specified date range
    plot_data = sample_data[(sample_data['t'] >= start_date) & (sample_data['t'] <= end_date)]

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
# %% Simulate hierarchical data
#%%
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import py_online_forecast.core as c

#%% Simulate AR(2) process in line with Møller et al., 2024,

phi_1 = 0.75
phi_2 = 0.2
sigma = 1
n_train = 1000
n_test = 1000

#%%
np.random.seed(1)

def sim(n_samples):
    w = np.random.normal(0, sigma, n_samples + 2)
    Y_H = np.zeros(n_samples + 3)
    for i in range(n_samples + 1):
        Y_H[i+2] = phi_1 * Y_H[i+1] + phi_2 * Y_H[i] + w[i]

    Y_H1 = Y_H[2:-1]
    Y_H2 = Y_H[3:]

    Y = np.array([Y_H1, Y_H2]).T
    return Y

Y_train = sim(n_train)
Y_test = sim(n_test)


#%%
mH = ARIMA(Y_train[:,0], order=(1,0,0), trend='n').fit()
mA = ARIMA(Y_train[::2].sum(axis = 1), order=(1,0,0), trend='n').fit()
#%%
phiA = mA.arparams[0]
phiH = mH.arparams[0]

# Make forecasts
pred_mat = np.array([[phiA, phiA], [0, phiH], [0, phiH**2]])

predictions = Y_test @ pred_mat.T

#%% Export bottom level observations and all predictions to csv
columns = pd.MultiIndex.from_tuples([("Y_A", 0), ("Y_H", 0), ("pred_Y_A", 2), ("pred_Y_H", 1), ("pred_Y_H", 2)])
Y_test_top = Y_test.sum(axis=1).reshape(-1,1)
fc_data = pd.DataFrame(np.hstack([Y_test_top, Y_test[:, [1]], predictions]), columns=columns).fc.convert()

#%%
lagged_predictions = fc_data.fc.lag()

# Compute residuals
resid_bot = lagged_predictions[["pred_Y_H"]].sub(Y_test[:, 1], axis = 0)
resid_top = lagged_predictions[["pred_Y_A"]].sub(Y_test[:, 0], axis = 0)

#%%
if __name__ == "__main__":

    resid_all = pd.concat([resid_bot, resid_top], axis=1)
    rmse = resid_all.apply(c.rmse)

    # Plot the sampled data vs the 1-step predictions
    plt.figure(figsize=(12, 6))

    fig, axes = plt.subplots(nrows=2, figsize=(15, 10))

    # First plot for the bottom level
    lagged_predictions[["Y_H", "pred_Y_H"]].fc.lag().plot(ax=axes[0])

    axes[0].set_title(f'Bottom Level Forecasts RMSE: {rmse["pred_Y_H"]}')
    axes[0].legend()

    # Second plot for the top level
    lagged_predictions[["Y_A", "pred_Y_A"]].fc.lag().plot(ax=axes[1])

    axes[1].set_title(f'Top Level Forecasts RMSE: {rmse["pred_Y_A"]}')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# %%
#fc_data.to_csv("../py_online_forecast/data/hierarchical_data.csv", index = False)
sample_hierarchical_data = fc_data

# %%

