# %%
from functools import lru_cache

import numpy as np
import pandas as pd

from py_online_forecast.core import DEFAULT_SOURCE
from py_online_forecast.prediction import ARX

from . import forecast_tools as _  # Register the fc accessor  # noqa: F401


@lru_cache(maxsize=1)
def _sim_sample():
    np.random.seed(42)

    date_range = pd.date_range(start="2130-01-01", end="2133-12-31 23:00:00", freq="h")

    # Simulate temperature data
    base_temperature = 15

    daily_temp_seasonality = 5 * np.sin(2 * np.pi * date_range.hour / 24)
    yearly_temp_seasonality = 10 * np.sin(2 * np.pi * date_range.dayofyear / 365)

    # Random noise
    temp_noise = np.random.normal(0, 1, len(date_range))

    # Total temperature
    temperature = (
        base_temperature + daily_temp_seasonality + yearly_temp_seasonality + temp_noise
    )

    # Simulate seasonal energy data
    # Base load
    base_load = 50

    # Simulate energy consumption data based on temperature
    # Energy consumption increases when temperature is lower (heating) or higher (cooling)
    energy_consumption = (
        base_load + 0.5 * (20 - temperature) + np.random.normal(0, 5, len(date_range))
    )

    # Create a DataFrame with 't' as the index
    sample_data = pd.DataFrame(
        {"energy": energy_consumption, "Taobs": temperature}, index=date_range
    ).fc.convert()

    # Simulate temperature forecasts
    forecast_horizons = list(range(1, 37))

    for horizon in forecast_horizons:
        sample_data["Ta", horizon] = sample_data["Taobs", 0].shift(
            -horizon
        ) + np.random.normal(0, 5, len(sample_data))

    # Smooth the forecasts using a rolling mean
    for horizon in forecast_horizons:
        sample_data["Ta", horizon] = (
            sample_data["Ta", horizon].rolling(window=10, min_periods=1).mean()
        )

    # Drop the rows with NaN values due to shifting
    sample_data.dropna(inplace=True)

    return sample_data


# %% Simulate hierarchical data

# %% Simulate AR(2) process for the bottom level of the hierarchy.
# Model is based on the study in,
# Møller, J.K., Nystrup, P., and Madsen, H. (2024).
# "Likelihood-based inference in temporal hierarchies."
# International Journal of Forecasting, 40, 515–531.


@lru_cache(maxsize=1)
def _sim_hierarchy_sample():

    # Parameters
    phi_1 = 0.75
    phi_2 = 0.2
    sigma = 1
    n_sim = 1000
    burn_in = 500

    n = n_sim + burn_in

    np.random.seed(1)
    w = np.random.normal(0, sigma, n)
    Y_H = np.zeros(n)
    for i in range(n):
        Y_H[i] = phi_1 * Y_H[i-1] + phi_2 * Y_H[i-2] + w[i]


    Y_A = Y_H[1:] + Y_H[:-1] # Y_{A,t} = Y_{H,t} + Y_{H,t-1}

    Y_H = Y_H[1:] # Align Y_{H,t} with Y_{A,t}

    # Model Y_H1 and Y_A as AR(1) processes
    mH = ARX(DEFAULT_SOURCE, 2, 1, init_K=0.01, track_memory=True)
    mA = ARX(DEFAULT_SOURCE, 2, 1, init_K=0.01, track_memory=True)

    pH = mH(Y_H, track_state=True, mem=1)["mean"]
    pA = mA(Y_A, track_state=True, mem=1)["mean"]

    # Stack Y_
    array = np.hstack([Y_A.reshape(-1, 1), Y_H.reshape(-1, 1), pA[:, [1]], pH])
    columns = pd.MultiIndex.from_tuples(
        [("Y_A", 0), ("Y_H", 0), ("pred_Y_A", 2), ("pred_Y_H", 1), ("pred_Y_H", 2)]
    )
    return (
        pd.DataFrame(array, columns=columns)
        .fc.convert()
        .iloc[-1000:]
        .reset_index(drop=True)
    )


def __getattr__(name):
    if name == "sample_data":
        return _sim_sample()
    elif name == "sample_hierarchical_data":
        return _sim_hierarchy_sample()
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


sample_data: pd.DataFrame
sample_hierarchical_data: pd.DataFrame
