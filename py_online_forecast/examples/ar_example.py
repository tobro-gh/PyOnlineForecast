#%%
from py_online_forecast.prediction import rmse
from py_online_forecast.datasets import sample_data, sample_hierarchical_data

import matplotlib.pyplot as plt

#%%
sample_hierarchical_data.fc.subset("Y_H","pred_Y_H").iloc[:20].plot()

#%% Plot versus previous value
pred_Y = sample_hierarchical_data[("pred_Y_A", 2)]
Y = sample_hierarchical_data[("Y_A", 0)]
plt.figure(figsize=(10, 5))
plt.scatter(Y, pred_Y, alpha=0.5)



#%%
pred_H = sample_hierarchical_data.fc.subset("pred_Y_H", horizons = (1,)).fc.lag()
naive = sample_hierarchical_data.fc.subset("Y_H", horizons = (0,)).shift(1)
resid_h = pred_H.sub(sample_hierarchical_data[("Y_H", 0)], axis=0)
reisd_naive = naive.sub(sample_hierarchical_data[("Y_H", 0)], axis=0)

#%%
rmse(resid_h)
#%%

rmse(reisd_naive)


#%%
test = sample_data
test2 = sample_hierarchical_data


#%%
from py_online_forecast.core import DEFAULT_SOURCE as source
from py_online_forecast.datasets import ar_data, sample_hierarchical_data, phiA, phiH, phi_1, phi_2, Y_train, Y_test, Y_test_top
from py_online_forecast.prediction import ARX
from py_online_forecast.forecast_tools import ForecastFormat
import numpy as np
# %%
# %%
model_H = ARX(source, 2, 2, init_K = 0.00001)
model_H.set_format(ForecastFormat)
# %%
res = model_H(ar_data[[("Y_H", 0)]], track_state = True, mem = 1)
pred_mean = res["mean"]
pred_var = res["var"]
# %% Plot data and predictions
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(ar_data[("Y_H", 0)], label="Observed Y_H")
plt.plot(pred_mean, label="Predicted Y_H")


#%%
model_H.predictor.theta


# %%

# %%






















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
ar_data = pd.DataFrame(np.hstack([Y_test_top, Y_test[:, [1]]]), columns = pd.MultiIndex.from_tuples([("Y_A", 0), ("Y_H", 0)])).fc.convert()

#%%
lagged_predictions = fc_data.fc.lag()

# Compute residuals
resid_bot = lagged_predictions[["pred_Y_H"]].sub(Y_test[:, 1], axis = 0)
resid_top = lagged_predictions[["pred_Y_A"]].sub(Y_test[:, 0], axis = 0)

#%%
if __name__ == "__main__":

    resid_all = pd.concat([resid_bot, resid_top], axis=1)
    rmse = resid_all.apply(p.rmse)

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

sample_hierarchical_data = fc_data

# %%

