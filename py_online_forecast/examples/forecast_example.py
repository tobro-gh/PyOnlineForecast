#%%
from py_online_forecast import *
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#%%============================================================================
# SECTION 0: DATA
#%%============================================================================
#%% 0.1 We use already prepared sample data
from py_online_forecast.datasets import sample_data


#%%
data = sample_data.fc.subset(horizons = (0,1,2,4), end_index=200)

# The sample data is a pandas dataframe, which uses a special "fc" accessor.
# Note: for the basic model setup, this format is not strictly required.

#%% 0.2 The fc acessor
# The fc acessor provides functionality for the "forecast matrix" format for a dataframe.
# This format uses a multi-index, to match variables and forecast horizons.
# The fc accessor provides methods for subsetting and lagging columns in the dataframe
# according to the horizon, e.g. a forecast at horizon 2 is lagged 2 time steps forward,
# to match the time of the observation. Observations should have horizon 0.
data.fc.lag()

# We can subset the data to only include certain horizons and variables
data.fc.subset("Ta", "Taobs", horizons = (0,1,2))

# In case dataframes "almost" match the forecast matrix format, the fc accessor
# provides a method for converting to the correct format.
data.fc.convert()


#%%============================================================================
# SECTION 1: BASIC MODEL SETUP
#%%============================================================================

#%% 1.1 Define Input Variables  
# Our goal is to model energy given ambient temperature observations and forecasts
energy = DEFAULT_SOURCE[["energy"]]
taobs = DEFAULT_SOURCE[["Taobs"]]
ta = Subset("Ta", horizons = (1,))
const = One()

#%% 1.2 Create Simple OLS Model

# Model energy_{t+1} = f(X(Taobs_t, Ta_t)) using low-pass filters
lp_Taobs = LowPass(taobs, alpha = 0.7)
lp_Ta = LowPass(ta, alpha = 0.8)

# We then create a Prediction for the 1-hour ahead forecast
X = Combine(const, lp_Taobs, lp_Ta)
pred = WLS(X, energy, horizon = 1)

# And to manage data and access additional functionality, we wrap it in a Model
model = Model(pred)

# Note, if we do not need the prediction outside the model, we can also create the model directly,
#model = Model.construct((const, lp_Taobs, lp_Ta), energy, ols_config, 1)

#%% 1.3 Fit Model and Make Predictions
# Fit the model and make predictions
result = model.fit(data)

# Plot the predicted mean versus the observations
fig, ax = plt.subplots()
data.plot(y="energy", ax = ax)
result.shift(1).plot(ax = ax) # Shift predictions 1 hour ahead
plt.legend(["Observations", "Forecast"])
plt.show()

# Compare with least squares estimate
beta_ols = model.predictor

#%%
model.reset_state()
X_data = pred.X.apply(data)
X_train_data = X_data.shift(1)
Y_data = pred.Y.apply(data)

beta_np = np.linalg.lstsq(X_train_data[1:], Y_data.to_numpy()[1:], rcond=None)[0]

print(f"Model parameters from OLS model: {beta_ols}")
print(f"Model parameters from numpy lstsq: {beta_np}")

#%%============================================================================
# SECTION 3: ONLINE FORECASTING WITH RRR
#%%============================================================================

#%% 3.1 Configure Model for Online Forecasting
# In an online setting, switch to recursive ridge regressor (RRR).

# We make a new prediction for RRR
pred = RRR(X, energy, horizon = 1, track_memory = True, tilde_k_init_val = 0.00001) # Note, if memory tracking is not enabled, covariance estimation will not work properly when memory = 1. If covariance estimation is not needed, this can be left out.
model = Model(pred)

#%% 3.2 Fit Model and Make Predictions
# Fit the model and make predictions as before
result = model.fit(data, mem = 1)
fig, ax = plt.subplots()
data.plot(y="energy", ax = ax)
result["mean"].shift(1).plot(ax = ax) # Shift predictions 1 hour ahead
plt.legend(["Observations", "Forecast"])
plt.show()

#%% 3.3 Multivariate target
# The model also supports multivariate targets, e.g. energy and Taobs
target = DEFAULT_SOURCE[["energy", "Taobs"]]
model_multi = Model(RRR(X, target, horizon = 1, track_memory = True, tilde_k_init_val = 0.00001))
result_multi = model_multi.fit(data, mem = 1)
fig, ax = plt.subplots()
data.plot(y="energy", ax = ax)
data.plot(y="Taobs", ax = ax)
result_multi["mean"].shift(1).plot(ax = ax) # Shift predictions 1 hour ahead
plt.legend(["Energy obs", "Taobs", "Energy forecast", "Taobs forecast"])
plt.show()

#%% 3.4 Online Data Update Simulation
# To emulate a real online setting, provide data in small chunks. Note, inputs are always assumed
# to have observations along the first axis. This means, if a single observation is provided, it should be padded with
# an extra dimension, e.g. data.iloc[[i]] for the i'th observation.
subsets = [data.iloc[[i]] for i in range(len(data))]

# Reset model for proper comparison
model.reset_state()
predictions = []


# Update model with each data subset
for s in subsets:
    p = model.update(s, mem = 1)["mean"]
    predictions.append(p)

predictions = pd.concat(predictions)

# Compare predictions with the previous results
print(np.allclose(result["mean"].to_numpy()[1:], predictions.to_numpy()[1:])) # Should be True


# Models can also be updated without updating predictor parameters, i.e. only predict,
model.update(data, update_predictor = False)

#%%============================================================================
# SECTION 4: MULTI-HORIZON FORECASTING
#%%============================================================================

#%% 4.1 Separate predictions for each horizon
# Create a new model for 2-hour ahead forecasts
ta2 = Subset("Ta", horizons = (2,)) # Use all available forecasts of Ta

# Low-pass filter for the new model
lp_Ta2 = LowPass(ta2, alpha = 0.8)

X2 = Combine(const, lp_Taobs, lp_Ta2)

# We then make a prediction for the 2-hour ahead forecast
pred_2h = RRR(X2, energy, horizon = 2, track_memory = True, tilde_k_init_val = 0.00001)

# Note, we can still run the prediction as a standalone object
ref_1h = pred.apply(data, mem = 1)
ref_2h = pred_2h.apply(data, mem = 1)


#%% 4.2 Model Ensemble
# For convenience, we can create a model of both outputs,
models = Model(pred, pred_2h)

# Fit the ensemble model
result = models.fit(data, mem = 1)

# Results are returned in a dict indexed by the predictions
result_model_1h = result[pred]
result_model_2h = result[pred_2h]

# Note: ensembles also support the update methods, and the update_predictor argument

#%% 4.3 Horizon Ensemble
# For the RRR prediction, ensembles can also be created using the "RRREnsemble" subclass of the model.
# In this case, we first include all forecast horizons, and let the ensemble handle
# the separation into different models.
ta_all = DEFAULT_SOURCE[["Ta"]]
lp_Ta_all = LowPass(ta_all, alpha = 0.8)
Xall = Combine(const, lp_Taobs, lp_Ta_all)
horizon_models = RRREnsemble(Xall, energy, horizons = (1,2), track_memory = True, tilde_k_init_val = 0.00001)

# Then fit the ensemble as normal
horizon_result = horizon_models.fit(data, mem = 1)

# Note, the horizon ensemble concatenates results for all horizons in one dataframe
# Ensembles can also be updated in an online fashion
horizon_models.reset_state()
predictions = []
for s in subsets:
    p = horizon_models.update(s, mem = 1)["mean"]
    predictions.append(p)

predictions = pd.concat(predictions, axis = 0)

# Compare (except two first rows, as they contain NaNs)
print(np.allclose(horizon_result["mean"].to_numpy()[2:], predictions.to_numpy()[2:])) # Should be True

#%%============================================================================
# SECTION 5: TRANSFORMS AND TRANSFORMERS
#%%============================================================================

#%% 5.1 Basic Transform Usage
# Transforms are data-less objects that can be applied on-the-fly to data. The Transformer
# and ForecastModel classes use transforms to define the inputs for regression models.
# Custom transforms can be defined by inheriting from the Transform class and implementing
# the evaluate method.

# Transforms are instantiated with parameters, e.g.
low_pass_filter = LowPass(ta_all, alpha = 0.8)

# Here "Ta" is a reference to the name of the column in the dataframe that the transform will be applied to.

#%% 5.2 Transform Parameter Updates
# Parameters are stored in the transform object and should be updated accordingly
low_pass_filter.alpha = 0.9

# Transforms can be applied to dataframes using the apply method. 
# This returns the transformed data and any recursion parameters in a tuple.
low_pass_filter.apply(data)

#%% 5.3 Advanced Transform Usage
# Other references to data can be used in transforms, including other transforms and some
# special placeholders. For example we can apply a fourier series transform to the low-pass filtered Ta data.
fs = FourierSeries(low_pass_filter, nharmonics= 10)

# Basic arithmetic operations can be used to combine transforms, e.g. addition, subtraction, multiplication and division.
# This works as a placeholder for performing the operation on the data returned by application of the transforms.
lp_square = low_pass_filter**2

# When using binary operations, care should be taken to ensure that the transformed outputs are compatible.
sum_transform = low_pass_filter + lp_square
sum_transform.apply(data)

#%%============================================================================
# SECTION 6: SOURCES
#%%============================================================================
#%% 6.1 Basic Source Usage
# The workflow using transformations more generally uses "Sources", which are
# abstract placeholders for data. Transformations use Sources as inputs, and are
# themselves Sources for other transformations or models. A basic source can be
# created directly using,
raw_data = Source("Raw data")

# Then it can be used as input for transformations, e.g.
lp_raw = LowPass(raw_data, alpha = 0.8)

# When evaluaintg transformations, either directly or through a Transformer, the
# sources are matched to data, e.g.,
lp_raw.apply({raw_data: data})

# In case apply is called without mentioning a source, it is assumed to be the DEFAULT_SOURCE
DEFAULT_SOURCE

# For some Transformations, Source arguments are optional. In this case, the
# DEFAULT_SOURCE is used if no source is provided.
#%% 6.2 Special sources
# There are some special sources, i.e. DEFAULT_SOURCE, DefaultIndex, MEMORY, PREDICTOR_PARAMETERS, UPDATE_PREDICTOR,
#  X_init, Y_init, DIM_X and DIM_Y.
# DEFAULT_SOURCE represents the default input data, when no other source is specified.
# DefaultIndex represents the index of the default input data.
# MEMORY represents the memory state of the model/transform, i.e. the recursion parameters.
# PREDICTOR_PARAMETERS is used to pass parameters to the predictor during updates.
# UPDATE_PREDICTOR is used to indicate whether the predictor should be updated during an update.
# X_init, Y_init, DIM_X and DIM_Y are used in predictor initialization. Specifically, X_init and Y_init are placeholders
# for the initial input and target data and DIM_X and DIM_Y correspond to Dim(X_init) and Dim(Y_init). These sources are
# only available within scopes when creating custom predictors.

from py_online_forecast.features import TimeOfDay
from py_online_forecast.core import parse_data

# The DEFAULT_INDEX source returns the index of the data, e.g.
tod = TimeOfDay(DEFAULT_INDEX)
tod.apply(data, ref = DEFAULT_SOURCE)

# The Index source data is created when data is parsed into a special dict,
data_dict = parse_data(data)
data_dict[DEFAULT_INDEX]

# They act as placeholders for the target variable, and predictions thereof. In an
# ensemble with multiple Models, the Target and Prediction sources will be specific
# to each model.

# The PredictionHorizon source is used to specify the (integer) horizon for the prediction.

#%%============================================================================
# SECTION 7: HYPERPARAMETER OPTIMIZATION
#%%============================================================================
#%% 7.1 Basic Hyperparameter Optimization
# The hyperparameters of a model, i.e. the parameters of the transformations and
# the predictor, can be optimized using the fit_and_score method.

# For a non-trivial case, we add a slow trend in the data
trend_data = data.copy()
trend_data[("energy",0)] += 0.1*np.arange(len(trend_data))

# Set score mode to true
pred.set_score_mode()

# Or via. the model
model.set_score_mode()

def obj(x):
    alpha, mem = x
    lp_Taobs.alpha = alpha
    lp_Ta.alpha = alpha
    return model.fit(trend_data, mem = mem)["score"]

res = minimize(obj, x0 = [0.5, 0.5], bounds = [(0, 1), (0, 1)], method = "Nelder-Mead")
print(res)

#%% 7.2 Hyperparameter Optimization for Ensembles
# And similarly for the horizon ensemble
horizon_models.set_score_mode()
def obj_horizon(x):
    alpha, mem = x
    lp_Taobs.alpha = alpha
    lp_Ta_all.alpha = alpha
    result = horizon_models.fit(trend_data, mem = mem)
    return np.mean(list(result["score"].values())) # Average score across horizons

res_horizon = minimize(obj_horizon, x0 = [0.5, 0.5], bounds = [(0, 1), (0, 1)], method = "Nelder-Mead")
print(res_horizon)
#%%============================================================================
# SECTION 8: CUSTOM TRANSFORMS AND PREDICTORS
#%%============================================================================
#%% 8.1 Custom Transform
# Custom transforms can be created by inheriting from the Transformation class
# and implementing the evaluate method.
class CustomTransform(Transformation):
    def __init__(self, source, param1 = 0.5, param2 = 1.0):
        super().__init__(data = source, old_param = MEMORY)
        self.param1 = param1
        self.param2 = param2
    
    def evaluate(self, data, old_param = None):
        old_param = old_param or 1.0
        param = old_param*self.param1 + self.param2
        return data*param, param

# In the above ...
