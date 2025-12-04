#%%
from py_online_forecast.core import * 
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#%%============================================================================
# SECTION 0: DATA
#%%============================================================================
#%% 0.1 We use already prepared sample data
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
energy = Map("energy")
taobs = Map("Taobs")
ta = Subset("Ta", horizons = (1,))

#%% 1.2 Create Simple OLS Model
# Model energy_{t+1} = f(X(Taobs_t, Ta_t)) using low-pass filters
lp_Taobs = LowPass(taobs, alpha = 0.7)
lp_Ta = LowPass(ta, alpha = 0.8)
model = Model((lp_Taobs, lp_Ta), energy, OLS, 1)

#%% 1.3 Fit Model and Make Predictions
# Fit the model and make predictions
result = model.fit(data, return_y = True)

# Plot the predicted mean versus the observations
fig, ax = plt.subplots()
data.plot(y="energy", ax = ax)
result["mean"].shift(1).plot(ax = ax) # Shift predictions 1 hour ahead
plt.legend(["Observations", "Forecast"])
plt.show()

# Compare with least squares estimate
beta_ols = model.predictor.get_model_params()
model.reset_state()
X = model.X.apply(data)
X_train = X["X_train"]
Y = model.Y.apply(data)

beta_np = np.linalg.lstsq(X_train[1:], Y.to_numpy()[1:], rcond=None)[0]

print(f"Model parameters from OLS model: {beta_ols}")
print(f"Model parameters from numpy lstsq: {beta_np}")

#%%============================================================================
# SECTION 3: ONLINE FORECASTING WITH RRR
#%%============================================================================

#%% 3.1 Configure Model for Online Forecasting
# In an online setting, switch to recursive ridge regressor (RRR)
model.configure_predictor(RRR, predictor_init_params={"track_memory": True}) # Note, if memory tracking is not enabled, covariance estimation will not work properly when memory = 1. If covariance estimation is not needed, this can be left out.
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
target = Map("energy", "Taobs")
model_multi = Model((lp_Taobs, lp_Ta), target, RRR, 1)
result_multi = model_multi.fit(data, mem = 1)
fig, ax = plt.subplots()
data.plot(y="energy", ax = ax)
data.plot(y="Taobs", ax = ax)
result_multi["mean"].shift(1).plot(ax = ax) # Shift predictions 1 hour ahead
plt.legend(["Energy obs", "Taobs", "Energy forecast", "Taobs forecast"])
plt.show()

#%% 3.4 Online Data Update Simulation
# To emulate a real online setting, provide data in small chunks
subsets = [data.iloc[i] for i in range(len(data))]

# Reset model for proper comparison
model.reset_state()
predictions = []


# Update model with each data subset
for s in subsets:
    p = model.update(s, mem = 1)["mean"]
    predictions.append(p)

predictions = pd.concat(predictions)

# Compare predictions with the previous results
print(np.allclose(result["mean"].to_numpy().squeeze()[1:], predictions.to_numpy()[1:])) # Should be True


# Models can also be updated without updating predictor parameters, i.e. only predict,
model.update(data, update_predictor = False)

#%%============================================================================
# SECTION 4: MULTI-HORIZON FORECASTING
#%%============================================================================

#%% 4.1 Separate Models for Each Horizon
# Create a new model for 2-hour ahead forecasts
ta2 = Subset("Ta", horizons = (2,)) # Use all available forecasts of Ta

# Low-pass filter for the new model
lp_Ta2 = LowPass(ta2, alpha = 0.8)

model_2h = Model((lp_Taobs, lp_Ta2), energy, RRR, 2)

#%% 4.2 Model Ensemble
# Create an ensemble of the two models
models = Ensemble(model, model_2h)

# Fit the ensemble model
result = models.fit(data)

# Results are stored in a dict indexed by the models
result_model_1h = result[model]
result_model_2h = result[model_2h]

# Note: ensembles also support the update methods, and the update_predictor argument

#%% 4.3 Horizon Ensemble
# Ensembles can also be created using the HorizonEnsemble class.
# In this case, we first include all forecast horizons, and let the ensemble handle
# the separation into different models.
ta_all = Map("Ta")
lp_Ta_all = LowPass(ta_all, alpha = 0.8)
horizon_models = HorizonEnsemble((lp_Taobs, lp_Ta_all), energy, RRR, horizons = (1,2))

# Then fit the ensemble as normal
horizon_result = horizon_models.fit(data)

# Note, the horizon ensemble concatenates results for all horizons in one dataframe
# Ensembles can also be updated in an online fashion
horizon_models.reset_state()
predictions = []
for s in subsets:
    p = horizon_models.update(s)["mean"]
    predictions.append(p)

predictions = pd.concat(predictions, axis = 1).T

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

#%% 5.4 Using Transformer for Efficient Data Processing
# For complex combinations of transforms, a "Transformer" should be used to efficiently handle
# data dependencies and recursion.
transformer = Transformer()
transformer.add_transforms(low_pass_filter)
transformer.add_transforms(fs)
transformer.transform(data)

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

# In case apply is called without mentioning a source, it is asummed to be the DefaultSource
DefaultSource

# For some Transformations, Source arguments are optional. In this case, the
# DefaultSource is used if no source is provided. This is the case for the
# built-in transformations Map and Subset, which use the DefaultSource if no source is provided.
map_default = Map("Ta")

# For a transformer, we may specify what the incoming sources are, e.g. 
other_data = Source("Other data")
transformer = Transformer(raw_data, other_data)

# Then when transforming data, we need to provide both sources
transformer.transform({raw_data: data, other_data: data})

# In case the transformer is called with only a data argument, it will treat it
# as the DefaultSource
transformer = Transformer()
transformer.transform(data)
# is equivalent to
transformer.transform({DefaultSource: data})

#%% 6.2 Special sources
# There are some special sources, i.e. Index, Horizons, Target and Prediction.
# Index returns the index of a dataframe in the input data, by default the first
# element of the input data dict. It can be altered by calling ref = source, in
# either the apply or transform methods.

# The Index source returns the index of the data, e.g.
tod = TimeOfDay(DefaultIndex)
tod.apply(data, ref = DefaultSource)

# The Index source data is created when a Data object is created from theinput, i.e.
data_dict = Data(data)
data_dict[DefaultIndex]

# The Target and Prediction sources only works in tandem with a Model (or Ensemble).
# They act as placeholders for the target variable, and predictions thereof. In an
# ensemble with multiple Models, the Target and Prediction sources will be specific
# to each model.

# The PredictionHorizon source is used to specify the (integer) horizon for the prediction.

#%% Residuals transform
# The Residuals transform uses the Target and Prediction sources to compute
# residuals for a specific model. It is only valid within a model or ensemble.

residuals = Residuals(0) # Residuals per model, with default value 0

# The residuals can be used to create moving average type models, e.g.
ma1 = Model((One(), residuals), energy, RRR, 1)
res_ma1 = ma1.fit(data, Q = 0.01) # Here Q is a regularization parameter for RRR to avoid singularity caused by artificial 0 residuals

#%%============================================================================
# SECTION 7: HYPERPARAMETER OPTIMIZATION
#%%============================================================================
#%% 7.1 Basic Hyperparameter Optimization
# The hyperparameters of a model, i.e. the parameters of the transformations and
# the predictor, can be optimized using the fit_and_score method.

def obj(x):
    alpha, mem = x
    lp_Taobs.alpha = alpha
    lp_Ta2.alpha = alpha
    lp_Ta_all.alpha = alpha
    return model_2h.fit_and_score(data, scorefun = rmse, mem = mem)

res = minimize(obj, x0 = [1, 0.8], bounds = [(0.1, 0.99), (0.1, 0.99)], method = "L-BFGS-B")
print(res)

#%% 7.2 Hyperparameter Optimization for Ensembles
# And similarly for the horizon ensemble
def obj_horizon(x):
    alpha, mem = x
    lp_Taobs.alpha = alpha
    lp_Ta2.alpha = alpha
    lp_Ta_all.alpha = alpha
    scores = models.fit_and_score(data, scorefun = rmse, mem = mem)
    return np.mean(list(scores.values()))

res_horizon = minimize(obj_horizon, x0 = [1, 0.8], bounds = [(0.1, 0.99), (0.1, 0.99)], method = "L-BFGS-B")
print(res_horizon)


#%%=============================================================================
# SECTION 8: MULTISTEP FORECASTING
#%%=============================================================================
# The above examples have focused on individual "1-step" ahead forecasts each fitted
# to a specific horizon. However, the framework also supports multistep forecasting, 
# as usually done with ARMAX type models.

#%% 8.1 The ARX model
# ARX models can be used in the framework, by instantiating the ARX class. We chose to forecast 2 horizons,
horizon = 2
# The exogenous input should include forecasts for all the horizons, e.g. in this case 1 and 2 hour ahead forecasts of Ta.
exog = Subset("Ta", horizons = (1,2)) 
# We then create the ARX model, using autoregressive order p=2
arx_model = ARX(exog, energy, p = 2, horizon = horizon, predictor_init_params={"tilde_k_init_val": 0.001})

# The ARX model by construction uses the RRR predictor for parameter estimation and generation of forecasts. 
# Thus, any parameters for RRR can be provided when fitting the model. We use tilde_k_init_val = 0.001 here to avoid singularities for the initial fit.
arx_result = arx_model.fit(data)

fig, ax = plt.subplots()
data[[("energy",0)]].plot(ax = ax)
arx_result["mean"].fc.lag().plot(ax = ax)

#%% 8.2 The Exogenous transform
# The ARX model uses a special Exogenous transformation to handle the exogenous inputs.
ExogenousTransform(exog, 2).apply(data)
# The transform checks that all required forecast horizons are present and sorts the forecasts
# to ensure consistent input to the ARX model.

#%%============================================================================
# SECTION 9: CUSTOM TRANSFORMS AND PREDICTORS
#%%============================================================================
#%% 9.1 Custom Transform
# Custom transforms can be created by inheriting from the Transformation class
# and implementing the evaluate method.
class CustomTransform(Transformation):
    def __init__(self, source, param1 = 0.5, param2 = 1.0):
        super().__init__(data = source, old_param = Memory)
        self.param1 = param1
        self.param2 = param2
    
    def evaluate(self, data, old_param = None):
        old_param = old_param or 1.0
        param = old_param*self.param1 + self.param2
        return data*param, param

# In the above ...

# %%
