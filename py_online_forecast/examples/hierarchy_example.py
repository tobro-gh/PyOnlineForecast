#%%
import numpy
from py_online_forecast.hierarchies import *
import py_online_forecast.core as c
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% Load simulated data based on Møller, J.K., Nystrup, P., Madsen, H., 2024. Likelihood-based inference in temporal hierarchies. International Journal of Forecasting 40, 515–531.
# NOTE: the data uses phi1 = 0.75 and phi2 = 0.2 for simulation
from py_online_forecast.datasets import sample_hierarchical_data as data

#%% We first setup the summation matrix for the hierarchy,
S = np.array([[1,1],[1,0],[0,1]])
S_top = S[[0]]


#%% Since this is a temporal hierarchy, we setup a backshift matrix B for obtaining the lagged observations
# Our hierarchy requires [Y_{H,t-1}, Y_{H,t}] as input columns
B = [1, 0] # [Y_{H,t-1}, Y_{H,t-0}]

# We can then instantiate a TemporalReconciler for online reconciliation.
#config = c.RRR.configure()
#config = SRRR.configure(S_top = S_top, predictor_params = {"l_shrink": "auto"})
#temporal_reconciler = TemporalReconciler(S_top, B, config, horizon = 2, variance_name = "cov", skip_duplicates=True)
temporal_reconciler = TemporalRidgeReconciliation(S_top, B, full_cov = True, skip_duplicates=True, opt_shrink = False, full_hierarchy_cov=False)

# The parameters are as follows:
# S_top: The summation matrix for the top level(s) of the hierarchy
# B: The backshift structure for the lagged observations
# predictor_type: The type of predictor to use for reconciliation (here, recursive ridge regression)
# horizon: The forecast horizon (2 in this case, since we have 1-step and 2-step ahead forecasts)
# variance_name: (optional) the name of the output from the predictor that contains the variance estimates
# skip_duplicates: (optional) whether to skip duplicate lagged observations for parameter estimation

# We then extract the data needed for reconciliation
Y_hat = data[["pred_Y_A","pred_Y_H"]]
Y_bot = data[["Y_H"]]

#%% To ensure the output matches the input format (i.e. the ForecastFormat), we need to specify a formatter for the reconciler
from py_online_forecast.forecast_tools import ForecastFormat
temporal_reconciler.set_format(ForecastFormat)
# If we dont specify a formatter, the reconciler won't keep track of the sources of the data and the output will just be a plain numpy array
#%% To fit the model, we use rec_fit.
res = temporal_reconciler.fit(Y_bot, Y_hat)

#%% If instead we require incremental updates (in a "real" online setting), we can use rec_update

# First, reset the state of the reconciler (i.e. clear previous data and parameter estimates)
temporal_reconciler.reset_state()

# Then, loop over rows in the data and update the reconciler one step at a time
result = []
result_cov  = []
for t in range(len(Y_bot)):
    Y_bot_t = Y_bot.iloc[[t]]
    Y_hat_t = Y_hat.iloc[[t]]
    rec_t = temporal_reconciler.update(Y_bot_t, Y_hat_t)
    result.append(rec_t["mean"])
    result_cov.append(rec_t["cov"])

result_df = pd.concat(result)
result_cov_df = pd.concat(result_cov)
#%% For general hierarchies, we instead use the Reconciler class, which does not use the backshift structure
#reconciler = Reconciler(S_top, config, horizon = 2, variance_name = "cov")
reconciler = RidgeReconciliation(S_top, horizon = 2, opt_shrink = False)

# Alternatively,

# Use the recursive ridge regressor with optimal shrinkage
#reconciler = Reconciler(S_top, predictor_type = SRRR, horizon = 2, variance_name = "cov", predictor_init_params={"S_top": S_top}, predictor_params = {"l_shrink": "auto"})

# Use a simple OLS batch predictor (in sample)
#reconciler = Reconciler(S_top, predictor_type = c.OLS, horizon = 2) # Use this for in sample reconciliation

# NOTE: in sample results on variance-reduction are not guaranteed in the online setting, hence, forecasts may get worse depending on the setup

# Since the hierarchy is temporal, we need to manually lag the bottom level observations. For this we use the BackShift class (which is also used internally by the TemporalReconciler).
bs = p.BackShift([[1], [0]], skip_duplicates=True)
Y_bot_lagged = bs.apply(Y_bot)

# Then we can fit the reconciler using rec_fit
rec_result = reconciler.fit(Y_bot_lagged, Y_hat)

# The output contains both mean and covariance estimates (if we use RRR), so we extract the mean forecasts
Y_hat_rec = rec_result["mean"]

# Or again, using ForecastFormat
reconciler.set_format(ForecastFormat)
rec_result = reconciler.fit(Y_bot_lagged, Y_hat)
Y_hat_rec = rec_result["mean"]

#%% Plot the data
Y_top = data[["Y_A"]]

n_start = 100
n_plot = 50
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.set_title("Half-year")
Y_bot.iloc[n_start:n_start + n_plot].plot(ax=ax1)
Y_hat_rec[["pred_Y_H"]].fc.lag().iloc[n_start:n_start + n_plot].plot(ax=ax1)
Y_hat[["pred_Y_H"]].fc.lag().iloc[n_start:n_start + n_plot].plot(ax=ax1)
ax1.legend(["Observed", "rec. 1-step", "rec. 2-step", "base 1-step", "base 2-step"])

ax2.set_title("Annual")
Y_top.iloc[n_start:n_start + n_plot].plot(ax=ax2)
Y_hat_rec[["pred_Y_A"]].fc.lag().iloc[n_start:n_start + n_plot].plot(ax=ax2)
Y_hat[["pred_Y_A"]].fc.lag().iloc[n_start:n_start + n_plot].plot(ax=ax2)
ax2.legend(["Observed", "rec. 2-step", "base 2-step"])

#%% Get residuals
resid_base_bot = -Y_hat[["pred_Y_H"]].fc.lag().sub(Y_bot.to_numpy(), axis = 0)
resid_base_top = -Y_hat[["pred_Y_A"]].fc.lag().sub(Y_top.to_numpy())
resid_rec_bot = -Y_hat_rec[["pred_Y_H"]].fc.lag().sub(Y_bot.to_numpy(), axis = 0)
resid_rec_top = -Y_hat_rec[["pred_Y_A"]].fc.lag().sub(Y_top.to_numpy())

rmse_base_bot = p.rmse(resid_base_bot)
rmse_base_top = p.rmse(resid_base_top)
rmse_rec_bot = p.rmse(resid_rec_bot)
rmse_rec_top = p.rmse(resid_rec_top)

#%% Make histograms of residuals
fig, axes = plt.subplots(2, 2)
axes[0, 0].hist(resid_base_bot.dropna(), bins=30, alpha=0.7, density = True)
axes[0, 0].legend(["1-step", "2-step"])
axes[0, 0].set_title(f"Base bottom level, rmse: {rmse_base_bot:.3f}")
axes[0, 1].hist(resid_rec_bot.dropna(), bins=30, alpha=0.7, density = True)
axes[0, 1].legend(["1-step", "2-step"])
axes[0, 1].set_title(f"Reconciled bottom level, rmse: {rmse_rec_bot:.3f}")

# Also include gaussian with unity variance (true noise distribution) for reference
x = np.linspace(-4, 4, 100)
y = 1/np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)
axes[0, 0].plot(x, y)
axes[0, 1].plot(x, y)


axes[1, 0].hist(resid_base_top.dropna(), bins=30, alpha=0.7, density = True)
axes[1, 0].set_title(f"Base top level, rmse: {rmse_base_top:.3f}")
axes[1, 1].hist(resid_rec_top.dropna(), bins=30, alpha=0.7, density = True)
axes[1, 1].set_title(f"Reconciled top level, rmse: {rmse_rec_top:.3f}")
plt.tight_layout()

#%% To see how the reconciler works, we can manually apply data transformations to retrieve the design matrix and target variables
X_rec = reconciler.X({reconciler.Y_bot: Y_bot_lagged, reconciler.Y_hat: Y_hat})
Y_rec = reconciler.Y({reconciler.Y_bot: Y_bot_lagged, reconciler.Y_hat: Y_hat})

# Notice, X_rec is a dict with keys "X_pred" and "X_train", where X_train is the same as X_pred but lagged according to the forecast horizon (2 in this case).

#%% For more complex hierarchies, we the library provides tools to build and manipulate hierarchies using Nodes.
v00 = Node(name="v00")
v00_1 = v00.shift(1)
v01 = Node(name="v01")
v10 = Node(v00, v00_1, name = "v10")
v2 = Node(v10, v01, name = "v2")

v2.print_hierarchy()
bot_nodes = v2.get_leaf_nodes()
top_nodes, S_top = v2.build_S_top(*bot_nodes)

# Building A_lat will be same as S_top if the the observed nodes are the bottom nodes
lat_nodes, A_lat = v2.build_A_lat(*bot_nodes)

#%% Or alternatively, if we observe v2, v00 and v00_1
lat_nodes, A_lat = v2.build_A_lat(v2, v00, v00_1)


#%% Constructing more complex hierarchies can be done manually using Nodes
level_names = {}
def get_name(level):
    if level not in level_names:
        level_names[level] = 0
    else:
        level_names[level] += 1
    return f"Y_{level}_{level_names[level]}"
bot_nodes = [Node(name = get_name(0)) for i in range(8)]

level_1 = [Node(*bot_nodes[i*2:(i+1)*2], name = get_name(1)) for i in range(4)] 
level_2 = [Node(*level_1[:2], name = get_name(2)), Node(*level_1[2:], name = get_name(2))]
level_3 = Node(level_2[0], level_2[1], name = get_name(3))
level_3.print_hierarchy()

#%% Alternatively, the make_hierarchy function can be used
nodes, bot_nodes, top_nodes = make_hierarchy({"v3": ["v20", "v21"],
                                            "v20": ["v00", "v01"],
                                            "v21": ["v02", "v03"],
                                            "v00": ["v000", "v001"],
                                            "v01": ["v010", "v011"],
                                            "v02": ["v020", "v021"],
                                            "v03": ["v030", "v031"]})
nodes["v3"].print_hierarchy()

#%% S_top and A_lat can then be retrieved from the top node
top_nodes, S_top = nodes["v3"].build_S_top(*bot_nodes)


#%% Temporal hierarchy

# Hierarchies should be built so that only leaf nodes are lagged. Any hierarchy can be built this way?

#nodes, bot_nodes, top_nodes = make_hierarchy({"v2": [("v10", 1)], "v10": ["v00", "v01"]})
# Instead of the above, use
nodes, bot_nodes, top_nodes = make_hierarchy({"v2": ["v10_1"], "v10_1": [("v00", 1), ("v01", 1)]})

#%% A little more complex hierarchy
nodes, bot_nodes, top_nodes = make_hierarchy({"v2": ["v10", "v11", "v11_1"], "v11": ["v00", "v01"], "v11_1": [("v00", 1), ("v01", 1)]})
nodes["v2"].print_hierarchy()
obs_nodes, B = nodes["v2"].build_B("v00", "v01", "v10")
lat_nodes, A_lat = nodes["v2"].build_A_lat(*obs_nodes)
# %%
