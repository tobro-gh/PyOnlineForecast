"""Basic functionality for making discrete horizon forecasts.

The module is built around transformations for making predictions for discrete horizon
forecasting problems. The Prediction transformation provides an extensible base class
for handling data updates for forecasting models. The class implies that subclasses
should implement an ``update`` and  ``predict`` method, which are used to train the model
and make predictions without altering the model state.

Prediction is subclassed by another base class, OnlinePrediction, which requires the use
of a corresponding online pair of methods that enable incremental updates.

The module further contains some ready to use prediction subclasses,
- WLS: simple weighted least squares predictor
- RRR: an online multivariate linear model using recursive ridge regression
- ARX: autoregressive model with exogenous input using recursive ridge regression
"""

import inspect
from abc import abstractmethod

import numpy as np

from .core import (
    DEFAULT_SOURCE,
#    MEMORY,
#    Apply,
    CircularBuffer,
    Dim,
#    Source,
    Transformation,
    Data,
    Param,
)
from .features import (
    ForgettingVariance,
    Lag,
    ToArray,
)
def stack(forecasts):
    """Stack results from multiple rows into a single output."""

    # Convert to dict of lists
    forecasts = {key: [r[key] for r in forecasts] for key in forecasts[0]}

    # Stack any numpy arrays
    for key, values in forecasts.items():

        # Check for a reference value
        ref_val = next(iter(value for value in values if value is not None))

        # Check if any values are None
        if any(value is None for value in values) and ref_val is not None:

            if isinstance(ref_val, np.ndarray):
                ref_shape = ref_val.shape

                # Convert all None values to appropriate arrays of NaNs
                for j, value in enumerate(values):
                    if value is None:
                        new_val = np.full(ref_shape, np.nan)
                        values[j] = new_val

        # Stack values
        forecasts[key] = np.array(values)

    return forecasts

class Stack(Transformation):
    """Stack results from multiple rows into a single output."""

    def __init__(self, values):
        super().__init__(stack, values)

class RRRPredictor:
    """A class for performing recursive ridge regression."""

    def __init__(self, n, m, init_K):
        if init_K is None:
            init_K = 1e-6
        self.K = np.eye(n) * init_K
        self.L = np.zeros((n, m))
        self.H = np.zeros((n, n))
        self.theta = np.full((n, m), np.nan)
        self.psi = np.full((n, n), np.nan)

        self.n, self.m = n, m

    def update(self, x, y, mem, theta0, Q):
        """Update parameters."""

        n, m = self.n, self.m
        # Check inputs
        if mem < 0 or mem > 1:
            raise ValueError("Memory must be between 0 and 1.")

        if theta0 is None:
            theta0 = np.zeros((n, m))

        if Q is None:
            Q = np.zeros((n, n))
        elif isinstance(Q, (float, int)):
            Q = np.eye(n) * Q

        elif not np.allclose(Q, Q.T):
            raise ValueError("Q must be symmetric.")

        if theta0 is None:
            theta0 = np.zeros((n, m))

        data_ready = not np.isnan(y).any() and not np.isnan(x).any()

        if data_ready:

            # Perform update

            x_outer = np.outer(x, x)

            if self.K is None:
                self.K = x_outer
            else:
                self.K = mem * self.K + x_outer

            if self.L is None:
                self.L = np.outer(x, y)
            else:
                self.L = mem * self.L + np.outer(x, y)

            KpQ = self.K + Q
            LpQtheta = self.L + Q @ theta0

            self.theta = np.linalg.solve(KpQ, LpQtheta)

            # Update estimate of variance
            self.H = mem**2 * self.H + x_outer

            temp1 = np.linalg.solve(KpQ, self.H)

            self.psi = np.linalg.solve(KpQ, temp1.T).T  # K^-1 H K^-1^T

class RRR(Transformation):

    def __init__(
        self,
        X,
        Y,
        horizon=0,
        Q = None,
        theta0 = None,
        memory=0.99,
        init_K=None,
        update_theta = True,
        full_cov = True,
        cov_memory = None,
        track_memory = True,
        center_cov = False,
        return_theta = False,
        return_psi = False,
        return_residuals = False,
        return_residual_cov = False,
        return_pred_cov = True,
        iterate_Q = False,
        iterate_theta0 = False,
        iterate_memory = False,
        iterate_update_theta = False,
    ):

        self._init_K = init_K
        self.return_theta = return_theta
        self.return_psi = return_psi
        self.return_residuals = return_residuals
        self.return_residual_cov = return_residual_cov
        self.return_pred_cov = return_pred_cov
        self._full_cov = full_cov
        self.horizon = horizon

        self.X = X
        self.Y = Y


        # First, ensure X and Y are numpy arrays
        X = ToArray(X)
        Y = ToArray(Y)

        X_train = Lag(X, horizon)

        # Compose a parameter estimation transformation
        lm_iterate = ["x", "x_train", "y"] +  ["Q"] * iterate_Q + ["theta0"] * iterate_theta0 + ["memory"] * iterate_memory + ["update_theta"] * iterate_update_theta
        self._lm =Transformation(self._lm_func, X, X_train, Y, Q, theta0, memory, update_theta,
                                    STATE="state", ITERATE = lm_iterate)

        lm_stacked = Stack(self._lm)

        self.theta = lm_stacked["theta"]
        self.mean = lm_stacked["mean"]
        self.psi = lm_stacked["psi"]

        # Compute residuals in batch
        self.residuals = Transformation(self._residuals, Y, Lag(self.mean, horizon))

        # Then compute variance using ForgettingVariance
        mem_cov = cov_memory if cov_memory is not None else memory
        self.cov = ForgettingVariance(mem_cov, self.residuals, track_memory, center_cov, full_cov)

        if return_pred_cov:
            self.pred_cov = Stack(Transformation(self._pred_variance, X, self.cov, self.psi, ITERATE = ("x", "var", "psi")))
        else:
            self.pred_cov = None

        super().__init__(self._finalize, self.mean, self.theta, self.psi, self.cov, self.pred_cov, self.residuals)
        
    def _lm_func(self, x, x_train, y, Q, theta0, mem, update_theta, state = None):
        """Compute parameters and mean forecast for a single time step."""
        # Update state
        if state is None:
            state = RRRPredictor(n=x_train.shape[0], m=y.shape[0], init_K=self._init_K)

        if update_theta:
            state.update(x_train, y, mem, theta0, Q)

        result = {"mean": x.T @ state.theta, 
                  "theta": state.theta, 
                  "psi": state.psi}

        return result, state

    def _residuals(self, Y, mean):
        """Compute residuals."""
        return Y - mean

    def _pred_variance(self, x, var, psi):
        """Compute prediction variance and collect results."""

        # Compute covariance of prediction error
        var_pred_err = var * (1 + x.T @ psi @ x)

        if not self._full_cov:
            var_pred_err = np.diag(var_pred_err)

        return {"cov": var_pred_err}

    def _finalize(self, mean, theta, psi, cov, pred_cov, residuals):
        """Finalize output."""
        result = {"mean": mean}

        if pred_cov is not None:
            result["cov"] = pred_cov["cov"]

        if self.return_theta:
            result["theta"] = theta

        if self.return_psi:
            result["psi"] = psi

        if self.return_residual_cov:
            result["residual_cov"] = cov

        if self.return_residuals:
            result["residuals"] = residuals

        return result

    @property
    def parameters(self):
        """Return current parameters."""
        state = self.states[self._lm]
        return {"theta": state.theta, "psi": state.psi}



def rmse(x):
    """Compute the root-mean-square error of x."""
    return np.sqrt(np.mean(x**2))

class RMSE(Transformation):
    """Compute the root-mean-square error of x."""

    def __init__(self, x):
        super().__init__(rmse, x)


class BackShift(Transformation):
    r"""Create columns of time-lagged variables.
    
    Creates output variables as lags of input variables, specified by a list of shifts.
    Useful for constructing autoregressive features.
    
    Parameters
    ----------
    shifts : list
        Shift specification as list. Either integer lags or tuples of (input_index, lag)
        can be used.
    data : Source, default DEFAULT_SOURCE
        Input data source.
    skip_duplicates : bool, default False
        If True, output contains values only at every max_shift+1 rows. This ensures
        that none of the input values are repeated in the output.
    initial_value : float, default np.nan
        Fill value for unavailable historical data.
    
    Attributes
    ----------
    n : int
        Number of output variables.
    max_shift : int
        Maximum lag across all shifts.
    shifts : dict
        Normalized shift specification as {(i, j): lag} mapping.
    """

    def __init__(
        self,
        shifts: list,
        data=DEFAULT_SOURCE,
        skip_duplicates=False,
        initial_value=np.nan,
    ):
        self.shifts = []
        for shift in shifts:
            if isinstance(shift, int):
                self.shifts.append((0, shift))
            else:
                self.shifts.append(shift)

        self.n = len(shifts)
        self.skip_duplicates = skip_duplicates
        self.max_shift = max(shift[1] for shift in self.shifts)
        self.initial_value = initial_value

        super().__init__(self.evaluate, data=data, STATE="memory")

    def evaluate(self, data, memory=None):
        """Return combinations of lagged input data.
        
        Parameters
        ----------
        data : ndarray of shape (n_obs, n_inputs)
            Current input data.
        memory : tuple of (ndarray, int or None), optional
            Previous (historical_data, offset) from prior evaluation.
        
        Returns
        -------
        X : ndarray of shape (n_obs, n_outputs)
            Transformed output with lagged combinations. If skip_duplicates=True,
            most rows are NaN. n_outputs is equal to the number of shifts specified.
        memory : tuple
            Updated (historical_data, offset) for next call.
        """
        # Ensure 2D
        if data.ndim != 2:
            data = np.reshape(data, (data.shape[0], -1))  # Ensure 2D array

        # Fetch data from memory
        # TODO: use CircularBuffer for efficiency
        if memory is None:
            old_data = np.full((self.max_shift, data.shape[1]), self.initial_value)
            if self.skip_duplicates:
                offset = 0
        else:
            old_data, offset = memory

        all_data = np.vstack((old_data, data))

        shifted_data = []
        # Input Z (t x n) -> Temp Y -> Output X
        t = data.shape[0]

        # Collect lagged series
        for (j, lag) in self.shifts:
            if lag == 0:
                shifted_data.append(all_data[
                    self.max_shift : self.max_shift + t, j
                ])
            elif lag > 0:
                shifted_data.append(all_data[-(t + lag) : -lag, j]
                )

        # Concatenate arrays
        result = np.column_stack(shifted_data)

        # Get index for next memory
        end_idx = len(all_data) - self.max_shift

        # Form output
        if self.skip_duplicates:

            # Get mask to select every max(lag)'th row, starting from an offset
            mask = np.arange(self.max_shift - offset, t, self.max_shift + 1)

            # Update offset
            offset = (offset + t) % (self.max_shift + 1)

            # Fill non-masked rows with NaN
            masked = np.full_like(result, np.nan)
            masked[mask, :] = result[mask, :]

            return masked, (all_data[end_idx:], offset)
        
        else:
            return result, (all_data[end_idx:], None)

class ARX(Transformation):
    r"""Autoregressive model with exogenous input using recursive ridge regression.

    Combines ``BackShift`` to construct AR lags of the endogenous variable and
    stacks with exogenous features for forecasting via ``RRRPredictor``.

    The model learns a one-step-ahead predictor and rolls forward to produce
    h-step-ahead forecasts for horizon h.

    Parameters
    ----------
    exog : Source
        Exogenous features. Should produce a an array of forecasted exogenous values.
    endog : Source
        Endogenous target variable. Should produce an array of endogenous values.
    horizon : int
        Forecast horizon in time steps.
    p : int or list
        Autoregressive order (number of lags), or a list of specific lags to include.
    burn_in, init_K, track_memory, combine_variance, full_cov, scorefun, default_params
        Forwarded to ``RRRPredictor``. See ``RRR`` for details.

    Attributes
    ----------
    p : int or list
        AR order or list of specific lags used as features.

    See Also
    --------
    RRR : Recursive ridge regression wrapper.
    BackShift : Lag transformation for AR features.
    """
    
    def __init__(
        self,
        endog,
        horizon: int,
        order: int | list,
        exog = None,
        Q = None,
        theta0 = None,
        memory=0.99,
        init_K=None,
        update_theta = True,
        cov_memory = None,
        track_memory = True,
        center_cov = False,
        return_theta = False,
        return_psi = False,
        return_residuals = False,
        return_residual_cov = False
    ):
        self._init_K = init_K
        self.return_theta = return_theta
        self.return_psi = return_psi
        self.return_residuals = return_residuals
        self.return_residual_cov = return_residual_cov
        self.horizon = horizon

        self.endog = endog
        
        self._lag_indices = np.array(order)-1 if isinstance(order, list) else np.arange(order)
        self.p = max(self._lag_indices) + 1

        # Ensure endog is (t, 1)
        endog = Transformation(lambda x: x.reshape(x.shape[0], 1) if x.ndim == 1 else x, endog)

        # Make regression model for 1-step forecasts
        self.endog_all = BackShift(
            list(range(self.p)), endog
        )  # ordered as (y_t, y_t-1, ..., y_t-p+1)

        self.regressor = self.endog_all[:, self._lag_indices] # We only need a subset of lags for the regression part

        # If exogenous features are provided, include the 1-step-ahead forecast
        if exog is not None:
            self.regressor = Transformation(
                lambda x, y: np.hstack((x, y)), self.regressor, exog[:, 0]
            )

        self.regression = RRR(self.regressor, endog, 1, Q, theta0, memory, init_K, update_theta,
                True, cov_memory, track_memory, center_cov, True, 
                return_psi, return_residuals, True, False)
 

        super().__init__(self._loop, self.endog_all, self.regression, exog)

    def _loop(self, endog, rrr, exog = None):

        # Manually iterate over time steps

        n = endog.shape[0]
        forecasts = []

        for t in range(n):
            # Get current endogenous and exogenous values
            endog_t = endog[t]
            exog_t = exog[t] if exog is not None else None
            theta_t = rrr["theta"][t].flatten()
            resid_cov = rrr["residual_cov"][t]

            # Compute forecasts for this time step
            pred_t = self._forecast(endog_t, theta_t, resid_cov, exog_t)

            forecasts.append(pred_t)

        # Stack forecasts
        result = stack(forecasts)

        # Combine results with RRR outputs
        if self.return_theta:
            result["theta"] = rrr["theta"]
        if self.return_psi:
            result["psi"] = rrr["psi"]
        if self.return_residuals:
            result["residuals"] = rrr["residuals"]
        if self.return_residual_cov:
            result["residual_cov"] = rrr["cov"]

        return result

    def _forecast(self, endog, theta, resid_cov, exog = None):
        """Compute forecasts for a single time step."""

        result = {}

        arx_mean = np.full(self.horizon, np.nan)
        arx_var = np.full(
            self.horizon, np.nan
        )  # TODO: consider computing full covariance between horizons rather than just variances

        weights = np.zeros(max(self.horizon, self.p))  # Initialize AR weights for variance computation
        weights[0] = 1

        ar_params = theta[: len(self._lag_indices)]  # AR coefficients from theta

        # Compute forecasts for all horizons
        for h in range(self.horizon):

            reg_i = endog[self._lag_indices] # Select only the lags specified by order

            # Combine AR with exogenous features for horizon h
            if exog is not None:
                reg_i = np.hstack((reg_i, exog[h]))

            # Predict next step ahead using theta and current regressor
            arx_mean[h] = (reg_i.T @ theta).item()

            # Compute variance, using weights so far.
            arx_var[h] = (resid_cov * (weights @ weights)).item()

            # Push weights forward for next prediction. Use only the specified lags
            new_weight = weights[self._lag_indices] @ ar_params

            # Prepend new weight
            weights[1:] = weights[:-1]
            weights[0] = new_weight

            # Update endogenous history
            endog[1:] = endog[:-1]
            endog[0] = arx_mean[h]

        result["mean"] = arx_mean
        result["var"] = arx_var

        return result