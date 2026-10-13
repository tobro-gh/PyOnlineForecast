"""Transformations and utilities for discrete-horizon forecasting.

The module provides recursive ridge regression, autoregressive forecasting
with exogenous inputs, lag construction, and forecast-result utilities. The
forecasting models are implemented as callable transformations and can be
composed with other transformations from the package.
"""
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
    """Stack dictionaries of forecast results along a new first axis.

    Array values are converted to NumPy arrays. Missing array values are
    replaced with NaN-filled arrays matching the first available value.

    Parameters
    ----------
    forecasts : sequence of dict
        Forecast result dictionaries with matching keys.

    Returns
    -------
    dict
        Dictionary containing the stacked forecast results.
    """
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
        """Update the recursive ridge-regression parameters.

        Parameters
        ----------
        x : ndarray
            Predictor vector with length ``n``.
        y : ndarray
            Response vector with length ``m``.
        mem : float
            Forgetting factor between zero and one.
        theta0 : ndarray or None
            Prior parameter matrix. If None, use zeros.
        Q : ndarray, scalar, or None
            Prior precision matrix. A scalar is interpreted as a multiple of
            the identity matrix. If None, use a zero matrix.

        Raises
        ------
        ValueError
            If ``mem`` is outside ``[0, 1]`` or ``Q`` is not symmetric.
        """
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
    """Transformation implementing recursive ridge regression.

    The transformation estimates a multivariate linear model sequentially,
    optionally updating its parameters for each observation. It can return
    forecasts, parameter estimates, residuals, and prediction or residual
    covariance estimates.

    Parameters
    ----------
    X : Transformation
        Predictor transformation.
    Y : Transformation
        Response transformation.
    horizon : int, default=0
        Number of observations by which the response is aligned with the predictors 
        during training.
    Q : ndarray, scalar, or None, optional
        Ridge prior precision. A scalar is interpreted as a multiple of the identity
        matrix.
    theta0 : ndarray or None, optional
        Prior parameter matrix.
    memory : float, default=0.99
        Forgetting factor used for recursive parameter updates.
    init_K : float or None, optional
        Initial diagonal value of the accumulated predictor matrix.
    update_theta : bool, default=True
        Whether to update the regression parameters at each observation.
    full_cov : bool, default=True
        Whether to return full covariance matrices for prediction errors.
    cov_memory : float or None, optional
        Forgetting factor for residual covariance estimation. Defaults to ``memory``.
    track_memory : bool, default=True
        Whether to track effective covariance-estimation memory.
    center_cov : bool, default=False
        Whether to center residuals before estimating covariance.
    return_theta : bool, default=False
        Whether to include parameter estimates in the output.
    return_psi : bool, default=False
        Whether to include parameter covariance factors in the output.
    return_residuals : bool, default=False
        Whether to include residuals in the output.
    return_residual_cov : bool, default=False
        Whether to include residual covariance estimates in the output.
    return_pred_cov : bool, default=True
        Whether to include prediction covariance estimates in the output.
    iterate_Q, iterate_theta0, iterate_memory : bool, default=False
        Whether the corresponding parameter is supplied separately for each observation.
    iterate_update_theta : bool, default=False
        Whether ``update_theta`` is supplied separately for each observation.

    Notes
    -----
    Calling an instance returns a dictionary containing at least ``mean``.
    Optional keys are controlled by the ``return_*`` arguments. The
    transformation uses state to continue recursive estimation across calls.
    """

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
    """Transformation that computes the root-mean-square error.

    Parameters
    ----------
    x : Transformation
        Input transformation providing the values for which to compute the
        error.

    Notes
    -----
    Calling an instance returns the root-mean-square value of the evaluated
    input.
    """

    def __init__(self, x):
        super().__init__(rmse, x)


class BackShift(Transformation):
    r"""Transformation that creates columns of time-lagged variables.

    The transformation constructs output columns by selecting input columns at the 
    requested lags. It is useful for constructing autoregressive features.

    Parameters
    ----------
    shifts : sequence of int or tuple
        Shift specifications. An integer denotes a lag of the default input
        column. A tuple ``(input_index, lag)`` selects a specific input
        column and lag.
    data : Transformation, default=DEFAULT_SOURCE
        Input transformation providing an array with shape
        ``(n_observations, n_inputs)``.
    skip_duplicates : bool, default False
        If True, return values only at selected non-overlapping rows and fill
        the remaining rows with NaN.
    initial_value : float, default np.nan
        Fill value for unavailable historical observations.

    Attributes
    ----------
    n : int
        Number of output variables.
    max_shift : int
        Maximum lag across all shifts.
    shifts : dict
        Normalized shift specifications.

    Notes
    -----
    Calling an instance returns an array containing the requested lagged
    columns. The transformation uses a state to continue lagging across
    separate input batches.
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
    r"""Transformation for autoregressive forecasting with exogenous input.

    The transformation combines autoregressive lags of the endogenous variable with 
    optional exogenous features and estimates the coefficients using recursive ridge 
    regression.

    Calling an instance rolls the one-step model forward to produce forecasts for the 
    configured horizon.

    Parameters
    ----------
    endog : Transformation
        Endogenous target transformation.
    horizon : int
        Number of future time steps to forecast.
    order : int or list of int
        Autoregressive order, or one-indexed lag positions to include.
    exog : Transformation, optional
        Exogenous feature transformation. Its first horizon column is used
        during model fitting and its horizon columns are used for forecasts.
    Q : ndarray, scalar, or None, optional
        Ridge prior precision passed to :class:`RRR`.
    theta0 : ndarray or None, optional
        Prior parameter matrix passed to :class:`RRR`.
    memory : float, default=0.99
        Forgetting factor for recursive parameter updates.
    init_K : float or None, optional
        Initial diagonal value for the recursive predictor.
    update_theta : bool, default=True
        Whether to update the regression parameters at each observation.
    cov_memory : float or None, optional
        Forgetting factor for residual covariance estimation.
    track_memory : bool, default=True
        Whether to track effective covariance-estimation memory.
    center_cov : bool, default=False
        Whether to center residuals before covariance estimation.
    return_theta : bool, default=False
        Whether to include parameter estimates in the output.
    return_psi : bool, default=False
        Whether to include parameter covariance factors in the output.
    return_residuals : bool, default=False
        Whether to include residuals in the output.
    return_residual_cov : bool, default=False
        Whether to include residual covariance estimates in the output.

    Attributes
    ----------
    p : int or list
        Largest autoregressive lag used by the model.

    See Also
    --------
    RRR : Recursive ridge regression wrapper.
    BackShift : Transformation for constructing lagged features.

    Notes
    -----
    The transformation uses state to continue recursive estimation and lag
    construction across separate input batches.
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