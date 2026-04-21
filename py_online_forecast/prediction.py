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
    MEMORY,
    Apply,
    CircularBuffer,
    Dim,
    Source,
    Transformation,
    make_keyword,
)
from .features import (
    ForgettingVariance,
    Lag,
)

X_INIT = make_keyword("X_INIT")
Y_INIT = make_keyword("Y_INIT")
Z_INIT = make_keyword("Z_INIT")

DIM_X = Dim(X_INIT)
DIM_Y = Dim(Y_INIT)
DIM_Z = Dim(Z_INIT)


def rmse(x):
    """Compute the root-mean-square error of x."""
    return np.sqrt(np.mean(x**2))


class Prediction(Transformation):
    r"""Base class for discrete horizon forecasting transformations.

    This class provides generic behavior for discrete horizon forecasting problems
    where the model learns from lagged inputs and targets. Subclasses must implement
    core methods for state creation, model updates, and prediction.

    Parameters
    ----------
    X : Source
        Input data source. Output should be compatible with the Lag transformation.
    Y : Source
        Target data source.
    horizon : int
        Forecast horizon in time steps.
    Z : Source, optional
        Additional exogenous features (not lagged) for use in prediction. If None,
        exogenous features are not used. Default is None.
    score_mode : bool, default False
        If True, the ``score`` method is called during evaluation to compute metrics.
    default_params : dict, optional
        Default parameters to pass to ``update`` and ``predict`` methods. These can be
        overridden by providing parameters in the ``apply`` method. Default is None.
    *args, **kwargs
        Additional arguments passed to the ``create`` method for initialization. Note,
        Sources will be resolved and their values passed in place of the Source objects.

    Attributes
    ----------
    predictor : object or None
        The fitted predictor state/parameters. Populated when ``apply`` is called with
        ``track_state`` enabled.

    Methods
    -------
    create
        Initialize predictor state (abstract, implemented by subclasses).
    update
        Update predictor state with new data and make predictions (abstract).
    predict
        Make predictions without updating predictor state (abstract).
    score
        Compute evaluation metric on predictions (optional).

    Notes
    -----
    The forecasting model assumes a relationship between lagged inputs and targets:

    .. math::

        Y_t = f(X_{t-h})

    where :math:`h` is the forecast horizon. Predictions are then made according to:

    .. math::

        \hat{Y}_{t+h} = g(\text{state}_t, Z_t)

    where :math:`\text{state}_t`` contains the learned model parameters and :math:``Z_t`
    are optional exogenous features.
    """

    def __init_subclass__(cls):
        """Set parameters for the predictor by inspecting ``update`` and ``predict``."""
        super().__init_subclass__()
        cls.set_params()

    @classmethod
    def set_params(cls):
        """Set the parameters that are accepted by the ``update`` and ``predict`` methods.

        Detected parameters are used to validate default parameters provided at
        initialization. May be overridden by subclasses if different behavior is desired.
        """
        # TODO: consider removing this functionality and simply waiting for error on
        # application of prediction (if parameters are misspecified).

        # Set update parameters for the predictor
        update_sig = inspect.signature(cls.update)
        cls.params = [
            k
            for k, v in list(update_sig.parameters.items())[1:]
            if v.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and k not in ["state", "X", "Y", "X_train"]
        ]

        # Set parameters for the predict method
        predict_sig = inspect.signature(cls.predict)
        predict_params = [
            k
            for k, v in list(predict_sig.parameters.items())[1:]
            if v.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls.predict_params = predict_params

    def __init__(
        self,
        X,
        Y,
        horizon,
        *args,
        Z=None,
        score_mode=False,
        default_params=None,
        **kwargs,
    ):
        self.X = X
        self.Y = Y
        self.X_train = Lag(X, amount=horizon)
        self.args = args
        self.kwargs = kwargs
        self._score_mode = score_mode
        self._default_params = default_params or {}

        # Check that default_params are valid
        for param in self._default_params.keys():
            if param not in self.params:
                raise ValueError(f"Parameter '{param}' not recognized for predictor.")

        if Z is None:
            Z_kwarg = {}
            self._use_Z = False
        else:
            Z_kwarg = {"Z": Z}
            self._use_Z = True

        super().__init__(X, Y, state=MEMORY, **Z_kwarg)

    @property
    def score_mode(self):
        """Return whether score mode is enabled."""
        return self._score_mode

    @score_mode.setter
    def score_mode(self, value: bool):
        self._score_mode = value

    def set_score_mode(self):
        """Enable score mode."""
        self._score_mode = True

    def unset_score_mode(self):
        """Disable score mode."""
        self._score_mode = False

    def _get_value(self, v, data):
        if isinstance(v, Source):
            if v in data:
                return data[v]
            elif isinstance(v, Transformation):
                return v.apply(data)
        else:
            return v

    def _create(self, X, Y, Z):
        data = {X_INIT: X, Y_INIT: Y, Z_INIT: Z}
        # Construct init params using data if required
        args = []
        for arg in self.args:
            args.append(self._get_value(arg, data))
        kwargs = {}
        for k, v in self.kwargs.items():
            kwargs[k] = self._get_value(v, data)

        # Call create method with constructed params
        return self.create(*args, **kwargs)

    @abstractmethod
    def create(self, *args, **kwargs):
        """Initialize and return predictor state.

        This method is called during ``evaluate`` if the ``state`` is None. Subclasses may
        use the provided arguments to set up the predictor state based on data
        dimensions or other parameters.

        Parameters
        ----------
        *args
            Positional arguments resolved from the ``args`` provided to ``__init__``.
        **kwargs
            Keyword arguments resolved from the ``kwargs`` provided to ``__init__``.

        Returns
        -------
        state : object
            The initialized predictor state to be passed to ``update`` and ``predict``
            methods. This can be any object; subclasses define the state representation.

        Notes
        -----
        Arguments are automatically resolved by ``_create`` so that Source objects are
        evaluated to their values before being passed here.
        """
        pass

    @abstractmethod
    def update(self, state, X, Y, X_train, Z=None, **params) -> tuple:
        r"""Update predictor state using new data and return prediction.

        Subclasses should implement this to nake predictions and update the predictor
        state based on lagged training data and targets.

        Parameters
        ----------
        state : object
            Current predictor state from previous evaluation or initialization.
        X : object
            Current input features, to be used for prediction.
        Y : object
            Current target data, to be used for training and evaluation.
        X_train : object
            Input features lagged by the forecast horizon, to be used for training the
            model.
        Z : object
            Current exogenous input features, to be used for prediction and evaluation.
        **params : object
            Additional parameters specified via ``default_params`` or ``apply`` method.

        Returns
        -------
        prediction : object
            The prediction output
        state : object
            Updated predictor state for the next evaluation.

        Notes
        -----
        Fitting should be performed as: :math:`Y \sim X_{train}`
        """
        pass

    @abstractmethod
    def predict(self, state, X, Z=None, **params):
        """Make predictions without updating predictor state.

        Subclasses should implement this to make predictions using the current state.
        The implementation should be a pure function, and in particular leave the state
        unchanged.

        Parameters
        ----------
        state : object
            Current predictor state from previous evaluation.
        X : object
            Current input features (lagged values also used for training).
        Z : object
            Current exogenous features.
        **params : Any
            Additional parameters specified via ``default_params`` or ``apply`` method.

        Returns
        -------
        prediction : object
            Prediction(s) for the current data.

        Notes
        -----
        This method should not update internal state to enable prediction without
        learning.
        """
        pass

    def score(self, state, X, Y, prediction, Z=None, **params):
        """Compute evaluation metric on predictions.

        This method is optional and only called when ``score_mode`` is enabled. Subclasses
        may override to compute custom metrics on predictions.

        Parameters
        ----------
        state : object
            Current predictor state.
        X : object
            Current input features (lagged values also used for training).
        Y : object
            Current target data.
        prediction : object
            Prediction output from ``update`` or ``predict``.
        Z : object, optional
            Exogenous features if prpvoded
        **params : Any
            Additional parameters.

        Returns
        -------
        prediction : object
            Updated prediction or score object

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass and ``score_mode`` is enabled.
        """
        raise NotImplementedError("Score method not implemented for this predictor.")

    def evaluate(self, X, Y, update_predictor=True, state=None, Z=None, **params):
        """Evaluate the transformation with training or prediction mode.

        Initializes state if needed, then calls either ``update`` or ``predict`` depending
        on the ``update_predictor`` keyword. Optionally computes scores.

        Parameters
        ----------
        X : object
            Current input features.
        Y : object
            Target data.
        update_predictor : bool, default True
            If True, calls ``update`` to train and predict. If False, calls ``predict`` to
            make predictions without training.
        state : tuple or None, optional
            Previous predictor and lag buffer state from a prior evaluation. If None,
            initializes fresh.
        Z : object, optional
            Exogenous features.
        **params : Any
            Additional parameters for ``update`` or ``predict`` methods.

        Returns
        -------
        result : object
            Prediction output, optionally with scores if ``score_mode`` is enabled.
        state : tuple
            Updated (predictor_state, X_lag_state) for the next evaluation.
        """
        # Combine provided and default predictor parameters
        params = self._default_params | params

        if self._use_Z:
            params["Z"] = Z

        if state is None:
            # Create predictor state
            predictor_state = self._create(X, Y, Z)

            X_state = None

        else:
            predictor_state, X_state = state

        if update_predictor:

            # Get training data
            X_train, X_state = self.X_train.evaluate(X, X_state)
            result, predictor_state = self.update(
                predictor_state, X, Y, X_train, **params
            )

        else:
            result = self.predict(predictor_state, X, **params)

        if self.score_mode:
            result = self.score(predictor_state, X, Y, result, **params)

        # Return result and state
        return result, (predictor_state, X_state)

    @property
    def predictor(self):
        """Return the current predictor state if available."""
        if self.recursion_pars is None:
            return None
        else:
            return self.recursion_pars[self][0]


def _stack_results(forecasts: list[dict] | list[np.ndarray]) -> dict | np.ndarray:

    # Convert to dict of lists
    if isinstance(forecasts[0], dict):
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


class OnlinePrediction(Prediction):
    """Base class for row-by-row online prediction transformations.

    Extends ``Prediction`` to support incremental (row-by-row) updates for online
    forecasting. The ``update`` and ``predict`` methods iterate over data rows,
    calling ``online_update`` and ``online_predict`` for each observation.

    Parameters
    ----------
    Same as ``Prediction``.

    Methods
    -------
    online_update
        Update model with a single row and make prediction (abstract).
    online_predict
        Make prediction for a single row without updating (abstract).

    Notes
    -----
    To enable online updates, the class will treat X, Y and Z as ndarrays. Rows are
    looped over in the ``update`` and ``predict`` methods and NaN values are ignored for
    training. Content of the arrays is not assumed to be numeric and may in principle
    be used for any type of data.

    See Also
    --------
    Prediction : Parent class for batch-mode prediction.
    """

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.set_params()

    @classmethod
    def set_params(cls):
        """Detect parameters from ``online_update`` and ``online_predict`` signatures.

        Introspects both methods to build combined parameter lists for validation.
        Separates update-specific from predict-specific parameters.
        """
        cls._set_predict_params()
        cls._set_update_model_params()
        cls.params = cls.predict_params + cls.update_model_params

    @classmethod
    def _set_predict_params(cls):
        predict_sig = inspect.signature(cls.online_predict)
        predict_params = [
            k
            for k, v in list(predict_sig.parameters.items())[1:]
            if v.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls.predict_params = predict_params

    @classmethod
    def _set_update_model_params(cls):
        update_model_sig = inspect.signature(cls.online_update)
        update_model_params = [
            k
            for k, v in list(update_model_sig.parameters.items())[1:]
            if v.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls.update_model_params = update_model_params

    @classmethod
    def _convert_arrays(cls, X, Y=None, X_train=None, Z=None):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(Y, np.ndarray) and Y is not None:
            Y = np.asarray(Y)
        if not isinstance(X_train, np.ndarray) and X_train is not None:
            X_train = np.asarray(X_train)
        if not isinstance(Z, np.ndarray) and Z is not None:
            Z = np.asarray(Z)

        # Ensure 2D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y is not None and Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if X_train is not None and X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        return X, Y, X_train, Z

    def update(
        self,
        state,
        X: np.ndarray,
        Y: np.ndarray,
        X_train: np.ndarray,
        Z: np.ndarray = None,
        **params,
    ):
        """Update predictor with batch data by processing rows incrementally.

        Iterates over each row, calling ``online_update`` when data is valid (no NaN
        values) and ``online_predict`` otherwise. Distributes parameters to appropriate
        methods.

        Parameters
        ----------
        state : object
            Current predictor state.
        X : ndarray of shape (n_obs, ...)
            Input features.
        Y : ndarray of shape (n_obs, ...)
            Target data.
        X_train : ndarray of shape (n_obs, ...)
            Input features lagged by the forecast horizon, used for training.
        Z : ndarray of shape (n_obs, n_exog), optional
            Exogenous features.
        **params : object
            Additional parameters distributed to update_params and predict_params.

        Returns
        -------
        forecasts : dict or ndarray
            Stacked predictions from all rows. If ``online_update`` returns dicts,
            result is a dict of arrays; otherwise a single array.
        state : object
            Updated predictor state after processing all rows.

        Notes
        -----
        Rows with NaN values in Y or X_train are skipped for updates but still get
        predictions via ``online_predict``.
        """
        X, Y, X_train, Z = self._convert_arrays(X, Y, X_train, Z)

        # Distribute params
        update_params = {
            k: v for k, v in params.items() if k in self.update_model_params
        }
        predict_params = {k: v for k, v in params.items() if k in self.predict_params}

        n = X.shape[0]
        forecasts = []

        # Loop over each row of data
        for i in range(n):

            x = X[i]
            y = Y[i]
            x_train = X_train[i]

            if self._use_Z:
                update_params["z_i"] = Z[i]
                predict_params["z_i"] = Z[i]

            y_ready = not np.isnan(y).any()
            x_train_ready = not np.isnan(x_train).any()

            # Only update if data is valid
            if x_train_ready and y_ready:
                forecast, state = self.online_update(
                    state, x, y, x_train, **update_params
                )
            else:
                forecast = self.online_predict(state, x, **predict_params)

            forecasts.append(forecast)

        forecasts = _stack_results(forecasts)

        return forecasts, state

    def predict(self, state, X: np.ndarray, Z=None, **params):
        """Make predictions for batch data by processing rows incrementally.

        Iterates over each row, calling ``online_predict`` for each without updating
        the predictor state.

        Parameters
        ----------
        state : object
            Current predictor state.
        X : ndarray of shape (n_samples, ...)
            Input features.
        Z : ndarray of shape (n_samples, ...), optional
            Exogenous features.
        **params : object
            Additional parameters for ``online_predict``.

        Returns
        -------
        forecasts : dict or ndarray
            Stacked predictions from all rows.

        Raises
        ------
        ValueError
            If any parameter key is not recognized for prediction.
        """
        # Check parameters
        for k in params.keys():
            if k not in self.predict_params:
                raise ValueError(f"Parameter '{k}' not recognized for prediction.")

        X, _, _, Z = self._convert_arrays(X, Z=Z)

        # Predict multiple rows.
        n = X.shape[0]
        forecasts = []

        for i in range(n):
            x = X[i]
            if self._use_Z:
                params["z"] = Z[i]
            forecast = self.online_predict(state, x, **params)
            forecasts.append(forecast)

        forecasts = _stack_results(forecasts)

        return forecasts

    @abstractmethod
    def online_update(self, state, x_i, y_i, x_train_i, z_i=None, **params) -> tuple:
        r"""Update model with a single data row and return prediction.

        Subclasses must implement this method to perform incremental updates with
        individual observations.

        Parameters
        ----------
        state : object
            Current predictor state.
        x_i : object
            Single entry of input features.
        y_i : object
            Single entry of target values.
        x_train_i : object
            Single entry of lagged training features.
        z_i : object, optional
            Single entry of exogenous features.
        **params : object
            Additional update parameters.

        Returns
        -------
        prediction : dict or ndarray
            Prediction for the current entry x_i.
        state : object
            Updated predictor state.

        Notes
        -----
        Fitting should use the lagged features: :math:`y_i \sim x_{train,i}`
        """

    @abstractmethod
    def online_predict(self, state, x_i, z_i=None, **params):
        # Predict a single row.
        pass


def evaluate_score(Y_hat, Y, burn_in=0, remove_nan=True, scorefun=rmse):
    """Evaluate a score function on the residuals between predictions and targets.

    Parameters
    ----------
    Y_hat : ndarray
        Predicted values.
    Y : ndarray
        True target values.
    burn_in : int, default 0
        Number of initial observations to discard from evaluation.
    remove_nan : bool, default True
        If True, removes any rows with NaN values from the residuals before scoring.
    scorefun : function, default rmse
        Function to compute the score on the residuals. Should take an array of
        residuals and return a scalar score.
    """
    resid = Y - Y_hat

    if burn_in > 0:
        resid = resid[burn_in:]
    if remove_nan:
        mask = ~np.isnan(resid).any(axis=1)
        resid = resid[mask]

    return scorefun(resid)


class WLS(Prediction):
    """Weighted least squares prediction.

    This predictor assumes a linear models and uses weighted least squares to fit
    parameters and make predictions.

    Parameters
    ----------
    X : Source
        Input data source. Output should be compatible with the Lag transformation.
    Y : Source
        Target data source.
    horizon : int
        Forecast horizon in time steps.
    """

    def __init__(self, X, Y, horizon):
        super().__init__(X, Y, horizon, n=DIM_X, m=DIM_Y)

    def create(self, n, m):
        """Initialise the (n, m) parameter array."""
        # Initialize state as parameters of WLS model (theta)
        return np.zeros((n, m))

    def update(
        self,
        state,
        X: np.ndarray,
        Y: np.ndarray,
        X_train: np.ndarray,
        W: np.ndarray = None,
    ):
        r"""Fit weighted least squares model and return predictions.

        Parameters
        ----------
        state : object
            Not used.
        X : ndarray of shape (n_obs, n_features)
            Current input data for prediction.
        Y : ndarray of shape (n_obs, n_targets)
            Current target data for model fitting.
        X_train : ndarray of shape (n_obs, n_features)
            Lagged input data at the forecast horizon, used for fitting the model.
        W : ndarray of shape (n_train, n_train), optional
            Positive-definite weight matrix for WLS. If None, use identity matrix
            (ordinary least squares). Default is None.

        Returns
        -------
        prediction : dict
            Dictionary with key "mean" containing predictions of shape
            (n_obs, n_targets).
        theta : ndarray of shape (n_features, n_targets)
            Fitted model parameters (weights).
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(Y, np.ndarray):
            Y = np.asarray(Y)
        if not isinstance(X_train, np.ndarray):
            X_train = np.asarray(X_train)

        # Fit WLS model
        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y).any(axis=1)
        X_fit = X_train[mask]
        Y_fit = Y[mask]

        if W is None:
            W = np.eye(
                X_fit.shape[0]
            )  # Default to identity weights if W is not provided
        else:
            W = W[np.ix_(mask, mask)]  # Subset W to match the filtered data

        XtW = X_fit.T @ W
        theta = np.linalg.solve(XtW @ X_fit, XtW @ Y_fit)

        pred = self.predict(theta, X)

        return pred, theta

    def predict(self, state, X: np.ndarray):
        """Apply the parameters to make predictions."""
        # Predict
        pred = X @ state
        return {"mean": pred}


class RRRPredictor:
    r"""Recursive ridge regression predictor with variance estimation.

    Implements a recursive formulation of ridge regression with exponential
    forgetting and optional covariance estimation for prediction uncertainty.

    Parameters
    ----------
    n : int
        Dimension of input features.
    m : int
        Dimension of target variables.
    horizon : int
        Forecast horizon in time steps (used for prediction error variance estimation).
    burn_in : int, default 1
        Number of initial observations to skip before estimating prediction error
        variance.
    init_K : float, default 0
        Initial value for the diagonal of the :math:`X^T P X` accumulator matrix K.
        Higher values add regularization at initialization.
    track_memory : bool, default False
        If True, tracks exact effective sample size during burn-in for variance
        estimation. If False, assumes saturated memory.
    combine_variance : bool, default True
        Not currently used.
    full_cov : bool, default True
        If True, estimates full covariance matrix for prediction errors. If False,
        estimates only diagonal variances.
    center_cov : bool, default False
        If True, centers the residuals before computing covariance.
    mem : float, default 0.99
        Forgetting factor in [0, 1] for exponential weighting. Higher values retain
        more historical data; lower values adapt faster to recent observations.

    Attributes
    ----------
    K : ndarray of shape (n, n)
        Accumulated (weighted) :math:`X_{train}^T P X_{train}` matrix.
    L : ndarray of shape (n, m)
        Accumulated (weighted) :math:`X_{train}^T P Y` matrix.
    theta : ndarray of shape (n, m)
        Current parameter estimates.
    psi : ndarray of shape (n, n)
        Uncertainty matrix :math:`K^{-1} H K^{-1}` for parameter variance computation.
    V : ndarray of shape (m, m) or (m,)
        Estimated prediction error covariance (full matrix or diagonal).
    H : ndarray of shape (n, n)
        Accumulated squared forgetting factor weighted :math:`X_{train}^T P X_{train}`.
    Y_hat : CircularBuffer
        Circular buffer storing recent predictions for residual-based variance estimation.

    Methods
    -------
    online_update
        Update model parameters with a single observation and return prediction.
    online_predict
        Make prediction for a single observation without updating.
    get_var_vec_theta
        Compute vectorized parameter covariance matrix.
    get_model_params
        Return current model parameters theta.

    Notes
    -----
    The recursive update solves the regularized normal equations at each step:

    .. math::

        \theta_t = (K_t + Q)^{-1} (L_t + Q \theta_0)

    where :math:`K_t = \lambda K_{t-1} + x_{train,t} x_{train,t}^T` and
    :math:`L_t = \lambda L_{t-1} + x_{train,t} y_t^T``, with :math:``\lambda` being
    the forgetting factor (``mem``).

    Mean point predictions are computed as:

        .. math::
            
            \hat{y} = x^T \theta

    Prediction uncertainty accounts for both parameter and error variance:

        .. math::
            
            \text{Cov}[\hat{y} - y] = V (1 + x^T \psi x)

    where V is estimated from recent prediction residuals and
    :math:`\psi_t = K_t^{-1} H_t K_t^{-1}` captures the parameter uncertainty
    contribution to prediction error variance.

    After burn-in, V is estimated from :math:`(y_t - \hat{y}_{t-h})` where
    :math:`\hat{y}_{t-h}` is the h-step-ahead forecast made at time t-h.

    The parameters posterior under the ridge model is matrix normal with mean 
    :math:`\theta_t`` and covariance :math:``V \otimes \psi`.
    """

    def __init__(
        self,
        n,
        m,
        horizon,
        burn_in=1,
        init_K=0,
        track_memory=False,
        combine_variance=True,
        full_cov=True,
        center_cov=False,
        mem=0.99,
    ):
        self.K = np.eye(n) * init_K
        self.L = np.zeros((n, m))
        self.burn_in = burn_in
        self._n_updates = 0
        self.H = np.zeros((n, n))
        self.theta = np.full((n, m), np.nan)
        self.psi = np.full((n, n), np.nan)
        self.V = np.full((m, m), np.nan)
        self.combine_variance = combine_variance
        self.n, self.m = n, m
        self._forgetting_var_state = None
        self._full_cov = full_cov
        self.Y_hat = CircularBuffer(horizon, m)

        self._forgetting_var = ForgettingVariance(
            forgetting=mem,
            track_memory=track_memory,
            covariance=full_cov,
            center=center_cov,
        )

    def online_update(
        self,
        x_i,
        y_i,
        x_train_i,
        Q: np.ndarray | float = None,
        theta0: np.ndarray = None,
        V: np.ndarray = None,
        estimate_V=True,
        mem=None,
        return_var_theta=False,
    ):
        r"""Update model with single observation and return prediction.
        
        Performs recursive ridge regression update with exponential forgetting and
        updates prediction error variance estimate if enabled.
        
        Parameters
        ----------
        x_i : ndarray of shape (n_features,)
            Current input features for prediction.
        y_i : ndarray of shape (n_targets,)
            Current target observation for model update.
        x_train_i : ndarray of shape (n_features,)
            Lagged input features at the forecast horizon for model fitting.
        Q : ndarray of shape (n_features, n_features) or float, optional
            Ridge regularization matrix. If float, uses :math:`Q = q I`. If None, no
            regularization is applied. Must be symmetric if array. Default is None.
        theta0 : ndarray of shape (n_features, n_targets), optional
            Prior mean for parameters (regularization target). If None, uses zero.
            Default is None.
        V : ndarray of shape (n_targets, n_targets) or (n_targets,), optional
            Fixed prediction error covariance. If provided, overrides internal estimate.
            Default is None.
        estimate_V : bool, default True
            If True, updates prediction error covariance V from residuals after burn-in.
        mem : float, optional
            Forgetting factor for this update. If None, uses value from initialization.
            Must be in [0, 1]. Default is None.
        return_var_theta : bool, default False
            If True, includes parameter covariance in the returned prediction dict.
        
        Returns
        -------
        result : dict
            Prediction dictionary with keys:
            
            - "mean" : ndarray of shape (n_targets,) - Point prediction
            - "cov" : ndarray - Prediction error covariance/variance
            - "cov_theta" : ndarray (optional) - Parameter covariance if requested
    
        Raises
        ------
        ValueError
            If mem is not in [0, 1] or if Q is not symmetric.
        """
        if mem < 0 or mem > 1:
            raise ValueError("Memory must be between 0 and 1.")

        n, m = self.n, self.m

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

        if V is not None:
            self.V = V

        x_outer = np.outer(x_train_i, x_train_i)

        if self.K is None:
            self.K = x_outer
        else:
            self.K = mem * self.K + x_outer

        if self.L is None:
            self.L = np.outer(x_train_i, y_i)
        else:
            self.L = mem * self.L + np.outer(x_train_i, y_i)

        KpQ = self.K + Q
        LpQtheta = self.L + Q @ theta0

        self.theta = np.linalg.solve(KpQ, LpQtheta)

        # Update estimate of variance
        self.H = mem**2 * self.H + x_outer

        temp1 = np.linalg.solve(KpQ, self.H)

        self.psi = np.linalg.solve(KpQ, temp1.T).T  # K^-1 H K^-1^T

        if estimate_V and self._n_updates >= self.burn_in:

            y_i_hat = self.Y_hat.get(1)  # Get oldest prediction
            resid = np.atleast_2d(y_i - y_i_hat)
            self.V = self._forgetting_var(resid, track_state=True, forgetting=mem)[0]
            if not self._forgetting_var.covariance:
                self.V = np.diag(self.V)

        # Make prediction
        result = self.online_predict(x_i, V=V, return_var_theta=return_var_theta)

        # Store prediction for future variance estimation
        self.Y_hat.append(result["mean"])

        self._n_updates += 1

        return result

        # TODO: use the Sherman-Morrison formula for more efficient updates

    def online_predict(self, x: np.ndarray, V=None, return_var_theta=False):
        r"""Make prediction without updating model state.

        Computes point prediction and uncertainty for a single entry using current model
        parameters.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input features for prediction.
        V : ndarray of shape (n_targets, n_targets), optional
            Prediction error covariance. If None, uses internal estimate. Default is 
            None.
        return_var_theta : bool, default False
            If True, includes parameter covariance in the returned prediction dict.

        Returns
        -------
        result : dict
            Prediction dictionary with keys:
            - "mean" : ndarray of shape (n_targets,) - Point mean prediction
            - "cov" : ndarray - Prediction error covariance
            - "cov_theta" : ndarray (optional) - Parameter covariance if requested
        """
        result = {}

        if V is None:
            V = self.V

        result["mean"] = x.T @ self.theta

        # Compute covariance of prediction error
        var_pred_err = self.V * (1 + x.T @ self.psi @ x)

        if not self._full_cov:
            var_pred_err = np.diag(var_pred_err)

        result["cov"] = var_pred_err

        # Compute variance of theta
        if return_var_theta:
            result["cov_theta"] = self.get_var_vec_theta(V)

        return result

    def get_var_vec_theta(self, V: np.ndarray = None):
        """Return the covariance of the vectorised parameters."""
        if V is None:
            V = self.V

        # Compute the variance of theta
        var_theta = np.kron(V, self.psi)

        return var_theta

class RRR(OnlinePrediction):
    """Online recursive ridge regression transformation.

    Thin wrapper around ``RRRPredictor`` that enables its use as a Prediction.

    See Also
    --------
    RRRPredictor : Core recursive ridge implementation and uncertainty estimates.
    OnlinePrediction : Row-wise update/predict orchestration.

    Parameters
    ----------
    X : Source
        Input data source. Output should be compatible with the Lag transformation.
    Y : Source
        Target data source.
    horizon : int
        Forecast horizon in time steps (used for prediction error variance estimation).
    scorefun : function, default rmse
        Function to compute the score on the residuals. Should take an array of
        residuals and return a scalar score.
    default_params : dict, optional
        Default parameters for the predictor. Can include any parameters accepted by
        ``RRRPredictor.online_update`` and ``RRRPredictor.online_predict``.
    Otherwise same as ``RRRPredictor``.
    """
    
    # TODO: consider including batch functionality into this class
    def __init__(
        self,
        X,
        Y,
        horizon,
        burn_in=1,
        init_K=0,
        track_memory=False,
        combine_variance=True,
        full_cov=True,
        center_cov=False,
        scorefun=rmse,
        default_params=None,
    ):
        self.horizon = horizon
        self.Y_hat = Lag(self, amount=horizon)
        self.scorefun = scorefun
        super().__init__(
            X,
            Y,
            horizon,
            n=DIM_X,
            m=DIM_Y,
            burn_in=burn_in,
            init_K=init_K,
            track_memory=track_memory,
            combine_variance=combine_variance,
            full_cov=full_cov,
            center_cov=center_cov,
            default_params=default_params,
        )

    def create(
        self,
        n,
        m,
        burn_in,
        init_K,
        track_memory,
        combine_variance,
        full_cov,
        center_cov,
        **default_params,
    ):
        """Create amd return ``RRRPredictor``."""
        return RRRPredictor(
            n,
            m,
            self.horizon,
            burn_in,
            init_K,
            track_memory,
            combine_variance,
            full_cov,
            center_cov,
        )

    def online_update(
        self,
        state: RRRPredictor,
        x_i,
        y_i,
        x_train_i,
        Q: np.ndarray | float = None,
        theta0: np.ndarray = None,
        V: np.ndarray = None,
        estimate_V=True,
        mem=0.99,
        return_var_theta=False,
    ):
        """Delegate single-step update to ``RRRPredictor.online_update``.

        Parameters
        ----------
        state : RRRPredictor
            The recerusive ridge predictor to use and update
        Otherwise same as ``RRRPredictor`` ``online_update``.

        Returns
        -------
        result : dict
            Prediction dictionary produced by ``RRRPredictor``.
        state : RRRPredictor
            Updated predictor instance.
        """
        result = state.online_update(
            x_i,
            y_i,
            x_train_i=x_train_i,
            Q=Q,
            theta0=theta0,
            V=V,
            estimate_V=estimate_V,
            mem=mem,
            return_var_theta=return_var_theta,
        )
        return result, state
    def online_predict(self, state: RRRPredictor, x_i, V=None, return_var_theta=False):
        """Delegate single-step prediction to ``RRRPredictor.online_predict``.

        Parameters
        ----------
        state : RRRPredictor
            The recerusive ridge predictor to use for prediction.
        Otherwise same as ``RRRPredictor`` ``online_predict``.

        Returns
        -------
        Same as ``RRRPredictor.online_predict``.
        """
        return state.online_predict(x_i, V=V, return_var_theta=return_var_theta)

    def score(self, state: RRRPredictor, X, Y, prediction, **params):
        """Compute forecast score using lagged predictions from predictor memory.

        Uses ``evaluate_score`` on aligned historical predictions stored in
        ``state.Y_hat``, then adds the scalar score to ``prediction["score"]``.
        """
        n = len(Y)

        # Get old predictions from state
        Y_hat = state.Y_hat.get(n)

        # Overwrite last n-horizon entries with new predictions (discard )
        Y_hat[self.horizon :] = prediction["mean"][: -self.horizon]

        # Evaluate score and update prediction
        prediction["score"] = evaluate_score(
            Y_hat, Y, burn_in=state.burn_in, remove_nan=True, scorefun=self.scorefun
        )
        return prediction


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

        super().__init__(data=data, memory=MEMORY)

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

        # Form output
        if self.skip_duplicates:

            # Get mask to select every max(lag)'th row, starting from an offset
            mask = np.arange(self.max_shift - offset, t, self.max_shift + 1)

            # Update offset
            offset = (offset + t) % (self.max_shift + 1)

            # Fill non-masked rows with NaN
            masked = np.full_like(result, np.nan)
            masked[mask, :] = result[mask, :]

            return masked, (all_data[-self.max_shift :], offset)
        
        else:
            return result, (all_data[-self.max_shift :], None)


class ARX(OnlinePrediction):
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
        burn_in:int=1,
        init_K=0,
        track_memory=False,
        combine_variance=True,
        full_cov=True,
        scorefun=rmse,
        default_params=None,
    ):
        self.horizon = horizon
        self._lag_indices = np.array(order)-1 if isinstance(order, list) else np.arange(order)
        self.p = max(self._lag_indices) + 1

        # Ensure endog is (t, 1)
        endog = Apply(lambda x: x.reshape(x.shape[0], 1) if x.ndim == 1 else x, endog)

        # Make regression model for 1-step forecasts
        X = BackShift(
            list(range(self.p)), endog
        )  # ordered as (y_t, y_t-1, ..., y_t-p+1)

        if exog is not None:
            # Include the first horizon of exogenous features
            X = Apply(lambda x, y: np.hstack((x, y)), X, exog[:, 0])

        self.scorefun = scorefun

        super().__init__(
            X,
            endog,
            1,
            Z=exog,
            n=DIM_X,
            m=DIM_Y,
            burn_in=burn_in,
            init_K=init_K,
            track_memory=track_memory,
            combine_variance=combine_variance,
            full_cov=full_cov,
            default_params=default_params,
        )

    def create(
        self,
        n,
        m,
        burn_in,
        init_K,
        track_memory,
        combine_variance,
        full_cov
    ):
        """Initialize RRR predictor state for 1-step predictions."""
        n_lags = len(self._lag_indices)  # Number of AR lags
        n_feat = n-self.p # Number of exogenous features
        n_reg = n_lags + n_feat  # Total number of regression features

        # Build indices for selecting input features, i.e. lags and features
        self._exog_indices = np.arange(self.p, self.p+n_feat)
        self._indices = np.concat([self._lag_indices, self._exog_indices])
        return RRRPredictor(
            n_reg,
            m,
            1,
            burn_in,
            init_K,
            track_memory,
            combine_variance,
            full_cov,
        )

    def online_update(
        self,
        state: RRRPredictor,
        x_i,
        y_i,
        x_train_i,
        z_i = None,
        Q: np.ndarray | float = None,
        theta0: np.ndarray = None,
        V: np.ndarray = None,
        estimate_V=True,
        mem=0.99,
        return_var_theta=False,
    ):
        """Update model and compute h-step-ahead predictions via forward recursion.

        Delegates one-step update to ``RRRPredictor``, then iterates forward h steps
        using AR coefficients and updated endogenous history to forecast h horizons.
        """
        state.online_update(
            x_i[self._indices],
            y_i,
            x_train_i=x_train_i[self._indices],
            Q=Q,
            theta0=theta0,
            V=V,
            estimate_V=estimate_V,
            mem=mem,
            return_var_theta=False,
        )
        return (
            self.online_predict(
                state, x_i, z_i, V=V, return_var_theta=return_var_theta
            ),
            state,
        )

    def online_predict(
        self, state: RRRPredictor, x_i, z_i=None, V=None, return_var_theta=False
    ):
        """Compute h-step-ahead predictions and variances.
        
        Iterates through horizons, predict mean and variance using accumulated
        weight matrices from AR coefficient propagation.

        Parameters
        ----------
        state : RRRPredictor
            The predictor state.
        x_i : ndarray of shape (p+m,)
            Current endogenous history for prediction (lags) and exogenous features for 
            horizon 0. Should be ordered as (y_t-p+1, ..., y_t-1, y_t, z_1).
        z_i : ndarray of shape (h, m) or None
            Current exogenous variables for prediction (lags). First dimension should
            match prediction horizon. If None, exogenous features are not used.
        V : ndarray of shape (p, p) or None
            Variance-covariance matrix of the prediction errors.
        return_var_theta : bool
            Whether to return the variance of the regression coefficients.


        """
        result = state.online_predict(x_i[self._indices], V=V, return_var_theta=return_var_theta)

        theta = state.theta.flatten()

        arx_pred = np.full(self.horizon, np.nan)
        arx_var = np.full(
            self.horizon, np.nan
        )  # TODO: consider computing full covariance between horizons rather than just variances

        weights = np.zeros(max(self.horizon, self.p))  # Initialize AR weights for variance computation
        weights[0] = 1

        ar_params = theta[: len(self._lag_indices)]  # AR coefficients from theta

        endog = x_i[: self.p]  # Full endogenous history

        if V is None:
            V = state.V

        # Modify result to include predictions for all horizons
        for h in range(self.horizon):

            reg_i = endog[self._lag_indices] # Select only the lags specified by order

            # Combine AR with exogenous features for horizon h
            if z_i is not None:
                reg_i = np.hstack((reg_i, z_i[h]))

            # Predict next step ahead using theta and 
            arx_pred[h] = (reg_i.T @ theta).item()

            # Compute variance, using weights so far.
            arx_var[h] = (V * (weights @ weights)).item()

            # Push weights forward for next prediction. Use only the specified lags
            new_weight = weights[self._lag_indices] @ ar_params

            # Prepend new weight
            weights[1:] = weights[:-1]
            weights[0] = new_weight

            # Update endogenous history
            endog[1:] = endog[:-1]
            endog[0] = arx_pred[h]

        result["mean"] = arx_pred
        result["var"] = arx_var

        return result

    def score(self, state: RRRPredictor, X, Y, prediction, **predictor_params):
        """Compute forecast score using lagged predictions from predictor memory."""
        n = len(Y)

        # Get old predictions from state
        Y_hat = state.Y_hat.get(n)

        # Overwrite last n-horizon entries with new predictions (discard )
        Y_hat[self.horizon :] = prediction["mean"][: -self.horizon]

        # Evaluate score and update prediction
        prediction["score"] = evaluate_score(
            Y_hat, Y, burn_in=state.burn_in, remove_nan=True, scorefun=self.scorefun
        )
        return prediction
