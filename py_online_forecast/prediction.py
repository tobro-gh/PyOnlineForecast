from .core import *
from .features import *
from abc import abstractmethod

def rmse(x):
    return np.sqrt(np.mean(x**2))

def stack_results(forecasts: list[dict] | list[np.ndarray]) -> dict | np.ndarray:

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


class Prediction(Transformation):

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.set_params()

    @classmethod
    def set_params(cls):
        # Set update parameters for the predictor
        update_sig = inspect.signature(cls.update)
        cls.params = [k for k, v in list(update_sig.parameters.items())[1:] if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) and k not in ["state", "X", "Y", "X_train"]]

        # Set parameters for the predict method
        predict_sig = inspect.signature(cls.predict)
        predict_params = [
            k for k, v in list(predict_sig.parameters.items())[1:]
            if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls.predict_params = predict_params

    def __init__(self, X, Y, horizon, *args, Z = None, score_mode = False, default_params = None, **kwargs):
        """
        Y_t = f(X_{t-horizon}) => state_t
        hat{Y}_{t+h} = g(state_t, Z_t)

        Generic transformation providing a base class for predictors.
        Parameters:
        - X: input data (Source with output that can be used with Lag)
        - Y: target (Source, not lagged)
        - Z: additional features (Source, not lagged) that may also be used for prediction.
        - horizon: forecast horizon (int)
        - default_params: dict of default predictor parameters to be used in the update and predict methods. These can be overridden by providing parameters in the evaluate method.
        - args, kwargs: arguments to be used by the create method to initialise the predictor.
        """
        self.X = X
        self.Y = Y
        self.X_train = Lag(X, amount = horizon)
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

        super().__init__(X, Y, state = MEMORY, **Z_kwarg)
    
    @property
    def score_mode(self):
        return self._score_mode

    @score_mode.setter
    def score_mode(self, value: bool):
        self._score_mode = value

    def set_score_mode(self):
        self._score_mode = True
    
    def unset_score_mode(self):
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
        """
        Method to create the predictor state. Should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def update(self, state, X, Y, X_train, Z = None, **params) -> tuple:
        """
        Method to update the predictor with new data. Should be implemented by subclasses. Return value should be the prediction for the current time step and the updated state of the predictor.
        Fitting should be done as Y~X_train.
        """
        pass

    @abstractmethod
    def predict(self, state, X, Z = None, **params):
        """
        Method to make predictions. Should be implemented by subclasses.
        """
        pass

    def score(self, state, X, Y, prediction, Z = None, **params):
        raise NotImplementedError("Score method not implemented for this predictor.")
    

    def evaluate(self, X, Y, update_predictor = True, state = None, Z = None, **params):

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
            result, predictor_state = self.update(predictor_state, X, Y, X_train, **params)

        else:
            result = self.predict(predictor_state, X, **params)

        if self.score_mode:
            result = self.score(predictor_state, X, Y, result, **params)

        # Return result and state
        return result, (predictor_state, X_state)

    @property
    def predictor(self):
        if self.recursion_pars is None:
            return None
        else:
            return self.recursion_pars[self][0]


class OnlinePrediction(Prediction):

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.set_params()

    @classmethod
    def set_params(cls):
        cls.set_predict_params()
        cls.set_update_model_params()
        cls.params = cls.predict_params + cls.update_model_params

    @classmethod
    def set_predict_params(cls):
        predict_sig = inspect.signature(cls.online_predict)
        predict_params =  [
            k for k, v in list(predict_sig.parameters.items())[1:]
            if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls.predict_params = predict_params

    @classmethod
    def set_update_model_params(cls):
        update_model_sig = inspect.signature(cls.online_update)
        update_model_params = [
            k for k, v in list(update_model_sig.parameters.items())[1:]
            if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls.update_model_params = update_model_params

    @classmethod
    def convert_arrays(cls, X, Y = None, X_train = None, Z = None):
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

    def update(self, state, X: np.ndarray, Y: np.ndarray, X_train: np.ndarray, Z: np.ndarray = None, **params):
        X, Y, X_train, Z = self.convert_arrays(X, Y, X_train, Z)
            
        # Distribute params
        update_params = {k: v for k, v in params.items() if k in self.update_model_params}
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
                forecast, state = self.online_update(state, x, y, x_train, **update_params)
            else:
                forecast = self.online_predict(state, x, **predict_params)
 
            forecasts.append(forecast)

        forecasts = stack_results(forecasts)

        return forecasts, state

    def predict(self, state, X: np.ndarray, Z = None, **params):
        # Check parameters
        for k in params.keys():
            if k not in self.predict_params:
                raise ValueError(f"Parameter '{k}' not recognized for prediction.")

        X, _, _, Z = self.convert_arrays(X, Z=Z)

        # Predict multiple rows.
        n = X.shape[0]
        forecasts = []

        for i in range(n):
            x = X[i]
            if self._use_Z:
                params["z"] = Z[i]
            forecast = self.online_predict(state, x, **params)
            forecasts.append(forecast)

        forecasts = stack_results(forecasts)

        return forecasts

    @abstractmethod
    def online_update(self, state, x_i, y_i, x_train_i, z_i = None, **params) -> tuple:
        """
        Update model with new data rows x_train_i, y_i, and make prediction for x_i. Should return the prediction for x_i and the updated state of the model.
        """

    @abstractmethod
    def online_predict(self, state, x_i, z_i = None, **params):
        # Predict a single row.
        pass

def evaluate_score(Y_hat, Y, burn_in = 0, remove_nan = True, scorefun = rmse):
    resid = Y - Y_hat

    if burn_in > 0:
        resid = resid[burn_in:]
    if remove_nan:
        mask = ~np.isnan(resid).any(axis=1)
        resid = resid[mask]

    return scorefun(resid)
    
class WLS(Prediction):

    def __init__(self, X, Y, horizon, format_as_Y = True):
        self.format_as_Y = format_as_Y
        super().__init__(X, Y, horizon, n = DIM_X, m = DIM_Y)

    def create(self, n, m):
        # Initialize state as parameters of WLS model (theta)
        return np.zeros((n, m))

    def update(self, state, X: np.ndarray, Y: np.ndarray, X_train: np.ndarray, W: np.ndarray = None):

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
            W = np.eye(X_fit.shape[0])  # Default to identity weights if W is not provided
        else:
            W = W[np.ix_(mask, mask)]  # Subset W to match the filtered data

        XtW = X_fit.T @ W
        theta = np.linalg.solve(XtW @ X_fit, XtW @ Y_fit)

        pred = self.predict(theta, X)

        return pred, theta
    
    def predict(self, state, X: np.ndarray):
        # Predict
        pred = X @ state
        return {"mean": pred}

class RRRPredictor:
    
    def __init__(self, n, m, horizon, burn_in = 1, tilde_k_init_val = 0, track_memory = False, combine_variance = True, full_cov = True, center_cov = False, mem = 0.99):
        self.tilde_K = np.eye(n) * tilde_k_init_val
        self.tilde_R = np.zeros((n,m))
        self.burn_in = burn_in
        self._n_updates = 0
        self.kappa = np.zeros((n, n))
        self.theta = np.full((n, m), np.nan)
        self.inner_var_theta = np.full((n, n), np.nan)
        self.V = np.full((m, m), np.nan)
        self.combine_variance = combine_variance
        self.n, self.m = n, m
        self._forgetting_var_state = None
        self._full_cov = full_cov
        self.Y_hat = CircularBuffer(horizon, m)

        self._forgetting_var = ForgettingVariance(forgetting=mem, track_memory=track_memory, covariance=full_cov, center=center_cov)

    def online_update(self, x_i, y_i, x_train_i, Q: np.ndarray | float = None, theta_tilde: np.ndarray = None, V: np.ndarray = None, estimate_V = True, mem = None, return_var_theta = False):        

        if mem < 0 or mem > 1:
            raise ValueError("Memory must be between 0 and 1.")

        n, m = self.n, self.m

        if theta_tilde is None:
            theta_tilde = np.zeros((n, m))
        
        if Q is None:
            Q = np.zeros((n, n))
        elif isinstance(Q, (float, int)):
            Q = np.eye(n) * Q

        elif not np.allclose(Q, Q.T):
            raise ValueError("Q must be symmetric.")

        if theta_tilde is None:
            theta_tilde = np.zeros((n, m))

        if not V is None:
            self.V = V

        x_outer = np.outer(x_train_i, x_train_i)

        if self.tilde_K is None:
            self.tilde_K = x_outer
        else:
            self.tilde_K = mem*self.tilde_K + x_outer

        if self.tilde_R is None:
            self.tilde_R = np.outer(x_train_i, y_i)
        else:
            self.tilde_R = mem*self.tilde_R + np.outer(x_train_i, y_i)

        K = self.tilde_K + Q
        R = self.tilde_R + Q @ theta_tilde

        self.theta = np.linalg.solve(K, R)

        # Update estimate of variance
        self.kappa = mem**2*self.kappa + x_outer

        temp1 = np.linalg.solve(K, self.kappa)
    
        self.inner_var_theta = np.linalg.solve(K, temp1.T).T # K^-1 kappa K^-1^T

        if estimate_V and self._n_updates >= self.burn_in:

            y_i_hat = self.Y_hat.get(1) # Get oldest prediction
            resid = np.atleast_2d(y_i - y_i_hat)
            self.V = self._forgetting_var(resid, track_state = True, forgetting = mem)[0]
            if not self._forgetting_var.covariance:
                self.V = np.diag(self.V)

        # Make prediction
        result = self.online_predict(x_i, V = V, return_var_theta=return_var_theta)

        # Store prediction for future variance estimation
        self.Y_hat.append(result['mean'])

        self._n_updates += 1

        return result
        
    def online_predict(self, x: np.ndarray, V = None, return_var_theta = False):
        result = {}

        if V is None:
            V = self.V

        result['mean'] = x.T @ self.theta

        # Compute covariance of prediction error
        var_pred_err = self.V*(1 + x.T @ self.inner_var_theta @ x)

        if not self._full_cov:
            var_pred_err = np.diag(var_pred_err)

        result['cov'] = var_pred_err

        # Compute variance of theta
        if return_var_theta:
            result["cov_theta"] = self.get_var_vec_theta(V)

        return result

    def get_var_vec_theta(self, V: np.ndarray = None):
        """
        Returns the variance of the model parameters theta.
        If V is not provided, uses the internal V.
        """
        if V is None:
            V = self.V

        # Compute the variance of theta
        var_theta = np.kron(V, self.inner_var_theta)

        return var_theta

    def get_model_params(self):
        return self.theta

class RRR(OnlinePrediction):

    #TODO: consider including batch functionality into this class

    def __init__(self, X, Y, horizon, burn_in = 1, tilde_k_init_val = 0, track_memory = False, combine_variance = True, full_cov = True, center_cov = False, format_as_Y = True, scorefun = rmse, default_params = None):
        self.horizon = horizon
        self.format_as_Y = format_as_Y
        self.Y_hat = Lag(self, amount = horizon)
        self.scorefun = scorefun
        super().__init__(X, Y, horizon, n = DIM_X, m = DIM_Y, burn_in=burn_in, tilde_k_init_val=tilde_k_init_val, track_memory=track_memory, combine_variance=combine_variance, full_cov=full_cov, center_cov = center_cov, default_params = default_params)

    def create(self, n, m, burn_in, tilde_k_init_val, track_memory, combine_variance, full_cov, center_cov, **default_params):
        return RRRPredictor(n, m, self.horizon, burn_in, tilde_k_init_val, track_memory, combine_variance, full_cov, center_cov)

    def online_update(self, state: RRRPredictor, x_i, y_i, x_train_i, Q: np.ndarray | float = None, theta_tilde: np.ndarray = None, V: np.ndarray = None, estimate_V = True, mem = 0.99, return_var_theta = False):
        result = state.online_update(x_i, y_i, x_train_i=x_train_i, Q=Q, theta_tilde=theta_tilde, V=V, estimate_V=estimate_V, mem=mem, return_var_theta=return_var_theta)
        return result, state

    def online_predict(self, state: RRRPredictor, x_i, V = None, return_var_theta = False):
        return state.online_predict(x_i, V = V, return_var_theta=return_var_theta)
    
    def score(self, state, X, Y, prediction, **params):
        n = len(Y)

        # Get old predictions from state
        Y_hat = state.Y_hat.get(n)

        # Overwrite last n-horizon entries with new predictions (discard )
        Y_hat[self.horizon:] = prediction["mean"][:-self.horizon]

        # Evaluate score and update prediction
        prediction["score"] = evaluate_score(Y_hat, Y, burn_in = state.burn_in, remove_nan = True, scorefun = self.scorefun)
        return prediction

class BackShift(Transformation):

    def __init__(self, shifts: list | dict, data = DEFAULT_SOURCE, skip_duplicates = False, initial_value = np.nan):
        """
        shifts: list of lists or dict. If list of lists, each inner list represents a row of the bacshift matrix, in which case None values are treated as zeros.
        If dict, keys are (i,j) tuples representing the row and column indices of non-zero entries, and values are the corresponding shifts.
        """

        # Check that object is a list of lists of same length
        if not isinstance(shifts, dict):

            if isinstance(shifts, list) and not isinstance(shifts[0], list):
                # Assume "diagonal" structure with single input, i.e. m = 1
                shifts = {(i, 0): s for i, s in enumerate(shifts) if s is not None}

            # Check that all sublists have same length
            elif not all(len(s) == len(shifts[0]) for s in shifts):
                raise ValueError("All sublists must have the same length")
            else:
                shifts = {(i, j): s for i, s in enumerate(shifts) for j, s in enumerate(s) if s is not None}

        self.n = max(i for i, j in shifts.keys()) + 1 # Number of outputs

        if self.n == 0:
            raise ValueError("Shifts cannot be empty")


        self.max_shifts = {i: max(s for (ii, j), s in shifts.items() if ii == i) for i in range(self.n)}
        self.max_shift = max(self.max_shifts.values())
        self.shifts = shifts
        self.skip_duplicates = skip_duplicates
        self.initial_value = initial_value
        super().__init__(data = data, memory = MEMORY)

    def evaluate(self, data, memory = None):
        # Fetch data from memory
        # TODO: use CircularBuffer for efficiency
        if memory is None:
            old_data = np.full((self.max_shift, data.shape[1]), self.initial_value)
            if self.skip_duplicates:
                offset = 0
        else:
            old_data, offset = memory

        all_data = np.vstack((old_data, data))

        shifted_data = {}
        # Input Z (t x n) -> Temp Y -> Output X
        t = data.shape[0]

        # Collect lagged series
        for (i,j), lag in self.shifts.items():
            if (j, lag) not in shifted_data:
                if lag == 0:
                    shifted_data[(j, lag)] = all_data[self.max_shift:self.max_shift + t, j]
                elif lag > 0:
                    shifted_data[(j, lag)] = all_data[-(t + lag):-lag, j]

        # Form output
        if self.skip_duplicates:
            X = np.full((t, self.n), np.nan)

            # Get mask to select every max(lag)'th row, starting from an offset
            mask = np.arange(self.max_shift - offset, t, self.max_shift + 1)
            X[mask, :] = 0

            # Update offset
            offset = (offset + t) % (self.max_shift + 1)

            # Sum contributions
            for (i,j), lag in self.shifts.items():
                X[mask, i] += shifted_data[(j, lag)][mask]
        else:
            X = np.zeros((t, self.n))

            for (i,j), lag in self.shifts.items():

                X[:, i] += shifted_data[(j, lag)]

        if self.skip_duplicates:
            return X, (all_data[-self.max_shift:], offset)
        else:
            return X, (all_data[-self.max_shift:], None)

class ARX(OnlinePrediction):

    def __init__(self, exog, endog, horizon, p, burn_in = 1, tilde_k_init_val = 0, track_memory = False, combine_variance = True, full_cov = True, format_as_Y = True, scorefun = rmse, default_params = None):
        self.horizon = horizon
        self.p = p

        # Make regression model for 1-step forecasts
        endog = BackShift(list(reversed(range(p))), endog) # ordered as (y_t-p+1, ..., y_t-1, y_t)
        X = Apply(np.hstack, endog, exog[:, 0])

        self.format_as_Y = format_as_Y

        self.scorefun = scorefun
        self.format_as_Y = format_as_Y

        super().__init__(X, endog, 1, Z = exog, n = DIM_X, m = DIM_Y, burn_in=burn_in, tilde_k_init_val=tilde_k_init_val, track_memory=track_memory, combine_variance=combine_variance, full_cov=full_cov, default_params = default_params)

    def create(self, n, m, burn_in, tilde_k_init_val, track_memory, combine_variance, full_cov, **default_params):
        return RRRPredictor(n, m, self.horizon, burn_in, tilde_k_init_val, track_memory, combine_variance, full_cov)
    
    def online_update(self, state: RRRPredictor, x_i, y_i, x_train_i, z_i, Q: np.ndarray | float = None, theta_tilde: np.ndarray = None, V: np.ndarray = None, estimate_V = True, mem = 0.99, return_var_theta = False):
        state.online_update(x_i, y_i, x_train_i=x_train_i, Q=Q, theta_tilde=theta_tilde, V=V, estimate_V=estimate_V, mem=mem, return_var_theta=False)
        return self.online_predict(state, x_i, z_i, V=V, return_var_theta=return_var_theta), state

    def online_predict(self, state: RRRPredictor, x_i, z_i, V = None, return_var_theta = False):

        result = state.online_predict(x_i, V=V, return_var_theta=return_var_theta)

        theta = state.theta.squeeze()

        arx_pred = np.full(self.horizon, np.nan)
        arx_var = np.full(self.horizon, np.nan) # TODO: consider computing full covariance between horizons rather than just variances

        weights = np.zeros(self.p)
        weights[-1] = 1

        ar_params = theta[:self.p]

        endog = x_i[:self.p] # x_i contains both endog and exog, but we only want the endog part.

        if V is None:
            V = state.V

        # Modify result to include predictions for all horizons
        for h in range(self.horizon):

            # Combine endogenous history with exogenous for horizon h
            reg_i = np.hstack([endog, z_i[h]])
            
            # Predict h-step ahead using theta and x_i
            arx_pred[h] = (reg_i.T @ theta).item()

            # Push weights forward for next prediction
            new_weight = weights @ ar_params
            weights = np.append(weights[1:], new_weight)

            # Compute variance
            arx_var[h] = (V * (weights @ weights)).item()

            # Update endogenous history
            endog = np.append(endog[1:], arx_pred[h])

        result["mean"] = arx_pred
        result["var"] = arx_var

        return result

    def score(self, state: RRRPredictor, X, Y, prediction, **predictor_params):
        n = len(Y)

        # Get old predictions from state
        Y_hat = state.Y_hat.get(n)

        # Overwrite last n-horizon entries with new predictions (discard )
        Y_hat[self.horizon:] = prediction["mean"][:-self.horizon]

        # Evaluate score and update prediction
        prediction["score"] = evaluate_score(Y_hat, Y, burn_in = state.burn_in, remove_nan = True, scorefun = self.scorefun)
        return prediction
    

