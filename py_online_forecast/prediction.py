from .core import *
from .features import *

class DataFormat:

    def __init__(self, variables, outer_prod = False):
        self.variables = variables # variable names or a dict of variables for nested data
        self.outer_prod = outer_prod

    def apply(self, data, index = None):
        return to_pandas(data, index, self.variables, outer_prod = self.outer_prod)

    def check(self, data):
        if self.variables is not None:
            check_var(data, self.variables)

    @classmethod
    def from_reference(cls, val: dict | pd.DataFrame | pd.Series | np.ndarray, outer_prod = False):
        vars = get_vars(val)
        return cls(variables = vars, outer_prod = outer_prod)

    def __call__(self, data, index = None):
        return self.apply(data, index)
    
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

def rmse(x):
    return np.sqrt(np.mean(x**2))

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

        super().__init__(X, Y, update_predictor = UPDATE_PREDICTOR, state = MEMORY, **Z_kwarg)
    
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
    

    def evaluate(self, X, Y, update_predictor, state = None, Z = None, **params):

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

def make_prediction_ensemble(X: dict | tuple | Transformation, Y : tuple | Transformation, prediction_type, horizons, *args, input_horizons: dict = None, **kwargs):
    predictions = []

    if input_horizons is None:
        input_horizons = {h: (0, h) for h in horizons}
    
    if isinstance(Y, (tuple, list)):
        Y_sub = [GetHorizons(Y_i, drop_horizon=True) for Y_i in Y] 
        Y = Combine(*Y_sub)  
    else:
        Y = GetHorizons(Y, drop_horizon=True)

    for h, h_in in input_horizons.items():
        if isinstance(X, (tuple, list)):
            X_sub = [GetHorizons(data = X_i, horizons = h_in) for X_i in X]
            X_h = Combine(*X_sub)
        else:
            X_h = GetHorizons(data = X, horizons = h_in)

        predictions.append(prediction_type(X_h, Y, h, *args, **kwargs))

    return predictions

class Model:

    def __init__(self, *output: Source):
        self.output = Combine(*output, as_dict = True)
        self.state = None
        self.data_format = None

        # Fetch all Prediction dependencies in output recursively
        self.predictions = []
        def fetch_predictions(source):
            if isinstance(source, Prediction):
                self.predictions.append(source)

            if isinstance(source, Transformation):
                for dep in source.dependencies:
                    fetch_predictions(dep)

        for o in output:
            fetch_predictions(o)

        # Store which predictions are directly in the output
        self.direct_predictions = [p for p in self.predictions if p in output]

    def set_score_mode(self):
        for p in self.direct_predictions:
            p.score_mode = True
        
    def unset_score_mode(self):
        for p in self.direct_predictions:
            p.score_mode = False

    @property
    def predictor(self):
        result = {p: self.state[p][0] for p in self.predictions}
        if len(result) == 1:
            return next(iter(result.values()))
        return result

    def reset_state(self):
        self.state = None
        self.data_format = None

    def update(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, check = False, update_predictor = True, **params):

        if self.data_format is None:
            self.data_format = DataFormat.from_reference(data)

        if check:
            self.data_format.check(data)

        # Transform data
        result, self.state = self.output.apply(data, recursion_pars = self.state, update_predictor=update_predictor, return_recursion_pars=True, **params)
        
        if len(result) == 1:
            return next(iter(result.values()))

        return result

    def fit(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, **params):
        self.reset_state()
        return self.update(data, **params)

    def save_model(self, name: str = None):
        if name is None:
            return pickle.dumps(self)
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(self, f)

    def print_model_tree(self):
        self.output.print_dependency_tree()

@data_decorator    
def evaluate_score(Y_hat, Y, burn_in = 0, remove_nan = True, scorefun = rmse):
    resid = Y - Y_hat

    if burn_in > 0:
        resid = resid[burn_in:]
    if remove_nan:
        mask = ~np.isnan(resid).any(axis=1)
        resid = resid[mask]

    return scorefun(resid)

class RRREnsemble(Model):

    def __init__(self, X: dict | tuple | Transformation, Y : tuple | Transformation, horizons, *args, input_horizons: dict = None, **kwargs):
        predictions = make_prediction_ensemble(X, Y, RRR, *args, horizons = horizons, input_horizons = input_horizons, **kwargs)
        super().__init__(*predictions)
        self.ref = predictions[0]
        self._column_names = {}
        self.X = X
        self.Y = Y

    def reset_state(self):
        super().reset_state()
        self._column_names = {}

    def update(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, check = False, update_predictor = True, combine_horizons = True, **params):
        result = super().update(data, check, update_predictor, **params)

        if combine_horizons:
            combined_results = {}
            for name in result[self.ref]:
                if name == "score":
                    # Combine scores in a dict per horizon
                    combined_results[name] = {p.horizon: result[p][name] for p in self.output.sources}
                else:
                    combined_results[name], self._column_names[name] = combine_data({p.horizon: result[p][name] for p in self.output.sources}, columns = self._column_names.get(name, None))

            # Remove individual horizon results
            for p in self.output.sources:
                del result[p]

            result[self] = combined_results

        if len(result) == 1:
            return next(iter(result.values()))
        
        return result

    def fit(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, combine_horizons = True, **params):
        self.reset_state()
        return self.update(data, combine_horizons=combine_horizons, **params)

def load_model(file_name):
    if not file_name.endswith(".pkl"):
        file_name = file_name + ".pkl"
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
class WLS(Prediction):

    def __init__(self, X, Y, horizon, format_as_Y = True):
        self.format_as_Y = format_as_Y
        super().__init__(X, Y, horizon, n = DIM_X, m = DIM_Y)

    def create(self, n, m):
        # Initialize state as parameters of WLS model (theta)
        return np.zeros((n, m))

    def evaluate(self, X, Y, update_predictor, state = None):
        result, state = super().evaluate(X, Y, update_predictor, state)

        # Format output as Y
        if self.format_as_Y and isinstance(Y, (pd.DataFrame, pd.Series)):
            result = format_like(result, Y)

        return result, state

    def update(self, state, X: np.ndarray, Y: np.ndarray, X_train: np.ndarray, W: np.ndarray = None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(Y, pd.DataFrame):
            Y = Y.to_numpy()
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()

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
        return pred

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
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(Y, pd.DataFrame):
            Y = Y.to_numpy()
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(Z, pd.DataFrame):
            Z = Z.to_numpy()
        
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

    def update(self, state, X: np.ndarray | pd.DataFrame, Y: np.ndarray | pd.DataFrame, X_train: np.ndarray | pd.DataFrame, Z: np.ndarray | pd.DataFrame = None, **params):
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
    def online_predict(self, state, x_i, z_i = None, **params) -> tuple[pd.Series] | pd.Series:
        # Predict a single row.
        pass

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
        self.track_memory = track_memory
        if track_memory:
            self._memory = 0
            self._total_var = np.zeros((m, m))

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

            y_i_hat = self.Y_hat.get_slice(-1)[0] # Get oldest prediction
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

    def evaluate(self, X, Y, update_predictor, state = None, **params):
        result, state = super().evaluate(X, Y, update_predictor, state, **params)

        # Format output as Y
        if self.format_as_Y and isinstance(Y, (pd.DataFrame, pd.Series)):
            result["mean"] = to_pandas_like(result["mean"], Y)
            result["cov"] = to_pandas_like(result["cov"], Y, outer_prod=True)

        return result, state

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