from .core import *
from .features import *
#from .tools import *
    

def rmse(x):
    return np.sqrt(np.mean(x**2))

#@data_decorator    
def evaluate_score(Y_hat, Y, burn_in = 0, remove_nan = True, scorefun = rmse):
    resid = Y - Y_hat

    if burn_in > 0:
        resid = resid[burn_in:]
    if remove_nan:
        mask = ~np.isnan(resid).any(axis=1)
        resid = resid[mask]

    return scorefun(resid)

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

        # Ensure format/shape of inputs
        Y = endog
#        Z = ToExog(horizon, exog)
        Z = exog

        # Make regression model for 1-step forecasts
        endog = BackShift(list(reversed(range(p))), Y) # ordered as (y_t-p+1, ..., y_t-1, y_t)
#        X = Combine(endog, Z[:, 0])
        X = Apply(np.hstack, endog, Z[:, 0])

        self.format_as_Y = format_as_Y

        self.scorefun = scorefun
        self.format_as_Y = format_as_Y

        super().__init__(X, Y, 1, Z = Z, n = DIM_X, m = DIM_Y, burn_in=burn_in, tilde_k_init_val=tilde_k_init_val, track_memory=track_memory, combine_variance=combine_variance, full_cov=full_cov, default_params = default_params)


    def create(self, n, m, burn_in, tilde_k_init_val, track_memory, combine_variance, full_cov, **default_params):
        return RRRPredictor(n, m, self.horizon, burn_in, tilde_k_init_val, track_memory, combine_variance, full_cov)

    def evaluate(self, X, Y, update_predictor, state = None, Z = None, **params):
        result, state = super().evaluate(X, Y, update_predictor, state, Z = Z, **params)

        # Format output as Y
#        if self.format_as_Y and isinstance(Y, (pd.DataFrame, pd.Series)):
#            index = get_index(Y)
#            columns = get_vars(Y).tolist()
#            pred_columns = fc_columns_from_product(columns, [i+1 for i in range(self.horizon)])
 
#            result["mean"] = to_pandas(result["mean"], index, columns = pred_columns)
#            result["var"] = to_pandas(result["var"], index, columns = pred_columns)

        return result, state
    
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

    def score(self, state, X, Y, prediction, **predictor_params):
        n = len(Y)

        # Get old predictions from state
        Y_hat = state.Y_hat.get(n)

        # Overwrite last n-horizon entries with new predictions (discard )
        Y_hat[self.horizon:] = prediction["mean"][:-self.horizon]

        # Evaluate score and update prediction
        prediction["score"] = evaluate_score(Y_hat, Y, burn_in = state.burn_in, remove_nan = True, scorefun = self.scorefun)
        return prediction
    

