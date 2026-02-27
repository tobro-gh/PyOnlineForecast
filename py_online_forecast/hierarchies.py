#%%
import numpy as np
from . import core as c
from . import prediction as p
from . import features as f
import os

def minT(S, Y: np.ndarray, Y_hat: np.ndarray, l_shrink = 0):
    # y: bottom level observations
    # y_hat: base forecasts at all levels
    tmp = Y @ S.T - Y_hat
    cov = 1/len(Y) * tmp.T @ tmp
    cov_shrink = (1-l_shrink)*cov + l_shrink*np.diag(np.diag(cov)) # Shrink towards diagonal
    W = np.linalg.inv(cov_shrink)
    StW = S.T @ W
    P = np.linalg.inv(StW @ S) @ StW
    res = (S @ P @ Y_hat.T).T
    return P, res, W

#if os.path.exists(c.data_folder) and 'hierarchical_data.csv' in os.listdir(c.data_folder):
#    sample_hierarchical_data = c.read_forecast_csv(c.data_folder + '/hierarchical_data.csv')

def construct_temporal_hierarchy(shape: list[tuple[int, int]]) -> np.ndarray:
    level_matrices = [np.kron(np.eye(q), np.ones(k)) for k, q in shape]    
    return np.vstack(level_matrices)

def hierarchy_shape(*h, as_groups = False):
    """
    Returns a list of tuples giving a hierarchy shape [(k_n, q_n), ..., (k_i, q_i), ..., (k_0, q_0)], where k_i is the number of 
    base forecasts required for level i, and q_i is the number of entries of level i in the hierarchy. 
    Here i >= 0 and i = 0 is the bottom level so that k_i > k_j for i < j.

    h: tuple of integers. If as_groups is False, h represents k_i. if as_groups
    is True, h represents the sequence of groupings, i.e. h_i = k_{n-(i+1)}/k_{n-i} where k_n = 1.
    """
    if as_groups:
        levels = [1]
        for g in h:
            if g == 1:
                raise ValueError("Grouping should be greater than 1")
            levels.insert(0, levels[0] * g)
    else:
        levels = h
        if not levels[-1] == 1:
            levels = levels + (1,)

    m = levels[0]
    lens = []
    for k in levels:
        quotient, remainder = divmod(m, k)
        if remainder != 0:
            raise ValueError(f"level {k} is not a factor of {m}")
        lens.append((k, quotient))
    return lens

def form_glm(Y, Y_hat, S_top):
    
    nsm, m = S_top.shape

    # Top level forecasts are the first n-m columns of Y_hat
    Y_hat_top = Y_hat[:, :nsm]

    # Bottom level forecasts are the last m columns of Y_hat
    Y_hat_bot = Y_hat[:, -m:]

    # Top level forecasts minus summed bottom level forecasts (reconciliation error)
    X_out = Y_hat_top - (Y_hat_bot @ S_top.T)

    # Observations minus bottom level forecasts (base forecast errors)
    Y_out = Y - Y_hat_bot

    return X_out, Y_out


class HierarchyRegressor(c.Transformation):
    """
    Returns X = Y_hat_top - Y_hat_bot @ S_top.T
    """

    def __init__(self, Y_hat_top, Y_hat_bot, S_top):
        super().__init__(Y_hat_top = Y_hat_top, Y_hat_bot = Y_hat_bot)
        self.S_top = S_top
    
#    @c.standardize_wrapper("Y_hat_top", "Y_hat_bot", ensure_dim = 2, output_as = "Y_hat_top")
    def evaluate(self, Y_hat_top, Y_hat_bot):
        # Convert to numpy arrays if not already
        if not isinstance(Y_hat_top, np.ndarray):
            Y_hat_top = np.asarray(Y_hat_top)
        if not isinstance(Y_hat_bot, np.ndarray):
            Y_hat_bot = np.asarray(Y_hat_bot)

        # Top level forecasts minus summed bottom level forecasts (reconciliation error)
        X = Y_hat_top - (Y_hat_bot @ self.S_top.T)

        return X

class HierarchyRegressand(c.Transformation):
    """
    Returns Y = Y_bot - Y_hat_bot, lagged by horizon
    """

    def __init__(self, Y_bot, Y_hat_bot, S_top, horizon: int):
        self.horizon = horizon
        self.S_top = S_top
        self.m = S_top.shape[1]
        super().__init__(Y_bot = Y_bot, Y_hat_bot = Y_hat_bot, Y_hat_bot_old = c.MEMORY)
    
#    @c.standardize_wrapper("Y_bot", "Y_hat_bot", ensure_dim = 2, output_as = "Y_bot")
    def evaluate(self, Y_bot, Y_hat_bot, Y_hat_bot_old = None):
        
        if not isinstance(Y_bot, np.ndarray):
            Y_bot = np.asarray(Y_bot)
        if not isinstance(Y_hat_bot, np.ndarray):
            Y_hat_bot = np.asarray(Y_hat_bot)

        if Y_hat_bot_old is None:
            Y_hat_bot_old = np.full((self.horizon, self.m), np.nan)

        Y_hat_bot_fit = np.vstack((Y_hat_bot_old, Y_hat_bot))

        # Observations minus bottom level forecasts (base forecast errors)
        Y = Y_bot - Y_hat_bot_fit[:-self.horizon]

        return Y, Y_hat_bot_fit[-self.horizon:]

class RidgeReconciliation(c.Transformation):

    def __init__(self, Y_bot, Y_hat, S_top, horizon, apply_format = True, full_hierarchy_cov = False, opt_shrink = False, **kwargs):
        self.S_top = S_top
        nsm, m = S_top.shape
        n = m + nsm
        self.n, self.m = n, m

        self.apply_format = apply_format
        self.full_hierarchy_cov = full_hierarchy_cov
        self.S = np.vstack((S_top, np.eye(S_top.shape[1])))

        Y_hat_top = f.SelectIndices(range(nsm), data = Y_hat) # First n-m columns
        Y_hat_bot = f.SelectIndices(range(nsm, n), data = Y_hat) # Last m columns

        X = HierarchyRegressor(Y_hat_top, Y_hat_bot, S_top)
        Y = HierarchyRegressand(Y_bot, Y_hat_bot, S_top, horizon)

        if opt_shrink:
            self.prediction = SRRR(X, Y, horizon, S_top, format_as_Y=False, **kwargs)
        else:
            self.prediction = p.RRR(X, Y, horizon, format_as_Y=False, **kwargs)        

        super().__init__(Y_hat_bot, Y_hat, self.prediction)

    def evaluate(self, Y_hat_bot, Y_hat, pred):

        if not isinstance(Y_hat_bot, np.ndarray):
            Y_hat_bot = np.asarray(Y_hat_bot)

        # Prepare result
        res = pred.copy()

        # Get predicted bottom level errors
        err_bot_hat = pred["mean"]

        # Reconciled bottom level forecasts
        Y_bot_rec = Y_hat_bot + err_bot_hat

        # Sum to get reconciled forecasts at all levels
        Y_hat_rec = Y_bot_rec @ self.S.T

        # Format like input
        if self.apply_format:
            Y_hat_rec = c.format_like(Y_hat_rec, Y_hat)

        # Store reconciled forecasts in result
        res["mean"] = Y_hat_rec

        # Compute (co)variance
        var_bot = pred["cov"]

        # If variance is given as diagonal, convert to full covariance matrix (assuming zero correlations)
        if var_bot.ndim == 2:

            t, n = var_bot.shape

            var_bot_full = np.full((t, n, n), np.nan)

            # Get mask for valid rows
            valid_rows = ~np.any(np.isnan(var_bot), axis=1)

            # Initialise valid rows to zero
            var_bot_full[valid_rows] = 0

            # Assign diagonal elements
            ran = np.arange(n)
            var_bot_full[:,  ran, ran] = var_bot[ :, ran]

            var_bot = var_bot_full

        # Compute reconciled variance S V[x] S.T
        rec_var_est = self.S @ var_bot @ self.S.T

        # Fetch only diagonal? (TODO: consider doing self.S @ var_bot @ self.S.T more efficiently if only diagonal is needed)
        if not self.full_hierarchy_cov:
            rec_var_est = np.diagonal(rec_var_est, axis1=1, axis2=2)

        # Format like input and store result
        if self.apply_format:
            rec_var_est = c.format_like(rec_var_est, Y_hat, outer_prod = self.full_hierarchy_cov)
        res["cov"] = rec_var_est

        return res

class RidgeReconciler(p.Model):

    def __init__(self, S_top, horizon = 1, Y_bot: c.Source = None, Y_hat: c.Source = None, full_hierarchy_cov = False, apply_format = True, opt_shrink = False, **kwargs):
        self.Y_bot = Y_bot or c.Source("Y_bot")
        self.Y_hat = Y_hat or c.Source("Y_hat")
        self.rec_err = c.Source("rec_err")
        self.rec_pred = RidgeReconciliation(self.Y_bot, self.Y_hat, S_top, horizon, apply_format = apply_format, full_hierarchy_cov = full_hierarchy_cov, opt_shrink=opt_shrink, **kwargs)
        super().__init__(self.rec_pred)

    @property
    def X(self):
        return self.rec_pred.prediction.X
    
    @property
    def Y(self):
        return self.rec_pred.prediction.Y

    @property
    def predictor(self):
        return self.state[self.rec_pred.prediction][0]

    def set_score_mode(self):
        self.rec_pred.prediction.set_score_mode()
    
    def unset_score_mode(self):
        self.rec_pred.prediction.unset_score_mode()

    def rec_update(self, Y_bot, Y_hat, **kwargs):
        return self.update({self.Y_bot: Y_bot, self.Y_hat: Y_hat}, **kwargs)

    def rec_fit(self, Y_bot, Y_hat, **kwargs):
        self.reset_state()
        return self.update({self.Y_bot: Y_bot, self.Y_hat: Y_hat}, **kwargs)

    @property
    def P(self):
        # Returns the projection matrix P as computed from the model parameters.
        nsm, m = self.S_top.shape
        n = m + nsm
        theta = self.predictor.get_model_params()
        
        if theta is None:
            raise ValueError("Model not fitted yet")

        null = np.full((n-m, m), 0)
        P = (np.vstack((null, np.eye(m))) + np.vstack((np.eye(n-m), -self.S_top.T)) @ theta).T
        return P

class TemporalRidgeReconciler(RidgeReconciler):

    def __init__(self, S_top, B: list | dict, horizon = 1, Z_bot: c.Source = None, Y_hat: c.Source = None, skip_duplicates = False, full_hierarchy_cov = False, apply_format = True, opt_shrink = False, **kwargs):
        # Construct backshift operator
        self.Z_bot = Z_bot or c.Source("Z_bot")
        Y_bot = f.BackShift(B, skip_duplicates=skip_duplicates, data=self.Z_bot)

        # Initialize Reconciler with lagged bottom level data as regressand input
        super().__init__(S_top, horizon, Y_bot, Y_hat, full_hierarchy_cov = full_hierarchy_cov, apply_format = apply_format, opt_shrink=opt_shrink, **kwargs)

    def rec_update(self, Y_bot, Y_hat = None, **model_params):
        return super().update({self.Z_bot: Y_bot, self.Y_hat: Y_hat}, **model_params)

    def rec_fit(self, Y_bot, Y_hat=None, **model_params):
        self.reset_state()
        return super().update({self.Z_bot: Y_bot, self.Y_hat: Y_hat}, **model_params)
    

class Hierarchy(c.Transformation):

    def __init__(self, Y_bot, Y_hat, S_top, horizon, predictor_configuration, variance_name = None, get_full_cov = False, apply_format = True):
        self.S_top = S_top
        nsm, m = S_top.shape
        n = m + nsm
        self.n, self.m = n, m

        self.S = np.vstack((S_top, np.eye(S_top.shape[1])))

        Y_hat_top = c.SelectIndices(range(nsm), data = Y_hat) # First n-m columns
        Y_hat_bot = c.SelectIndices(range(nsm, n), data = Y_hat) # Last m columns

        X = HierarchyRegressor(Y_hat_top, Y_hat_bot, S_top)
        Y = HierarchyRegressand(Y_bot, Y_hat_bot, S_top, horizon)

        self.prediction = p.Prediction(X, Y, horizon, predictor_configuration, apply_format = False)        
        self.variance_name = variance_name
        self.get_full_cov = get_full_cov
        self.apply_format = apply_format

        super().__init__(Y_hat_bot, Y_hat, self.prediction)


    @c.standardize_wrapper("Y_hat_bot", ensure_dim = 2)
    def evaluate(self, Y_hat_bot, Y_hat, pred):

        # Prepare result
        res = pred.copy()

        # Get name of target variable
        target = self.prediction.config.predictor_type.target or next(iter(self.prediction.config.predictors))

        # Get predicted bottom level errors
        err_bot_hat = pred[target]

        # Reconciled bottom level forecasts
        Y_bot_rec = Y_hat_bot + err_bot_hat

        # Sum to get reconciled forecasts at all levels
        Y_hat_rec = Y_bot_rec @ self.S.T

        # Format like input
        if self.apply_format:
            Y_hat_rec = c.format_like(Y_hat_rec, Y_hat)

        # Store reconciled forecasts in result
        res[target] = Y_hat_rec

        # Compute (co)variance
        if self.variance_name is not None:
            var_bot = pred[self.variance_name]

            # If variance is given as diagonal, convert to full covariance matrix (assuming zero correlations)
            if var_bot.ndim == 2:

                t, n = var_bot.shape

                var_bot_full = np.full((t, n, n), np.nan)

                # Get mask for valid rows
                valid_rows = ~np.any(np.isnan(var_bot), axis=1)

                # Initialise valid rows to zero
                var_bot_full[valid_rows] = 0

                # Assign diagonal elements
                ran = np.arange(n)
                var_bot_full[:,  ran, ran] = var_bot[ :, ran]

                var_bot = var_bot_full

            # Compute reconciled variance S V[x] S.T
            rec_var_est = self.S @ var_bot @ self.S.T

            # Fetch only diagonal? (TODO: consider doing self.S @ var_bot @ self.S.T more efficiently if only diagonal is needed)
            if not self.get_full_cov:
                rec_var_est = np.diagonal(rec_var_est, axis1=1, axis2=2)

            # Format like input and store result
            rec_var_est = c.format_like(rec_var_est, Y_hat, outer_prod = self.get_full_cov)
            res[self.variance_name] = rec_var_est

        return res


class Reconciler(p.Model):

    def __init__(self, S_top, predictor_configuration, horizon = 1, Y_bot: c.Source = None, Y_hat: c.Source = None, variance_name = None, get_full_cov = False, scorefun = None, burn_in = 0, remove_nan = True, apply_format = True):
        nsm, m = S_top.shape
        self.S_top = S_top
        n = m + nsm
        self.n, self.m = n, m

        self.Y_bot = Y_bot or c.Source("Y_bot")
        self.Y_hat = Y_hat or c.Source("Y_hat")
        self.rec_err = c.Source("rec_err")

        self.rec_pred = Hierarchy(self.Y_bot, self.Y_hat, S_top, horizon, predictor_configuration, variance_name, get_full_cov, apply_format = apply_format)

        super().__init__(self.rec_pred, scorefun = scorefun, burn_in = burn_in, remove_nan = remove_nan)


    @property
    def X(self):
        return self.rec_pred.prediction.X
    
    @property
    def Y(self):
        return self.rec_pred.prediction.Y

    @property
    def predictor(self):
        return self.state[self.rec_pred.prediction][0]

    def rec_update(self, Y_bot, Y_hat, **kwargs):
        return self.update({self.Y_bot: Y_bot, self.Y_hat: Y_hat}, **kwargs)

    def rec_fit(self, Y_bot, Y_hat, **kwargs):
        self.reset_state()
        return self.update({self.Y_bot: Y_bot, self.Y_hat: Y_hat}, **kwargs)

    @property
    def P(self):
        # Returns the projection matrix P as computed from the model parameters.
        nsm, m = self.S_top.shape
        n = m + nsm
        theta = self.predictor.get_model_params()
        
        if theta is None:
            raise ValueError("Model not fitted yet")

        null = np.full((n-m, m), 0)
        P = (np.vstack((null, np.eye(m))) + np.vstack((np.eye(n-m), -self.S_top.T)) @ theta).T
        return P

#def build_backshift(bot_vars: pd.Index, forecast_vars: pd.Index, skip_duplicates = False):
class TemporalReconciler(Reconciler):

    def __init__(self, S_top, B: list | dict, predictor_configuration, horizon = 1, Z_bot: c.Source = None, Y_hat: c.Source = None, variance_name = None, skip_duplicates = False, get_full_cov = False, scorefun = None, burn_in = 0, remove_nan = True, apply_format = True):
        # Construct backshift operator
        self.Z_bot = Z_bot or c.Source("Z_bot")
        Y_bot = f.BackShift(B, skip_duplicates=skip_duplicates, data=self.Z_bot)

        # Initialize Reconciler with lagged bottom level data as regressand input
        super().__init__(S_top, predictor_configuration, horizon, Y_bot, Y_hat, variance_name, get_full_cov, scorefun, burn_in, remove_nan, apply_format = apply_format)

    def rec_update(self, Y_bot, Y_hat = None, **model_params):
        return super().update({self.Z_bot: Y_bot, self.Y_hat: Y_hat}, **model_params)

    def rec_fit(self, Y_bot, Y_hat=None, **model_params):
        self.reset_state()
        return super().update({self.Z_bot: Y_bot, self.Y_hat: Y_hat}, **model_params)
    

class SRRRPredictor(p.RRRPredictor):
    def __init__(self, S_top, n, m, burn_in, tilde_k_init_val, track_memory, combine_variance, full_cov, center_cov = False):
        super().__init__(n, m, burn_in, tilde_k_init_val, track_memory, combine_variance, full_cov, center_cov = center_cov)
        self.S_top = S_top
        npm = n+m
        self.sigma_bot_hat_d = np.zeros(m)
        self.sigma_top_hat_d = np.zeros(n)
        self.sigma_hat = np.zeros((npm, npm))
        self.var_sigma_hat = np.zeros((npm, npm))
        self.diag_mask = np.eye(npm, dtype=bool)

    def update_prior(self, x_i, y_i, mem, l_shrink):
        # Compute optimal Q and theta_tilde based on current error estimates
        err_top = self.S_top @ y_i - x_i

        if l_shrink == "auto": # Estimate shrinkage parameter using full covariance matrix

            if mem == 1:
                raise ValueError("Automatic shrinkage parameter estimation is not supported for no forgetting (mem=1).")
    
            err_full = np.append(err_top, y_i)

            self.sigma_hat = mem * self.sigma_hat + (1 - mem) * np.outer(err_full, err_full)

            err_sq = err_full**2
            sigma_sq = self.sigma_hat**2

            # Update higer order variance estimates
            self.var_sigma_hat = mem*(1-mem)**2 * (np.outer(err_sq, err_sq) - sigma_sq) + mem**2 * self.var_sigma_hat

            # Sum non-diagonal elements to estimate shrinkage parameter
            l_shrink = np.where(self.diag_mask, 0, self.var_sigma_hat).sum() / np.where(self.diag_mask, 0, sigma_sq).sum()

            cov_top_hat_d = np.diag(np.diag(self.sigma_hat[:self.n, :self.n]))
            cov_bot_hat_d = np.diag(np.diag(self.sigma_hat[self.n:, self.n:]))

        else: # Estimate covariance diagonal

            # y_i contains bottom level errors
            self.sigma_bot_hat_d = self.sigma_bot_hat_d * mem + (1 - mem) * y_i**2

            # x_i contains coherency errors, from which top level errors can be computed
            self.sigma_top_hat_d = self.sigma_top_hat_d * mem + (1 - mem) * err_top**2

            cov_top_hat_d = np.diag(self.sigma_top_hat_d)
            cov_bot_hat_d = np.diag(self.sigma_bot_hat_d)

        b0 = cov_top_hat_d + self.S_top @ cov_bot_hat_d @ self.S_top.T
        b1 = self.S_top @ cov_bot_hat_d

        Q = 1/(1 - mem) * (l_shrink/(1-l_shrink))*b0 # Check that this is correct. Consider using a better reperesentative of current memory than the limit 1/(1 - mem).

        theta_tilde = np.linalg.solve(b0, b1)

        return Q, theta_tilde
 
class SRRR(p.RRR):

    def __init__(self, X, Y, horizon, S_top, **kwargs):
        super().__init__(X, Y, horizon, **kwargs)
        self.S_top = S_top

    def create(self, n, m, burn_in, tilde_k_init_val, track_memory, combine_variance, full_cov, center_cov):
        return SRRRPredictor(self.S_top, n, m, burn_in, tilde_k_init_val, track_memory, combine_variance, full_cov, center_cov=center_cov)

    def online_update(self, state: SRRRPredictor, x_i, y_i, x_train_i, mem=0.99, l_shrink = "auto", **kwargs):

        # Compute optimal Q and theta_tilde based on current error estimates
        Q, theta_tilde = state.update_prior(x_i, y_i, mem, l_shrink) # x_i or x_train_i?

        # Call parent online_update with computed Q and theta_tilde
        result = state.online_update(x_i, y_i, x_train_i, Q, theta_tilde, mem=mem, **kwargs)

        # Return result and updated state
        return result, state

def find_nodes(func):
    def wrapper(self, *args, **kwargs):
        nodes = []
        all_nodes = self.get_nodes()
        for arg in args:
            if isinstance(arg, Node):
                nodes.append(arg)
            else:
                # Find node in hierarchy with name arg
                nodes.append(next(n for n in all_nodes if n.name == arg))

        return func(self, *nodes, **kwargs)
    return wrapper

class Node:

    _counter = 0

    def __init__(self, *sources: 'Node', name = None):
        level = 0
        for node in sources:
            level = max(level, node.level + 1)
        self.level = level
        self.sources = list(sources)
        self.lags = {}
        if name is None:
            name = str(Node._counter)
            Node._counter += 1
        self.name = name

    def __add__(self, other):
        if not isinstance(other, Node):
            raise ValueError("Can only add Node to Node")
        return Node(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)

    def shift(self, h: int):
        if len(self.sources) > 0:
            raise ValueError("Can only shift leaf nodes")
        if h < 0:
            raise ValueError("Shift must be non-negative")
        if h not in self.lags:
            self.lags[h] = LaggedNode(h, self)
        return self.lags[h]
            
    def is_parent(self, other):
        if other in self.sources:
            return True
        for src in self.sources:
            if src.is_parent(other):
                return True
        return False

    def get_nodes(self):
        nodes = [self]
        current_nodes = nodes
        while len(current_nodes) > 0:
            new_nodes = []
            for node in current_nodes:
                new_nodes.extend(node.sources)
            nodes.extend(new_nodes)
            current_nodes = new_nodes

        return nodes

    def get_top_nodes(self):
        bot_nodes = self.get_leaf_nodes()
        top_nodes = []
        for c in self.get_nodes():
            if c not in bot_nodes:
                top_nodes.append(c)
        return top_nodes

    @find_nodes
    def build_B(self, *obs_vars):
        # Get all observable nodes, based on observable variables (not including lagged nodes/variables)
        # Check that obs_vars are not LaggedNodes
        for obs in obs_vars:
            if isinstance(obs, LaggedNode):
                raise ValueError("Observed variables cannot be lagged nodes")

        # Get all lagged variables of the observed variables
        lagged_vars = list(obs_vars)
        for obs in obs_vars:
            lagged_vars.extend(obs.lags.values())

        # Get only the lagged variables that are part of the hierarchy
        obs_nodes = [v for v in lagged_vars if self.is_parent(v)]

        # Build B matrix from temporal variables to observed variables
        B = {}
        for i, temp in enumerate(obs_nodes):
            for j, obs in enumerate(obs_vars):
                if obs is temp:
                    B[i,j] = 0
                elif temp in obs.lags.values():
                    B[i,j] = temp.lag
        return obs_nodes, B

    def get_leaf_nodes(self):
        # Get all leaf nodes (nodes without sources)
        if not self.sources:
            return [self]
        else:
            leafs = []
            for src in self.sources:
                # Add leaves not already in list
                for leaf in src.get_leaf_nodes():
                    if leaf not in leafs:
                        leafs.append(leaf)
            return leafs

    @find_nodes
    def build_S_top(self, *bot_nodes):
        # Check that bot_nodes are the right ones
        if not set(bot_nodes) == set(self.get_leaf_nodes()):
            raise ValueError("Bottom nodes do not match hierarchy.")
        top_nodes = self.get_top_nodes()
        n, m = len(top_nodes), len(bot_nodes)
        S_top = np.zeros((n, m))
        for i, lat in enumerate(top_nodes):
            for j, obs in enumerate(bot_nodes):
                if lat.is_parent(obs):
                    S_top[i,j] = 1

        return top_nodes, S_top

    @find_nodes
    def build_A_lat(self, *obs_nodes):
        # Get obs nodes if not provided
        bot_nodes = self.get_leaf_nodes()

        n_bot = len(bot_nodes)
        n_obs = len(obs_nodes)
#        if n_bot != n_obs:
#            raise ValueError("Number of observed nodes must match number of bottom nodes")
        # Get invers of K satisfying bot_nodes = K @ obs_nodes
        # obs_nodes = K_inv @ bot_nodes
        K_inv = np.zeros((n_obs,n_bot))
        for i, bot in enumerate(bot_nodes):
            for j, obs in enumerate(obs_nodes):
                if bot is obs or obs.is_parent(bot):
                    K_inv[j,i] = 1

        K = np.linalg.pinv(K_inv) # inv if n = m?
#        K = np.linalg.inv(K_inv) # inv if n = m?

        # Get latent nodes
        lat_nodes = [v for v in self.get_nodes() if not v in obs_nodes]

        # Get top nodes
        top_nodes, S_top = self.build_S_top(*bot_nodes)

        n_top = len(top_nodes)
        n_lat = len(lat_nodes)
        C_top_lat = np.zeros((n_top, n_lat))
        # Fill in entries of permutation matrix C_top_lat
        for i, top in enumerate(top_nodes):
            for j, lat in enumerate(lat_nodes):
                if top is lat:
                    C_top_lat[i,j] = 1
    
        C_bot_lat = np.zeros((n_bot, n_lat))
        for i, bot in enumerate(bot_nodes):
            for j, lat in enumerate(lat_nodes):
                if bot is lat:
                    C_bot_lat[i,j] = 1

        A_lat = (C_top_lat.T @ S_top + C_bot_lat.T) @ K
        return lat_nodes, A_lat.round().astype(int)

    def print_hierarchy(self, n = 0):
        indent = "  " * n
        print(f"{indent}- {self}")
        for node in self.sources:
            node.print_hierarchy(n + 1)

    def __repr__(self):
        return f"Node({self.name})"


class LaggedNode(Node):
    
    def __init__(self, lag: int, source: Node):
        if lag < 0:
            raise ValueError("Lag must be non-negative")
        name = f"{source.name}(-{lag})"
        super().__init__(name=name)
        self.lag = lag
        self.var = source
        
    def shift(self):
        return self.sources[0].shift(self.lag + 1)


def make_hierarchy(edges):
    """
    edges: dict of {parent: [child1, child2, ...]} or {parent: [(child1, lag1), (child2, lag2), ...]} where lag is a non-negative integer representing the lag of the child node.
    """

    top_nodes = list(edges.keys())
    node_names = top_nodes.copy()
    for nodes in edges.values():
        for node in nodes:
            if isinstance(node, tuple):
                if node[0] not in node_names:
                    node_names.append(node[0])
            if node not in node_names:
                node_names.append(node)

    bot_nodes = [n for n in node_names if n not in edges]

    # Then create top nodes
    to_process = node_names.copy()
    nodes = {}

    while to_process:
        for node in to_process:
            if isinstance(node, tuple):
                if node[0] in nodes:
                    nodes[node] = nodes[node[0]].shift(node[1])
                    to_process.remove(node)
            elif node in edges:
                children = edges[node]
                if all(child in nodes for child in children):
                    nodes[node] = Node(*[nodes[child] for child in children], name=node)
                    to_process.remove(node)
            else:
                nodes[node] = Node(name=node)
                to_process.remove(node)

    return nodes, bot_nodes, top_nodes