"""Transformations and tools for hierarchical forecast reconciliation.

This module provides transformations and tools for working with hierarchical forecast
reconciliation. The main class is `RidgeReconciliation`, which uses the `RRR` (or
optionally `SRRR`) prediction to model the bottom level forecast errors and then
constructs reconciled forecasts at all levels of the hierarchy.

For working with temporal hierarchies, the `TemporalRidgeReconciliation` class is a thin
wrapper around `RidgeReconciliation` that uses a `BackShift` transformation to construct
a bottom level variable for a temporal hierarchy.

The module also includes some tools for construct hierarchies using a `Node` class.
"""

import numpy as np
from . import core as c
from . import prediction as p
from . import features as f


class RidgeReconciliation(c.Transformation):
    r"""Hierarchical forecast reconciliation using ridge regression.

    Reconciles forecasts across a hierarchy by predicting bottom-level base forecast
    errors using a linear model in the top-level coherency errors. The transformation
    uses :class:`~prediction.RRR` or optionally :class:`SRRR` to perform online
    estimation and prediction at the bottom level and constructs reconciled forecasts by
    shifting and scaling the base forecasts error predictions.

    Parameters
    ----------
    S_top : np.ndarray, shape (n_top, n_bot)
        Summation matrix for top levels, i.e. the first ``n_top`` rows of the summation
        matrix, satisfying :math:`Y_{\\mathrm{top}}=S_\\mathrm{top} Y_{\\mathrm{bot}}`.
    Y_bot : Source or None, optional
        Source providing bottom-level observations. If ``None``, a new
        :class:`~core.Source` named ``"Y_bot"`` is created.
    Y_hat : Source or None, optional
        Source providing hierarchical forecasts at all levels, with shape
        ``(t, n_top + n_bot)``. Top-level forecasts are assumed to occupy the first
        ``n_top`` columns and bottom-level forecasts the last ``n_bot`` columns. If
        ``None``, a new :class:`~core.Source` named ``"Y_hat"`` is created.
    horizon : int, optional
        Forecast horizon used to align bottom-level observations with the
        corresponding forecasts when computing training targets. Default is ``1``.
    full_hierarchy_cov : bool, optional
        If ``True``, the reconciled forecast covariance is returned as a full
        ``(t, n_top + n_bot, n_top + n_bot)`` array. If ``False`` (default), only the
        diagonal variances are returned as a ``(t, n_top + n_bot)`` array.
    opt_shrink : bool, optional
        If ``True``, uses :class:`SRRR` with optimal covariance shrinkage to
        compute the prior for the ridge regression. Default is ``False``.
    **kwargs
        Additional keyword arguments passed to the underlying
        :class:`~prediction.RRR` or :class:`SRRR` predictor (e.g. ``mem``,
        ``burn_in``, ``init_K``, ``full_cov``). Note in particular when ``full_cov`` is
        False, only bottom level variances are used to estimate the full hierarchy
        covariance. Use ``full_cov=True`` for more accurate estimation.

    Attributes
    ----------
    S : np.ndarray, shape (n, m)
        Full summing matrix :math:`S = [S_{\\mathrm{top}}^T, I_m]^T`.
    prediction : RRR or SRRR
        The underlying online prediction transformation.
    Y_bot : Source
        Source node for bottom-level observations.
    Y_hat : Source
        Source node for hierarchical forecasts.

    Notes
    -----
    The model is formulated as:

    .. math::

        Y_{\\mathrm{bot}} - \\hat{Y}_{\\mathrm{bot}} =
        (\\hat{Y}_{\\mathrm{top}} - \\hat{Y}_{\\mathrm{bot}} S_{\\mathrm{top}}^T)\\,\\theta + E,
        \\qquad E \\sim \\mathcal{MN}(0, I, \\Sigma_r)

    Estimation and prediction is posed as a special case of the ridge regression problem
    solved by the :class:`~prediction.RRR` predictor, with design matrix :math:`X` given
    by the incoherence at the top level and the target :math:`Y` given by the
    bottom-level base forecast errors. Reconciled forecasts are then constructed by
    shifting and scaling the predictions. Reconciled mean forecasts are computed as

    .. math::
        \\hat{Y}_{\\mathrm{rec}} = (\\hat{Y}_{\\mathrm{bot}} + \\Delta_{\\mathrm{bot}}) S^T,

    where :math:`\\Delta_{\\mathrm{bot}} = X \\theta` are the predicted bottom-level
    base forecast errors. Reconciled forecast covariance is computed as

    .. math::
        S \\tilde{\\Sigma}_{\\mathr{bot}} S^T,

    where :math:`\\tilde{\\Sigma}_{\\mathrm{bot}}` is the predicted covariance at the
    bottom level.

    See Also
    --------
    TemporalRidgeReconciliation : Wrapper for temporal hierarchies using backshift.
    prediction.RRR : Online recursive ridge regression used internally.
    SRRR : Ridge regression with optimal shrinkage prior.
    """

    def __init__(
        self,
        S_top,
        Y_bot=None,
        Y_hat=None,
        horizon=1,
        full_hierarchy_cov=False,
        opt_shrink=False,
        **kwargs,
    ):

        Y_bot = Y_bot or c.Source("Y_bot")
        Y_hat = Y_hat or c.Source("Y_hat")

        self.Y_bot = Y_bot
        self.Y_hat = Y_hat
        self.rec_err = c.Source("rec_err")

        self.S_top = S_top
        nsm, m = S_top.shape
        n = m + nsm
        self._n, self._m = n, m

        self.full_hierarchy_cov = full_hierarchy_cov
        self.S = np.vstack((S_top, np.eye(S_top.shape[1])))

        Y_hat_arr = f.ToArray(Y_hat)  # Ensure output of Y_hat is a numpy array.
        Y_hat_top = Y_hat_arr[:, :nsm]  # First n-m columns
        Y_hat_bot = Y_hat_arr[:, -m:]  # Last m columns

        # Incoherence at top level
        X = Y_hat_top - Y_hat_bot @ S_top.T

        # Bottom level forecast errors
        Y = Y_bot - f.Lag(Y_hat_bot, horizon)

        if opt_shrink:
            self.prediction = SRRR(X, Y, horizon, S_top, format_as_Y=False, **kwargs)
        else:
            self.prediction = p.RRR(X, Y, horizon, format_as_Y=False, **kwargs)

        super().__init__(Y_hat_bot, self.prediction)

    def evaluate(self, Y_hat_bot, pred):
        """Evaluate reconciled forecasts and covariance from bottom-level inputs.

        Parameters
        ----------
        Y_hat_bot : array-like of shape (t, n_bot)
            Bottom-level base forecasts.
        pred : dict
            Output from the underlying predictor with keys:
            ``"mean"`` (shape ``(t, n_bot)``) and ``"cov"`` (shape ``(t, n_bot)`` or
            ``(t, n_bot, n_bot)``).

        Returns
        -------
        dict
            Copy of ``pred`` with reconciled outputs:
            ``"mean"`` as shape ``(t, n_bot + n_top)``, and ``"cov"`` as shape
            ``(t, n_bot + n_top)`` or ``(t, n_top + n_bot, n_top + n_bot)`` depending on
            ``full_hierarchy_cov``.
        """
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
            var_bot_full[:, ran, ran] = var_bot[:, ran]

            var_bot = var_bot_full

        # Compute reconciled variance S V[x] S.T
        rec_var_est = self.S @ var_bot @ self.S.T

        # Fetch only diagonal? (TODO: consider doing self.S @ var_bot @ self.S.T more efficiently if only diagonal is needed)
        if not self.full_hierarchy_cov:
            rec_var_est = np.diagonal(rec_var_est, axis1=1, axis2=2)

        res["cov"] = rec_var_est

        return res

    @property
    def X(self):
        """Return the design matrix transformation from the underlying predictor."""
        return self.prediction.X

    @property
    def Y(self):
        """Return the target variable transformation from the underlying predictor."""
        return self.prediction.Y

    @property
    def predictor(self):
        """Return the underlying predictor."""
        return self.recursion_pars[self.prediction][0]

    def update(self, Y_bot, Y_hat, **kwargs):
        """Update the model state with bottom-level observations and forecasts.

        Parameters
        ----------
        Y_bot : array-like of shape (1, n_bot)
            Bottom-level observations.
        Y_hat : array-like of shape (1, n_bot)
            New bottom-level forecasts.
        **kwargs
            Additional keyword arguments passed to transformations.

        Returns
        -------
        dict
            Dictionariy with keys ``"mean"`` and ``"cov"`` containing reconciled
            forecasts and covariance predictions.
        """
        return self({self.Y_bot: Y_bot, self.Y_hat: Y_hat}, track_state=True, **kwargs)

    def fit(self, Y_bot, Y_hat, **kwargs):
        """Reset the model state and update.

        See ``update`` for parameter descriptions.
        """
        self.reset_state()
        return self({self.Y_bot: Y_bot, self.Y_hat: Y_hat}, **kwargs)

    @property
    def P(self):
        """Return the projection matrix for the reconciliation."""
        nsm, m = self.S_top.shape
        n = m + nsm
        theta = self.predictor.theta

        if theta is None:
            raise ValueError("Model not fitted yet")

        null = np.full((n - m, m), 0)
        P = (
            np.vstack((null, np.eye(m)))
            + np.vstack((np.eye(n - m), -self.S_top.T)) @ theta
        ).T
        return P


class TemporalRidgeReconciliation(RidgeReconciliation):
    r"""Wrapper for temporal hierarchies using backshift.

    This class is a thin wrapper around :class:`RidgeReconciliation` that uses a
    `BackShift` transformation to construct a bottom level variable for a temporal
    hierarchy.

    Parameters
    ----------
    S_top : np.ndarray, shape (n_top, n_bot)
        Summation matrix for top levels, i.e. the first ``n_top`` rows of the summation
        matrix, satisfying :math:`Y_{\\mathrm{top}}=S_\\mathrm{top} Y_{\\mathrm{bot}}`.
    B : list or dict
        Backshift structure for constructing bottom level observations.
        See :class:`~features.BackShift` for details.
    horizon : int, optional
        Forecast horizon used to align bottom-level observations with the
        corresponding forecasts when computing training targets. If ``None`` (default),
        the horizon is inferred from ``B`` as the maximum lag plus one.
    Z_bot : Source or None, optional
        Source providing unlagged bottom-level observations. If ``None``, a new Source
        named ``"Z_bot"`` is created. Bottom level observations for the full temporal
        hierarchy are constructed by applying a backshift transformation to this source.
    Y_hat : Source or None, optional
        Source providing hierarchical forecasts at all levels, with shape
        ``(t, n_top + n_bot)``. Top-level forecasts are assumed to occupy the first
        ``n_top`` columns and bottom-level forecasts the last ``n_bot`` columns. If
        ``None``, a new :class:`~core.Source` named ``"Y_hat"`` is created.
    skip_duplicates : bool, optional
        Whether to skip duplicate lagged observations for parameter estimation. Default
        is ``False``. See :class:`~features.BackShift` for details.
    full_hierarchy_cov : bool, optional
        If ``True``, the reconciled forecast covariance is returned as a full
        ``(t, n_top + n_bot, n_top + n_bot)`` array. If ``False`` (default), only the
        diagonal variances are returned as a ``(t, n_top + n_bot)`` array.
    opt_shrink : bool, optional
        If ``True``, uses :class:`SRRR` with optimal covariance shrinkage to
        compute the prior for the ridge regression. Default is ``False``.
    **kwargs
        Additional keyword arguments passed to the underlying :class:`~prediction.RRR`
        or :class:`SRRR` predictor.
    """

    def __init__(
        self,
        S_top,
        B: list | dict,
        horizon=None,
        Z_bot: c.Source = None,
        Y_hat: c.Source = None,
        skip_duplicates=False,
        full_hierarchy_cov=False,
        opt_shrink=False,
        **kwargs,
    ):
        if horizon is None:
            if isinstance(B, list):
                horizon = max(B) + 1
            elif isinstance(B, dict):
                horizon = max(B.values()) + 1

        # Construct backshift operator
        self.Z_bot = Z_bot or c.Source("Z_bot")
        Y_bot = p.BackShift(B, skip_duplicates=skip_duplicates, data=self.Z_bot)

        # Initialize Reconciler with lagged bottom level data as regressand input
        super().__init__(
            S_top, Y_bot, Y_hat, horizon, full_hierarchy_cov, opt_shrink, **kwargs
        )

    def update(self, Z_bot, Y_hat, **kwargs):
        """Update the model state with bottom-level observations and forecasts.

        Parameters
        ----------
        Z_bot : array-like of shape (t, features)
            Unlagged bottom-level observations.
        Y_hat : array-like of shape (t, n_top + n_bot)
            Base forecasts at all levels of the hierarchy.
        **kwargs
            Additional keyword arguments passed to transformations.

        Returns
        -------
        dict
            Dictionariy with keys ``"mean"`` and ``"cov"`` containing reconciled 
            forecasts and covariance predictions.
        """
        return self(
            {self.Z_bot: Z_bot, self.Y_hat: Y_hat}, track_state=True, **kwargs
        )

    def fit(self, Y_bot, Y_hat=None, **model_params):
        """Reset the model state and update.
        
        See ``update`` for parameter descriptions.
        """
        self.reset_state()
        return self.update(Y_bot, Y_hat, **model_params)


class SRRRPredictor(p.RRRPredictor):
    """Predictor for ridge regression with shrinkage priors.
    
    This predictor extends :class:`~prediction.RRRPredictor` to include shrinkage
    priors used in forecast reconciliation.

    See :class:`SRRR` for details.
    """

    def __init__(
        self,
        S_top,
        n,
        m,
        burn_in,
        init_K,
        track_memory,
        combine_variance,
        full_cov,
        center_cov=False,
    ):
        super().__init__(
            n,
            m,
            burn_in,
            init_K,
            track_memory,
            combine_variance,
            full_cov,
            center_cov=center_cov,
        )
        self.S_top = S_top
        npm = n + m
        self.sigma_bot_hat_d = np.zeros(m)
        self.sigma_top_hat_d = np.zeros(n)
        self.sigma_hat = np.zeros((npm, npm))
        self.var_sigma_hat = np.zeros((npm, npm))
        self.diag_mask = np.eye(npm, dtype=bool)

    def update_prior(self, x_i, y_i, mem, l_shrink):
        """Update the prior for ridge regression based on current error estimates.
        
        The class supports fixed or automatically estimated shrinkage parameters. When
        ``l_shrink`` is set to ``"auto"``, the shrinkage parameter is estimated based on
        optimal shrinkage estimation, see Bergsteinsson et al. (2021) for details.
         
        Parameters
        ----------
        x_i : ndarray of shape (n_features,)
            Current input features for prediction.
        y_i : ndarray of shape (n_targets,)
            Current target observation for model update.
        mem : float
            Forgetting factor for this update.
        l_shrink : float or "auto"
            If float, the shrinkage parameter for the ridge regression prior. If "auto",
            the shrinkage parameter is estimated from the current error estimates.

        References
        ----------
        Bergsteinsson, H.G., Moller, J.K., Nystrup, P., Palsson, O.P., Guericke, D.,
        Madsen, H., 2021. Heat load forecasting using adaptive temporal hierarchies.
        Applied Energy 292, 116872.
        https://doi.org/10.1016/j.apenergy.2021.116872
        """
        # Compute optimal Q and theta0 based on current error estimates
        err_top = self.S_top @ y_i - x_i

        if (
            l_shrink == "auto"
        ):  # Estimate shrinkage parameter using full covariance matrix

            if mem == 1:
                raise ValueError(
                    "Automatic shrinkage parameter estimation is not supported for no forgetting (mem=1)."
                )

            err_full = np.append(err_top, y_i)

            self.sigma_hat = mem * self.sigma_hat + (1 - mem) * np.outer(
                err_full, err_full
            )

            err_sq = err_full**2
            sigma_sq = self.sigma_hat**2

            # Update higer order variance estimates
            self.var_sigma_hat = (
                mem * (1 - mem) ** 2 * (np.outer(err_sq, err_sq) - sigma_sq)
                + mem**2 * self.var_sigma_hat
            )

            # Sum non-diagonal elements to estimate shrinkage parameter
            l_shrink = (
                np.where(self.diag_mask, 0, self.var_sigma_hat).sum()
                / np.where(self.diag_mask, 0, sigma_sq).sum()
            )

            cov_top_hat_d = np.diag(np.diag(self.sigma_hat[: self._n, : self._n]))
            cov_bot_hat_d = np.diag(np.diag(self.sigma_hat[self._n :, self._n :]))

        else:  # Estimate covariance diagonal

            # y_i contains bottom level errors
            self.sigma_bot_hat_d = self.sigma_bot_hat_d * mem + (1 - mem) * y_i**2

            # compute top level covariance from top level errors
            self.sigma_top_hat_d = self.sigma_top_hat_d * mem + (1 - mem) * err_top**2

            cov_top_hat_d = np.diag(self.sigma_top_hat_d)
            cov_bot_hat_d = np.diag(self.sigma_bot_hat_d)

        b0 = cov_top_hat_d + self.S_top @ cov_bot_hat_d @ self.S_top.T
        b1 = self.S_top @ cov_bot_hat_d

        Q = (
            1 / (1 - mem) * (l_shrink / (1 - l_shrink)) * b0
        )  # Check that this is correct. Consider using a better reperesentative of current memory than the limit 1/(1 - mem).

        theta0 = np.linalg.solve(b0, b1)

        return Q, theta0


class SRRR(p.RRR):

    def __init__(self, X, Y, horizon, S_top, **kwargs):
        super().__init__(X, Y, horizon, **kwargs)
        self.S_top = S_top

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
    ):
        """Create and return SRRR predictor."""
        return SRRRPredictor(
            self.S_top,
            n,
            m,
            burn_in,
            init_K,
            track_memory,
            combine_variance,
            full_cov,
            center_cov=center_cov,
        )

    def online_update(
        self,
        state: SRRRPredictor,
        x_i,
        y_i,
        x_train_i,
        mem=0.99,
        l_shrink="auto",
        **kwargs,
    ):
        """Delegate single-step update to `SRRRPredictor.online_update`.
        
        Parameters
        ----------
        state : SRRRPredictor
            Current state of the predictor.
        Otherwise same as `RRR.online_update`.

        Returns
        -------
        result : dict
            Prediction dictionary produced by `SRRRPredictor.online_update`.
        state : SRRRPredictor
            Updated predictor instance.
        """
        # Compute optimal Q and theta0 based on current error estimates
        Q, theta0 = state.update_prior(x_i, y_i, mem, l_shrink)  # x_i or x_train_i?

        # Call parent online_update with computed Q and theta0
        result = state.online_update(x_i, y_i, x_train_i, Q, theta0, mem=mem, **kwargs)

        # Return result and updated state
        return result, state


def _find_nodes(func):
    """Return wrapped function that finds replaces node names with matching objecs."""
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
    """Class for constructing hierarchies of variables.
    
    Parameters
    ----------
    sources : list of Node
        List of source nodes.
    name : str, optional
        Name of the node. If None (default), a unique name is generated automatically.
    """

    _counter = 0

    def __init__(self, *sources: "Node", name=None):
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
        """Combine two nodes into a new node with both as sources."""
        if not isinstance(other, Node):
            raise ValueError("Can only add Node to Node")
        return Node(self, other)

    def __radd__(self, other):
        """Combine two nodes into a new node with both as sources."""
        return self.__add__(other)

    def shift(self, h: int):
        """Return a lagged version of the node with lag h."""
        if len(self.sources) > 0:
            raise ValueError("Can only shift leaf nodes")
        if h < 0:
            raise ValueError("Shift must be non-negative")
        if h not in self.lags:
            self.lags[h] = LaggedNode(h, self)
        return self.lags[h]

    def is_parent(self, other):
        """Return True if self is a parent of other, False otherwise."""
        if other in self.sources:
            return True
        for src in self.sources:
            if src.is_parent(other):
                return True
        return False

    def get_nodes(self):
        """Return a list of all nodes in the hierarchy with self as root."""
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
        """Return a list of top nodes in the hierarchy with self as root."""
        bot_nodes = self.get_leaf_nodes()
        top_nodes = []
        for c in self.get_nodes():
            if c not in bot_nodes:
                top_nodes.append(c)
        return top_nodes

    @_find_nodes
    def build_B(self, *obs_vars):
        """Build backshift matrix structure B for temporal hierarchy.
        
        Parameters
        ----------
        obs_vars : list of Node or str
            List of observable variables in the hierarchy.        
        """
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
                    B[i, j] = 0
                elif temp in obs.lags.values():
                    B[i, j] = temp.lag
        return obs_nodes, B

    def get_leaf_nodes(self):
        """Return a list of leaf nodes in the hierarchy with self as root."""
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

    @_find_nodes
    def build_S_top(self, *bot_nodes):
        """Return top level summation matrix S_top for hierarchy.
        
        Parameters
        ----------
        bot_nodes : list of Node or str
            List of bottom-level nodes in the hierarchy.

        Returns
        -------
        top_nodes : list of Node
            List of top-level nodes in the hierarchy.
        S_top : np.ndarray of shape (n_top, n_bot)
            Summation matrix for top levels in terms of bottom-level nodes.
        """
        # Check that bot_nodes are the right ones
        if not set(bot_nodes) == set(self.get_leaf_nodes()):
            raise ValueError("Bottom nodes do not match hierarchy.")
        top_nodes = self.get_top_nodes()
        n, m = len(top_nodes), len(bot_nodes)
        S_top = np.zeros((n, m))
        for i, lat in enumerate(top_nodes):
            for j, obs in enumerate(bot_nodes):
                if lat.is_parent(obs):
                    S_top[i, j] = 1

        return top_nodes, S_top

    @_find_nodes
    def build_A_lat(self, *obs_nodes):
        """Return aggregation matrix for latent nodes in hierarchy.
        
        Parameters
        ----------
        obs_nodes : list of Node or str
            List of observable variables in the hierarchy.

        Returns
        -------
        lat_nodes : list of Node
            List of latent nodes in the hierarchy.
        A_lat : np.ndarray of shape (n_lat, n_obs)
            Aggregation matrix for latent nodes in terms of observable nodes.
        """
        # Get obs nodes if not provided
        bot_nodes = self.get_leaf_nodes()

        n_bot = len(bot_nodes)
        n_obs = len(obs_nodes)
        #        if n_bot != n_obs:
        #            raise ValueError("Number of observed nodes must match number of bottom nodes")
        # Get invers of K satisfying bot_nodes = K @ obs_nodes
        # obs_nodes = K_inv @ bot_nodes
        K_inv = np.zeros((n_obs, n_bot))
        for i, bot in enumerate(bot_nodes):
            for j, obs in enumerate(obs_nodes):
                if bot is obs or obs.is_parent(bot):
                    K_inv[j, i] = 1

        K = np.linalg.pinv(K_inv)  # inv if n = m?
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
                    C_top_lat[i, j] = 1

        C_bot_lat = np.zeros((n_bot, n_lat))
        for i, bot in enumerate(bot_nodes):
            for j, lat in enumerate(lat_nodes):
                if bot is lat:
                    C_bot_lat[i, j] = 1

        A_lat = (C_top_lat.T @ S_top + C_bot_lat.T) @ K
        return lat_nodes, A_lat.round().astype(int)

    def print_hierarchy(self, n=0):
        """Print the hierarchy structure starting from self as root."""
        indent = "  " * n
        print(f"{indent}- {self}")
        for node in self.sources:
            node.print_hierarchy(n + 1)

    def __repr__(self):
        """Return a string representation of the node."""
        return f"Node({self.name})"


class LaggedNode(Node):
    """Class for lagged nodes in a temporal hierarchy.
    
    Parameters
    ----------
    lag : int
        Lag of the node, must be non-negative.
    source : Node
        Source node for the lagged variable.
    """

    def __init__(self, lag: int, source: Node):
        if lag < 0:
            raise ValueError("Lag must be non-negative")
        name = f"{source.name}(-{lag})"
        super().__init__(name=name)
        self.lag = lag
        self.var = source

    def shift(self):
        """Return a further lagged version of the node with lag increased by 1."""
        return self.sources[0].shift(self.lag + 1)


def make_hierarchy(edges):
    """Construct a hierarchy of nodes from a dictionary of edges.
    
    Parameters
    ----------
    edges : dict
        Dictionary of edges defining the hierarchy, where keys are parent node names and
        values are lists of child node names or tuples of (child node name, lag) for
        temporal hierarchies. For example, for a simple hierarchy with top node "A" and
        bottom nodes "B" and "C", the edges could be defined as ``{"A": ["B", "C"]}``.
        For a temporal hierarchy with top node "A" and bottom node "B" lagged by 1, 
        the edges could be defined as ``{"A": [("B", 1)]}``. Nodes that are not parents
        (i.e. do not appear as keys) are assumed to be leaf nodes.

    Returns
    -------
    nodes : dict
        Dictionary mapping node names to Node instances.
    bot_nodes : list of Node
        List of bottom-level nodes in the hierarchy.
    top_nodes : list of Node
        List of top-level nodes in the hierarchy.
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