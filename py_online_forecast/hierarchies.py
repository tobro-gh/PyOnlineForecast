"""Transformations and tools for hierarchical forecast reconciliation.

This module provides transformations and tools for working with hierarchical forecast
reconciliation. The main class is ``RidgeReconciliation``, which uses the ``RRR`` (or
optionally ``SRRR``) prediction to model the bottom level forecast errors and then
constructs reconciled forecasts at all levels of the hierarchy. Behind the scenes, these
classes use the ``HierarchicalForecastReconciliation`` transformation, which provides
a generic way to setup forecast reconciliation using any prediction model.

For working with temporal hierarchies, the ``TemporalRidgeReconciliation`` class is a
thin wrapper around ``RidgeReconciliation`` that uses a ``BackShift`` transformation to
construct a bottom level variable for a temporal hierarchy.

The module also includes some tools for constructing hierarchies using a ``Node`` class.
"""

import numpy as np

from . import core as c
from . import features as f
from . import prediction as p


class HierarchicalForecastReconciliation(c.Transformation):
    r"""Hierarchical forecast reconciliation using predictions of bottom-level errors.

    This transformation performs hierarchical forecast reconciliation by predicting
    bottom-level base forecast errors from top-level incoherence using an arbitrary
    prediction model and then constructing reconciled forecasts by shifting and
    aggregating the predicted bottom-level errors.

    Parameters
    ----------
    S_top : np.ndarray, shape (n_top, n_bot)
        Summation matrix for top levels, i.e. the first ``n_top`` rows of the summation
        matrix, satisfying :math:`Y_{\text{top}} = S_{\text{top}} Y_{\text{bot}}`.
    prediction_factory : callable
        Function to create the prediction model. The function should take exactly two
        arguments, the design matrix :math:`X` and response variable :math:`Y` of the
        hierarchical regression problem, and return a transformation that produces
        predictions of the bottom-level errors at the given horizon, when called with
        appropriate inputs. The design and response variables are provided as
        transformations, and the output of the prediction model should be a dictionary
        containing at least the key specified by ``mean_key`` with the predicted
        bottom-level errors. If covariance predictions are also provided, the
        dictionary should also contain the key specified by ``cov_key``.
    Y_bot : Source or None, optional
        Bottom-level observations.
    Y_hat : Source or None, optional
        Bottom-level forecasts.
    horizon : int, default=1
        Forecast horizon.
    mean_key : any, default="mean"
        Key for accessing the mean predictions.
    cov_key : any, optional
        Key for accessing the covariance predictions.

    Attributes
    ----------
    S : np.ndarray, shape (n_top + n_bot, n_bot)
        Summation matrix.
    prediction : Transformation
        The underlying prediction transformation for bottom-level errors.
    Y_bot : Source
        Source node for bottom-level observations.
    Y_hat : Source
        Source node for hierarchical forecasts.

    Notes
    -----
    For a summation matrix :math:`S`, bottom-level observations :math:`Y_{\text{bot}}`,
    and base forecasts at the top and bottom levels :math:`\hat{Y}_{\text{top}}` and
    :math:`\hat{Y}_{\text{bot}}`, the regression model is given by

    .. math::

        Y_{\text{bot}} - \hat{Y}_{\text{bot}} =
        (\hat{Y}_{\text{top}} - \hat{Y}_{\text{bot}} S_{\text{top}}^T)\,\theta + E

    where :math:`\theta` are the regression parameters, and :math:`E` is a  noise term.
    Given a prediction of the bottom-level errors, the reconciled forecasts are 
    constructed by shifting and scaling the predictions. Given a forecast of the mean of
    the bottom-level errors :math:`\Delta_{\text{bot}}`, the reconciled mean forecasts
    are computed as

    .. math::
        \hat{Y}_{\text{rec}} = (\hat{Y}_{\text{bot}} + \Delta_{\text{bot}}) S^T,

    Given a forecast of the covariance of the bottom-level errors
    :math:`\Sigma_{\text{bot}}`, reconciled forecast covariance is computed as

    .. math::
        S \tilde{\Sigma}_{\text{bot}} S^T.
    """

    def __init__(
        self,
        S_top,
        prediction_factory,
        Y_bot=None,
        Y_hat=None,
        horizon=1,
        mean_key="mean",
        cov_key=None,
        full_cov=False,
    ):
        Y_bot = Y_bot or c.Source("Y_bot")
        Y_hat = Y_hat or c.Source("Y_hat")

        # Store inputs
        self.S_top = S_top
        self.S = np.vstack((S_top, np.eye(S_top.shape[1])))
        self.Y_bot = Y_bot
        self.Y_hat = Y_hat

        # Construct the regression problem
        nsm, m = S_top.shape

        Y_hat_arr = f.ToArray(Y_hat)  # Ensure output of Y_hat is a numpy array.
        Y_hat_top = Y_hat_arr[:, :nsm]  # First n-m columns
        Y_hat_bot = Y_hat_arr[:, -m:]  # Last m columns

        # Incoherence at top level
        X = Y_hat_top - Y_hat_bot @ S_top.T

        # Bottom level forecast errors
        Y = Y_bot - f.Lag(Y_hat_bot, horizon)

        # Construct the prediciton of bottom level errors
        self.prediction = prediction_factory(X, Y)

        # Get the predicted bottom level errors
        err_bot_hat = self.prediction[mean_key]

        # Construct the bottom level reconciled forecasts
        Y_hat_rec_bot = Y_hat_bot + err_bot_hat

        # Aggregate the reconciled forecasts to all levels of the hierarchy.
        Y_hat_rec = Y_hat_rec_bot @ self.S.T

        # Include covariance if provided.
        self.full_cov = full_cov
        if cov_key is None:
            super().__init__(self.prediction, Y_hat_rec)
        else:
            cov_bot = self.prediction[cov_key]
            super().__init__(self.prediction, Y_hat_rec, cov_bot=cov_bot)

    def evaluate(self, prediction, Y_hat_rec, cov_bot=None):
        """Return the reconciled results and original predictions.

        Parameters
        ----------
        prediction : dict
            The output of the underlying prediction transformation, containing at least
            the key specified by ``mean_key`` with the predicted bottom-level errors.
            If covariance predictions are also provided, the dictionary should also
            contain the key specified by ``cov_key``.
        Y_hat_rec : array of shape (t, n_top + n_bot)
            Reconciled forecasts at all levels of the hierarchy.
        cov_bot : array, optional
            Predicted covariance at the bottom level, with shape either (t, n_bot) if
            only diagonal variances are provided or (t, n_bot, n_bot) if full covariance
            matrices are provided. If not provided, covariance is not included in the
            output.

        Returns
        -------
        dict
            Dictionariy with keys ``"mean"`` and ``"cov"`` and
            ``"bottom_level_prediction"`` containing reconciled forecasts, covariance
            predictions, and original bottom-level error predictions
        """
        result = {"bottom_level_prediction": prediction, "mean": Y_hat_rec}

        # Aggregate covariance if provided
        if cov_bot is not None:
            if cov_bot.ndim == 2:

                t, n = cov_bot.shape

                cov_bot_full = np.full((t, n, n), np.nan)

                # Get mask for valid rows
                valid_rows = ~np.any(np.isnan(cov_bot), axis=1)

                # Initialise valid rows to zero
                cov_bot_full[valid_rows] = 0

                # Assign diagonal elements
                ran = np.arange(n)
                cov_bot_full[:, ran, ran] = cov_bot[:, ran]

                cov_bot = cov_bot_full

            # Compute reconciled variance S V[x] S.T
            rec_var_est = self.S @ cov_bot @ self.S.T

            # Fetch only diagonal? (TODO: consider doing self.S @ cov_bot @ self.S.T more efficiently if only diagonal is needed)
            if not self.full_cov:
                rec_var_est = np.diagonal(rec_var_est, axis1=1, axis2=2)

            result["cov"] = rec_var_est

        return result

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
            Dictionariy with keys ``"mean"`` and ``"cov"`` and
            ``"bottom_level_prediction"`` containing reconciled forecasts, covariance
            predictions, and original bottom-level error predictions.
        """
        return self({self.Y_bot: Y_bot, self.Y_hat: Y_hat}, track_state=True, **kwargs)

    def fit(self, Y_bot, Y_hat, **kwargs):
        """Reset the model state and update.

        See ``update`` for parameter descriptions.
        """
        self.reset_state()
        return self({self.Y_bot: Y_bot, self.Y_hat: Y_hat}, **kwargs)

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
        horizon,
        burn_in,
        init_K,
        track_memory,
        combine_variance,
        full_cov,
        center_cov=False,
        mem=0.99,
    ):
        super().__init__(
            n,
            m,
            horizon,
            burn_in,
            init_K,
            track_memory,
            combine_variance,
            full_cov,
            center_cov=center_cov,
            mem=mem,
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

            cov_top_hat_d = np.diag(np.diag(self.sigma_hat[: self.n, : self.n]))
            cov_bot_hat_d = np.diag(np.diag(self.sigma_hat[self.n :, self.n :]))

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
        )  # Consider using a better reperesentative of current memory than the limit 1/(1 - mem).

        theta0 = np.linalg.solve(b0, b1)

        return Q, theta0

    def online_update(
        self,
        x_i,
        y_i,
        x_train_i,
        V=None,
        update_V=True,
        mem=None,
        return_var_theta=False,
        update_theta=True,
        l_shrink="auto",
    ):
        """Return the online update for the predictor using shrinkage priors."""
        if not (np.isnan(x_i).any() or np.isnan(y_i).any()):
            Q, theta0 = self.update_prior(x_i, y_i, mem, l_shrink)
        else:
            update_theta = False
            Q, theta0 = None, None
        return super().online_update(
            x_i,
            y_i,
            x_train_i,
            Q,
            theta0,
            V,
            update_V,
            mem,
            return_var_theta,
            update_theta,
        )

class SRRR(p.RRR):
    r"""Online recursive ridge regression with hierarchical shrinkage priors.

    This wrapper around :class:`~prediction.RRR` uses :class:`SRRRPredictor` to
    perform online recursive ridge regression with shrinkage priors for hierarchical
    forecast reconciliation.

    The prior is updated from the estimated top- and bottom-level error covariance
    structure implied by

    .. math::

        Y_{\text{top}} = S_{\text{top}} Y_{\text{bot}}.

    The resulting prior parameters are then passed to the recursive ridge update.

    Parameters
    ----------
    S_top : np.ndarray, shape (n_top, n_bot)
        Summation matrix for top levels, i.e. the first ``n_top`` rows of the summation
        matrix, satisfying :math:`Y_{\text{top}} = S_{\text{top}} Y_{\text{bot}}`.
    **kwargs
        Additional keyword arguments passed to :class:`~prediction.RRR`.

    See Also
    --------
    prediction.RRR : Base recursive ridge regression model.
    SRRRPredictor : Predictor that updates the shrinkage prior online.
    """

    def __init__(self, X, Y, horizon, S_top, **kwargs):
        super().__init__(X, Y, horizon, **kwargs)
        self.S_top = S_top

    def online_update(
        self,
        state: SRRRPredictor,
        x_i,
        y_i,
        x_train_i,
        V=None,
        update_V=True,
        mem=0.99,
        return_var_theta=False,
        update_theta=True,
    ):
        """Return the online update for the predictor using shrinkage priors."""
        result = state.online_update(
            x_i,
            y_i,
            x_train_i=x_train_i,
            V=V,
            update_V=update_V,
            mem=mem,
            return_var_theta=return_var_theta,
            update_theta=update_theta,
        )
        return result, state

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
        mem,
    ):
        """Create and return SRRR predictor."""
        return SRRRPredictor(
            self.S_top,
            n,
            m,
            self.horizon,
            burn_in,
            init_K,
            track_memory,
            combine_variance,
            full_cov,
            center_cov,
            mem,
        )


class _TemporalRRRPredictor(p.RRRPredictor):
    """Wrapper to skip duplicate data updates for temporal hierarchies.
    
    This class is needed to get correct signatures and handling for RRRPredictor.
    """

    def __init__(self, n, m, horizon, *args, **kwargs):
        super().__init__(n, m, horizon, *args, **kwargs)
        self.horizon = horizon
        self.offset = 0

    def online_update(
        self,
        x_i,
        y_i,
        x_train_i,
        Q=None,
        theta0=None,
        V=None,
        update_V=True,
        mem=None,
        return_var_theta=False,
    ):
        """Return the online update for the predictor, skipping duplicate data."""
        update_theta = self.offset == 0
        result = super().online_update(
            x_i,
            y_i,
            x_train_i,
            Q,
            theta0,
            V,
            update_V,
            mem,
            return_var_theta,
            update_theta,
        )
        self.offset = (self.offset + 1) % self.horizon
        return result


class _TemporalSRRRPredictor(SRRRPredictor):
    """Wrapper to skip duplicate data updates for temporal hierarchies.
    
    This class is needed to get correct signatures and handling for SRRRPredictor.
    """

    def __init__(self, S_top, n, m, horizon, *args, **kwargs):
        super().__init__(S_top, n, m, horizon, *args, **kwargs)
        self.horizon = horizon
        self.offset = 0

    def online_update(
        self,
        x_i,
        y_i,
        x_train_i,
        V=None,
        update_V=True,
        mem=None,
        return_var_theta=False,
        l_shrink="auto",
    ):
        """Return the online update for the predictor, skipping duplicate data."""
        update_theta = self.offset == 0
        result = super().online_update(
            x_i,
            y_i,
            x_train_i,
            V,
            update_V,
            mem,
            return_var_theta,
            update_theta,
            l_shrink,
        )
        self.offset = (self.offset + 1) % self.horizon
        return result


class _TemporalRRR(p.RRR):
    """Wrapper to skip duplicate data updates for temporal hierarchies.
    
    This class is needed to get correct signatures and handling for RRR.
    """

    def online_update(
        self,
        state: _TemporalRRRPredictor,
        x_i,
        y_i,
        x_train_i,
        Q=None,
        theta0=None,
        V=None,
        update_V=True,
        mem=0.99,
        return_var_theta=False,
    ):
        result = state.online_update(
            x_i, y_i, x_train_i, Q, theta0, V, update_V, mem, return_var_theta
        )
        return result, state

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
        mem,
    ):
        return _TemporalRRRPredictor(
            n,
            m,
            self.horizon,
            burn_in,
            init_K,
            track_memory,
            combine_variance,
            full_cov,
            center_cov,
            mem,
        )


class _TemporalSRRR(SRRR):
    """Wrapper to skip duplicate data updates for temporal hierarchies.
    
    This class is needed to get correct signatures and handling for SRRR.
    """
    
    def online_update(
        self,
        state: _TemporalSRRRPredictor,
        x_i,
        y_i,
        x_train_i,
        V=None,
        update_V=True,
        mem=None,
        return_var_theta=False,
        l_shrink="auto",
    ):
        result = state.online_update(
            x_i, y_i, x_train_i, V, update_V, mem, return_var_theta, l_shrink
        )
        return result, state

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
        mem,
    ):
        return _TemporalSRRRPredictor(
            self.S_top,
            n,
            m,
            self.horizon,
            burn_in,
            init_K,
            track_memory,
            combine_variance,
            full_cov,
            center_cov,
            mem,
        )


class RidgeReconciliation(HierarchicalForecastReconciliation):
    r"""Hierarchical forecast reconciliation using ridge regression.

    Reconciles forecasts across a hierarchy by predicting bottom-level base forecast
    errors using a linear model in the top-level coherency errors. The transformation
    uses :class:`~prediction.RRR` or optionally :class:`SRRR` to perform online
    estimation and prediction at the bottom level and constructs reconciled forecasts
    by shifting and scaling the base forecast error predictions. When 
    ``temporal_skip=True``, wrapper classes are used internally instead, to restrict
    parameter updates to every ``horizon`` steps, which may be desirable for temporal
    hierarchies to avoid resuing observations for training.

    Parameters
    ----------
    S_top : np.ndarray, shape (n_top, n_bot)
        Summation matrix for top levels, i.e. the first ``n_top`` rows of the summation
        matrix, satisfying :math:`Y_{\text{top}} = S_{\text{top}} Y_{\text{bot}}`.
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
    temporal_skip : bool, optional
        If ``True``, the model parameters are only updated every ``horizon`` steps. May
        be used for temporal hierarchies, to avoid reusing observations for training.
    **kwargs
        Additional keyword arguments passed to the underlying
        :class:`~prediction.RRR` or :class:`SRRR` predictor (e.g. ``mem``,
        ``burn_in``, ``init_K``, ``full_cov``). Note in particular when ``full_cov`` is
        False, only bottom level variances are used to estimate the full hierarchy
        covariance. Use ``full_cov=True`` for more accurate estimation.

    Notes
    -----
    The model is formulated as:

    .. math::

        Y_{\text{bot}} - \hat{Y}_{\text{bot}} =
        (\hat{Y}_{\text{top}} - \hat{Y}_{\text{bot}} S_{\text{top}}^T)\,\theta + E,
        \qquad E \sim \mathcal{MN}(0, I, \Sigma_r)

    Estimation and prediction is posed as a special case of the ridge regression problem
    solved by the :class:`~prediction.RRR` predictor, with design matrix :math:`X` given
    by the incoherence at the top level and the target :math:`Y` given by the
    bottom-level base forecast errors. Reconciled forecasts are then constructed by
    shifting and scaling the predictions. Reconciled mean forecasts are computed as

    .. math::
        \hat{Y}_{\text{rec}} = (\hat{Y}_{\text{bot}} + \Delta_{\text{bot}}) S^T,

    where :math:`\Delta_{\text{bot}} = X \theta` are the predicted bottom-level
    base forecast errors. Reconciled forecast covariance is computed as

    .. math::
        S \tilde{\Sigma}_{\text{bot}} S^T,

    where :math:`\tilde{\Sigma}_{\text{bot}}` is the predicted covariance at the
    bottom level.

    See Also
    --------
    HierarchicalForecastReconciliation : General hierarchical forecast reconciliation
    using arbitrary prediction models.
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
        temporal_skip=False,
        **kwargs,
    ):

        def prediction_factory(X, Y):
            if opt_shrink:
                if temporal_skip:
                    return _TemporalSRRR(X, Y, horizon, S_top, **kwargs)
                else:
                    return SRRR(X, Y, horizon, S_top, **kwargs)
            else:
                if temporal_skip:
                    return _TemporalRRR(X, Y, horizon, **kwargs)
                else:
                    return p.RRR(X, Y, horizon, **kwargs)

        super().__init__(
            S_top,
            prediction_factory,
            Y_bot,
            Y_hat,
            horizon,
            mean_key="mean",
            cov_key="cov",
            full_cov=full_hierarchy_cov,
        )

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
    ``BackShift`` transformation to construct a bottom level variable for a temporal
    hierarchy.

    Parameters
    ----------
    S_top : np.ndarray, shape (n_top, n_bot)
        Summation matrix for top levels, i.e. the first ``n_top`` rows of the summation
        matrix, satisfying :math:`Y_{\text{top}} = S_{\text{top}} Y_{\text{bot}}`.
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
        Y_bot = p.BackShift(B, data=self.Z_bot)

        super().__init__(
            S_top,
            Y_bot=Y_bot,
            Y_hat=Y_hat,
            horizon=horizon,
            full_hierarchy_cov=full_hierarchy_cov,
            opt_shrink=opt_shrink,
            temporal_skip=skip_duplicates,
            **kwargs,
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
        return self({self.Z_bot: Z_bot, self.Y_hat: Y_hat}, track_state=True, **kwargs)

    def fit(self, Y_bot, Y_hat=None, **model_params):
        """Reset the model state and update.

        See ``update`` for parameter descriptions.
        """
        self.reset_state()
        return self.update(Y_bot, Y_hat, **model_params)


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

    def find_node(self, name):
        """Find node with given name in the hierarchy."""
        return next(n for n in self.get_nodes() if n.name == name)

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

    def get_nodes(self, nodes=None):
        """Return a list of all nodes in the hierarchy with self as root.

        Parameters
        ----------
        nodes : list of Node or None, optional
            List of nodes to consider when determining nodes. If None (default), all
            nodes in the hierarchy are considered. If provided, also specifies the
            ordering of the nodes in the output.
        """
        all_nodes = [self]
        current_nodes = all_nodes
        while len(current_nodes) > 0:
            new_nodes = []
            for node in current_nodes:
                new_nodes.extend(node.sources)
            all_nodes.extend(new_nodes)
            current_nodes = new_nodes

        if nodes is not None:
            # Find nodes if strings
            nodes = [self.find_node(n) if isinstance(n, str) else n for n in nodes]
            all_nodes = [n for n in nodes if n in all_nodes]

        return all_nodes

    def get_bot_nodes(self, nodes=None):
        """Return a list of leaf nodes in the hierarchy with self as root.

        Parameters
        ----------
        nodes : list of Node or None, optional
            List of nodes to consider when determining leaf nodes. If None (default),
            all nodes in the hierarchy are considered. If provided, also specifies the
            ordering of the nodes in the output.
        """
        # Get all leaf nodes (nodes without sources)
        if not self.sources:
            return [self]
        else:
            leafs = []
            for src in self.sources:
                # Add leaves not already in list
                for leaf in src.get_bot_nodes():
                    if leaf not in leafs:
                        leafs.append(leaf)

            if nodes is not None:
                # Find nodes if strings
                nodes = [self.find_node(n) if isinstance(n, str) else n for n in nodes]
                leafs = [n for n in nodes if n in leafs]

            return leafs

    def get_top_nodes(self, nodes=None):
        """Return a list of top nodes in the hierarchy with self as root.

        Parameters
        ----------
        nodes : list of Node or None, optional
            List of nodes to consider when determining top nodes. If None (default), all
            nodes in the hierarchy are considered. If provided, also specifies the
            ordering of the nodes in the output.

        """
        bot_nodes = self.get_bot_nodes()
        top_nodes = []
        for n in self.get_nodes():
            if n not in bot_nodes:
                top_nodes.append(n)

        if nodes is not None:
            # Find nodes if strings
            nodes = [self.find_node(n) if isinstance(n, str) else n for n in nodes]
            top_nodes = [n for n in nodes if n in top_nodes]

        return top_nodes

    def get_remaining_nodes(self, obs_nodes):
        """Return a list of remaining nodes in order.

        The function returns all nodes in the hierarchy that are not in the given list
        of observed nodes, in the order they appear in the hierarchy, as if returned by
        get_nodes().

        Parameters
        ----------
        obs_nodes : list of Node
            List of observed nodes to determine latent nodes.
        """
        all_nodes = self.get_nodes()
        remaining_nodes = [n for n in all_nodes if n not in obs_nodes]
        return remaining_nodes

    def build_summation_matrix(self, full=False, nodes=None):
        """Return summation matrix for hierarchy.

        Parameters
        ----------
        full : bool, optional
            Whether to return the full summation matrix including the identity for
            observed (bottom) nodes. If False (default), only the latent (top) nodes
            are included in the output.
        nodes : list of Node or None, optional
            List of nodes to include in the summation matrix. If None (default), all
            nodes in the hierarchy are included. If provided, also specifies the
            ordering of the nodes in the summation matrix.

        Returns
        -------
        S : np.ndarray of shape (n_top, n_bot) or (n_top + n_bot, n_bot)
            Summation matrix for top levels (if full=False) or all nodes (if full=True).
        """
        bot_nodes = self.get_bot_nodes()
        top_nodes = self.get_top_nodes()

        if nodes is not None:
            # Find nodes if strings
            nodes = [self.find_node(n) if isinstance(n, str) else n for n in nodes]

            bot_nodes = [n for n in nodes if n in bot_nodes]
            top_nodes = [n for n in nodes if n in top_nodes]

        return build_summation_matrix(bot_nodes, top_nodes, full=full)

    def build_aggregation_matrix(self, obs_nodes, lat_nodes=None, full=False):
        """Return aggregation matrix ordered by obs_nodes and lat_nodes.

        Parameters
        ----------
        obs_nodes : list of Node
            List of observable nodes to include.
        lat_nodes : list of Node or None, optional
            List of latent nodes to include. If None (default), all remaining
            descendants of self that are not in obs_nodes are used.
        full : bool, optional
            Whether to return the full aggregation matrix including the identity for
            observed (bottom) nodes. If False (default), only the latent (top) nodes
            are included in the output.

        Returns
        -------
        A : np.ndarray of shape (n_lat, n_obs) or (n_lat + n_obs, n_obs)
            Aggregation matrix for latent nodes (if full=False) or all nodes (if
            full=True).
        """
        # Find obs nodes if any are strings
        obs_nodes = [self.find_node(o) if isinstance(o, str) else o for o in obs_nodes]

        if lat_nodes is None:
            # Get latent nodes if not provided
            lat_nodes = [n for n in self.get_nodes() if not n in obs_nodes]
        else:
            # Find lat nodes if any are strings
            lat_nodes = [
                self.find_node(l) if isinstance(l, str) else l for l in lat_nodes
            ]

            # Check that lat nodes don't include obs nodes
            for lat in lat_nodes:
                if lat in obs_nodes:
                    raise ValueError("Latent nodes cannot include observed nodes")

        # Get a list of all nodes to be used (not necessarily all descendants of self)
        all_nodes = obs_nodes + lat_nodes

        # Get bottom and top level nodes
        bot_nodes = [node for node in self.get_bot_nodes() if node in all_nodes]
        top_nodes = [node for node in self.get_top_nodes() if node in all_nodes]

        # Build summation matrix from selected nodes
        S = build_summation_matrix(bot_nodes, top_nodes, full=True)

        # Build lists of old and new nodes for permutation matrix
        old_nodes = top_nodes + bot_nodes
        new_nodes = lat_nodes + obs_nodes

        # Build permutation matrix to reorder from old_nodes to new_nodes
        P = build_permutation_matrix(old_nodes, new_nodes)

        # Construct and return aggegation matrix
        return construct_aggregation_matrix(S, P, full=full)

    def build_backshift(self, output_nodes, input_nodes=None):
        """Return backshift structure for given output nodes.

        Parameters
        ----------
        output_nodes : list of Node
            List of output nodes to determine backshift structure.
        input_nodes : list of Node or None, optional
            List of input nodes to determine backshift structure. If None (default), all
            required input nodes are inferred using get_input_nodes(output_nodes).
        """
        output_nodes = [
            self.find_node(o) if isinstance(o, str) else o for o in output_nodes
        ]

        if input_nodes is None:
            input_nodes = get_input_nodes(output_nodes)
        else:
            input_nodes = [
                self.find_node(i) if isinstance(i, str) else i for i in input_nodes
            ]

        return build_backshift(input_nodes, output_nodes)

    def print_hierarchy(self, n=0):
        """Print the hierarchy structure starting from self as root."""
        indent = "  " * n
        print(f"{indent}- {self}")
        for node in self.sources:
            node.print_hierarchy(n + 1)

    def __repr__(self):
        """Return a string representation of the node."""
        return f"Node({self.name})"


def build_summation_matrix(bot_nodes, top_nodes, full=False):
    """Return summation matrix for hierarchy.

    Parameters
    ----------
    bot_nodes : list of Node or str
        List of observable (bottom) nodes to include in the summation matrix.
    top_nodes : list of Node or str
        List of latent (top) nodes to include in the summation matrix.

    full : bool, optional
        Whether to return the full summation matrix including the identity for
        observed (bottom) nodes. If False (default), only the latent (top) nodes
        are included.

    Returns
    -------
    S : np.ndarray of shape (m, n_bot)
        Summation matrix for latent nodes (if full=False) with m = n_lat, or for all
        nodes (if full=True) with m = n_lat + n_bot.
    """
    S = np.zeros((len(top_nodes), len(bot_nodes)))
    for i, lat in enumerate(top_nodes):
        for j, obs in enumerate(bot_nodes):
            if lat.is_parent(obs):
                S[i, j] = 1
    if full:
        S = np.vstack((S, np.eye(len(bot_nodes))))

    return S


def build_permutation_matrix(old_nodes, new_nodes):
    """Return permutation matrix for reordering nodes in hierarchy.

    Parameters
    ----------
    old_nodes : list of Node
        List of nodes in the original order.
    new_nodes : list of Node
        List of nodes in the new order.

    Returns
    -------
    C : np.ndarray of shape (n, n)
        Permutation matrix such that x_new = C @ x_old, where x_old and x_new are state
        vectors of the nodes in the old and new order, respectively.
    """
    # Check that old and new nodes contain the same nodes
    if set(old_nodes) != set(new_nodes):
        raise ValueError("Old and new nodes must contain the same nodes")
    n = len(old_nodes)
    C = np.zeros((n, n))
    for i, new in enumerate(new_nodes):
        for j, old in enumerate(old_nodes):
            if new is old:
                C[i, j] = 1
    return C


def construct_aggregation_matrix(S, P, full=False):
    """Return aggregation matrix for latent nodes in hierarchy.

    Parameters
    ----------
    S : np.ndarray of shape (n_tot, n_bot)
        Summation matrix for latent nodes in terms of observable nodes.
    P : np.ndarray of shape (n_tot, n_tot)
        Permutation matrix for reordering top and bottom level nodes to latent and
        observable nodes.

    Returns
    -------
    A: np.ndarray of shape (m, n_bot)
        Aggregation matrix for latent nodes in terms of observable nodes. If full=False
        (default), only the latent nodes are included, with m = n_top. If full=True, all
        nodes are included, with m = n_tot.
    """
    n_tot, n_bot = S.shape
    n_top = n_tot - n_bot
    P_lat = P[:n_top]
    P_obs = P[n_top:]

    #    A = P_lat @ S @ np.linalg.inv(P_obs @ S)
    tmp1 = P_lat @ S
    tmp2 = P_obs @ S
    A = np.linalg.solve(tmp2.T, tmp1.T).T

    if full:
        A = np.vstack((A, np.eye(n_bot)))

    return A.round().astype(int)


def get_input_nodes(output_nodes):
    """Return a list of required input nodes for given output nodes.

    Parameters
    ----------
    output_nodes : list of Node or None, optional
        List of output nodes to determine required input nodes.
    """
    input_nodes = []
    for out in output_nodes:
        if isinstance(out, LaggedNode):
            out = out.var
        if out not in input_nodes:
            input_nodes.append(out)

    return input_nodes


def build_backshift(input_nodes, output_nodes):
    """Return backshift matrix structure for temporal hierarchy.

    Constructs a backshift operator B such that y_bot = B z, where z are the states
    of the input nodes. The output is a list of pairs (i_k, j_k), k = 1, ..., m
    specifying that the i_k-th input state is lagged by j_k steps to construct the
    k-th output variable.

    Parameters
    ----------
    input_nodes : list of Node
        List of input nodes in the hierarchy.
    output_nodes : list of Node
        List of output nodes in the hierarchy.
    """
    B = []
    for out_node in output_nodes:

        if isinstance(out_node, LaggedNode):
            in_node = input_nodes.index(out_node.var)
            lag = out_node.lag
        else:
            in_node = input_nodes.index(out_node)
            lag = 0

        B.append((in_node, lag))

    return B

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

    return nodes
