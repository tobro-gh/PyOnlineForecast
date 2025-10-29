#%%
from py_online_forecast.core import *
from statsmodels.tsa.statespace.sarimax import SARIMAX as statsmodels_SARIMAX

class SARIMAXPredictor(OnlinePredictor):

    source_init_params = {"n": DimX}
    target = "1-step"

    def __init__(self, n, pred_horizon, order, seasonal_order, update_params_every=24, memory=168, 
                 extract_predictions = True, **kwargs):
        self.sarimax = None
        self.order = order
        self.seasonal_order = seasonal_order
        self.update_params_every = update_params_every
        self.step = 0
        self.memory = memory
        self.n_eff = n // pred_horizon # Effective number of exogenous variables per horizon

        self.pred_horizon = pred_horizon

        self.stored_X = CircularBuffer(self.memory, self.n_eff)
        self.stored_Y = CircularBuffer(self.memory, 1)


        self.extract_predictions = extract_predictions
        if self.extract_predictions:
            self.format = {"prediction": None, "ci_lower": None, "ci_upper": None}

        if "initialization" in kwargs:
            raise ValueError("Initialization for SARIMAXPredictor cannot be changed and is assumed 'known'.")

        self._initial_state = kwargs.pop("init_state", None)
        self._initial_state_cov = kwargs.pop("init_state_cov", None)

        self._kwargs = kwargs

        super().__init__()


    def update_model(self, x_i, y_i, y_i_hat):
        # Append data
        self.stored_X.append(x_i[:self.n_eff]) # Store only the 1-step exogenous data, i.e. first n_eff entries
        self.stored_Y.append(y_i)

        self.step = self.step + 1
        refit = self.step >= self.update_params_every
        self.step = self.step % self.update_params_every

        # If length of stored data is greater than update_params_every, fit the model
        if refit:
            Y_train = self.stored_Y.get_slice()
            X_train = self.stored_X.get_slice()
            # Check for nans
            if not (np.isnan(Y_train).any() or np.isnan(X_train).any()):
                sarimax = statsmodels_SARIMAX(Y_train, exog=X_train, order=self.order, seasonal_order=self.seasonal_order, **self._kwargs)

                # Initialize model
                if self.sarimax is None:
                    state = self._initial_state or np.zeros(sarimax.k_states)
                    state_cov = self._initial_state_cov or np.eye(sarimax.k_states)
                else:
                    state = self.sarimax.predicted_state[:, -1]
                    state_cov = self.sarimax.filtered_state_cov[:, :, -1]

                sarimax.initialize_known(state, state_cov)
                self.sarimax = sarimax.fit(disp=False)
        else:
            # Append data to sarimax results if model exists
            if self.sarimax is not None:
                self.sarimax = self.sarimax.append(endog=y_i, exog=np.atleast_2d(x_i[:self.n_eff]), refit=False)
    

    def predict(self, x_i, **params):
        # Reshape x_i from (n * horizon,) to (horizon, n)
        x_i = x_i.reshape(self.pred_horizon, self.n_eff)

        if self.sarimax is None:
            if self.extract_predictions:
                return {"1-step": np.array([np.nan]), "prediction": None, "ci_lower": None, "ci_upper": None}
            else:
                return {"1-step": np.array([np.nan]), "prediction": None}

        # Make prediction
        pred = self.sarimax.get_forecast(steps=self.pred_horizon, exog=x_i, **params)

        result = {"1-step": np.atleast_1d(pred.predicted_mean[0])}

        if self.extract_predictions:
            # Extract mean and confidence intervals
            result["prediction"] = np.atleast_1d(pred.predicted_mean)
            result["ci_lower"] = np.atleast_1d(pred.conf_int()[:,0])
            result["ci_upper"] = np.atleast_1d(pred.conf_int()[:,1])
        else:
            result["prediction"] = np.atleast_1d(pred.predicted_mean)

        return result

    def get_model_params(self):
        return self.sarimax.params if self.sarimax is not None else None


class SARIMAX(Model):

    def __init__(self, X, Y, order, seasonal_order, horizon, update_params_every=24, memory=168):
        X_sorted = ExogenousTransform(X, horizon)
        super().__init__(X_sorted, Y, SARIMAXPredictor, 1,
                         predictor_init_params={"pred_horizon": horizon, "order": order, "seasonal_order": seasonal_order, "update_params_every": update_params_every, "memory": memory})





# %%
