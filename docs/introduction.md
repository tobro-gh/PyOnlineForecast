# Introduction
The purpose of the `PyOnlineForecast` package is to provide a flexible framework for online forecasting and hierarchical reconciliation. The package splits forecasting into a design phase and an evaluation phase. In the design phase, the user should
1. Identify incoming data sources and define `Source` objects as placeholders for these.
2. Decide on appropriate data transformations and define `Transformation` sources by composing input sources or other transformations.
3. Configure a `Prediction` transformation to define estimation and prediction behaviour.

In the evaluation phase, the model may be applied in an online or offline setting. In either case, input data are mapped to their placeholder sources and transformations and predictions are computed.

### Forecast model
The package assumes a generic data generating process

$$
y_t = f_{\theta_t}(x_t, \epsilon_t).
$$

Here time indices $t$ are assumed discrete and regularly spaced, and the forecast horizon $k$ corresponds to a fixed number of steps. The data generating process $f$ depends on (time varying) parameters $\theta_t$ and maps input features $x_t$ and noise $\epsilon_t$ to outputs $y_t$. Predictions are made using a model $\phi$ and estimated parameters $\hat{\theta}$,

$$
\hat{y}_{t+k|t} = \phi_{\hat{\theta}_t}(x_t)
$$

### Online updates
In an online forecasting setting, model parameters should be updated frequently and in sequence. To do this, we will use a pair of functions, $\mathcal{T}$ for transforming raw data and $\mathcal{U}$ for updating our prediction model. Letting $\mathcal{D}_t$ be the data at time $t$ and $\mathcal{M}_{t-1}$ be a memory state of the model from time $t-1$, the updating procedure is

$$
\begin{aligned}
(x_t, y_t, \mathcal{M}_t) &= \mathcal{T}(\mathcal{D}_t, \mathcal{M}_{t-1}) \\
\hat{\theta}_t &= \mathcal{U}(\hat{\theta}_{t-1}, x_{t-k}, y_t, \hat{y}_{t|t-k})
\end{aligned}
$$

The data transformation first maps raw data and the memory state to input features, a target variable and an updated memory state. The update function then maps old parameters, transformed features, target variable and old forecasts to updated parameters. After updating parameters, the prediction function $\phi_{\hat{\theta}_t}$ is applied to obtain forecasts and the process is repeated once new data arrives.

## Sources and transformations
The separation of model design and evaluation is inspired by various modern software in which a computational graph is defined and lazily evaluated once data becomes available. This separation is enabled by the central `Source` and `Transformation` classes. 

```{eval-rst}
.. autosummary::
   :nosignatures:

   py_online_forecast.core.Source
   py_online_forecast.core.Transformation
```


Each instance of a `Source` represents a node in a computational graph, connected by transformations instantiated with other sources as inputs. Transformations are special sources that define a computational procedure which may be applied to input data. When data becomes available, a given transformation node can be evaluated by passing a `dict` of inputs specifying the available data as pairs of `Source` instances and their corresponding data. All dependencies are recursively evaluated, keeping a record of already evaluated transformations, and the final output is returned. In the design phase, transformations may be arbitrarily nested to form complete data processing pipelines. The evaluation of transformations is designed to depend on 

1. the input data sources, 
2. parameters defined at instantiation, and
3. a memory state that is passed along with the input data.

The output of the evaluation is the transformed data and an updated state. As a design principle to ensure predictability, evaluation of a transformation should be a pure function with respect to the input data, parameters and state.

## Prediction
The `Prediction` class is a base class for prediction algorithms, such as the built in `RRR`.
```{eval-rst}
.. autosummary::
   :nosignatures:

   py_online_forecast.prediction.Prediction
   py_online_forecast.prediction.RRR   
```
The `Prediction` supplies basic functionality for lagging inputs...

The `RRR` class provides a simple and robust prediction algorithm. 

## Hierarchical reconciliation

### Temporal hierarchies