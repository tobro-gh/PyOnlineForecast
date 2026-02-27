
__version__ = "0.0.1"

from .core import (
    Source,
    MEMORY,
    DEFAULT_SOURCE,
    DEFAULT_INDEX,
    UPDATE_PREDICTOR,
    X_INIT,
    Y_INIT,
    Z_INIT
) 

from .features import (
    Combine,
    One,
    Lag,
    BackShift,
    LowPass,
    FourierSeries,
    SlidingSum,
    SlidingMean,
    ForgettingMean,
    ForgettingVariance,
    Subset
)

from .prediction import (
    Model,
    RRREnsemble,
    WLS,
    RRR,
    make_prediction_ensemble
)

def __getattr__(name):
    if name == "sample_data":
        from .datasets import data
        return data
    
__all__ = [
    "__version__",
    "Source",
    "MEMORY",
    "DEFAULT_SOURCE",
    "DEFAULT_INDEX",
    "UPDATE_PREDICTOR",
    "X_INIT",
    "Y_INIT",
    "Z_INIT",
    "ForecastMatrix",
    "DataFrame",
    "Combine",
    "One",
    "Lag",
    "BackShift",
    "LowPass",
    "FourierSeries",
    "SlidingSum",
    "SlidingMean",
    "ForgettingMean",
    "ForgettingVariance",
    "Subset",
    "Model",
    "RRREnsemble",
    "WLS",
    "RRR",
    "make_prediction_ensemble"
]
