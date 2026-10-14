"""A package for online forecasting and reconciliation.

The package provides tools for building data pipelines for online forecasting models.
The package is built around ``Transformation`` objects that define nodes in a
computational graph. Any transformation in the graph may be evaluated once data is
available for all its dependencies. Nodes nested beneath the called root transformation
are evaluated recursively if their values are not provided. The package mostly uses 
`numpy` for data handling and manipulation.

Various transformations are included in the ``features`` module of the package. These
transformations are used throughout the rest of the package and may prove useful for
other applications.

The ``prediction`` module includes transformations useful for estimating and making 
predictions with forecast models. A simple online forecasting model is included in the
package through the `RRR` prediction. This class enables recursive estimation in a 
regularised linear model.

Hierarchical forecast reconciliation can be done using the ``hierarchies`` module and
the ``RidgeReconciliation`` transformation which uses the ``RRR`` prediction to estimate
a linear model in the bottom level forecast errors.

The package further includes some tools for working with forecast data stored in 
`pandas` dataframes. These tools are available through the ``forecast_tools`` module
and use a custom accessor to define some convenience functions for managing and 
structuring forecast data.
"""

__version__ = "0.1.1"

from .core import DEFAULT_SOURCE, STATE, Transformation
from .features import (
    Combine,
    DesignMatrix,
    ForgettingMean,
    ForgettingVariance,
    FourierSeries,
    Lag,
    LowPass,
    One,
    SlidingMean,
    SlidingSum,
)
from .forecast_tools import (
    Concat,
    ForecastCollection,
    ForecastMatrix,
    ForecastModel,
    ToExog,
)
from .prediction import ARX, RRR, BackShift

__all__ = [
    "__version__",
    "Transformation",
    "STATE",
    "DEFAULT_SOURCE",
    "DesignMatrix",
    "Combine",
    "One",
    "BackShift",
    "LowPass",
    "FourierSeries",
    "SlidingSum",
    "SlidingMean",
    "ForgettingMean",
    "ForgettingVariance",
    "Lag",
    "RRR",
    "ARX",
    # optional exports (provided lazily via __getattr__)
    "ForecastMatrix",
    "ForecastModel",
    "ForecastCollection",
    "ToExog",
    "Concat",
]

_OPTIONAL_FORECAST_TOOLS = {
    "ForecastMatrix",
    "ForecastModel",
    "ForecastCollection",
    "ToExog",
    "Concat",
}

def __getattr__(name: str):
    if name in _OPTIONAL_FORECAST_TOOLS:
        from .forecast_tools import (
            Concat,
            ForecastCollection,
            ForecastMatrix,
            ForecastModel,
            ToExog,
        )

        return {
            "ForecastMatrix": ForecastMatrix,
            "ForecastModel": ForecastModel,
            "ForecastCollection": ForecastCollection,
            "ToExog": ToExog,
            "Concat": Concat,
        }[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")
