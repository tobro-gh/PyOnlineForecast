"""A package for online forecasting and reconciliation.

The package provides tools for building data pipelines for online forecasting models.
The package is built around the concept of ``Source`` objects that act as placeholders
for data in the pipeline and ``Transformation`` objects that define data transformations
on those sources. All transformations are also sources and can be used freely to compose
new transformations. This defines a computational graphs from input root ``Source``
nodes to leaf ``Transformation`` nodes. Once data is available, any transformations in
the graph may be resolved to produce output values, as long as all the required sources
are provided. The package mostly uses `numpy` for data handling and manipulation.

Various transformations are included in the ``features`` module of the package. These
transformations are used without the rest of the package and may prove useful for other
applications.

``Prediction`` transformations are included in the ``prediction`` module. They align
lagged input features with new observations to train model parameters which are then
used for making forecasts. A simple online forecasting model is included in the package
through the `RRR` prediction. This class enables recursive estimation in a regularised
linear model.

Hierarchical forecast reconciliation can be done using the ``hierarchies`` module and
the ``RidgeReconciliation`` transformation which uses the ``RRR`` prediction to estimate
a linear model in the bottom level forecast errors.

The package further includes some tools for working with data stored in formats typical
for forecasting. We say that such data is stored as a forecast matrix. The tools use
pandas and a custom accessor for working with forecast matrix data and defines some
convenience functions and classes for creating data pipeliens using this data.
"""

__version__ = "0.1.0"

from .core import DEFAULT_SOURCE, MEMORY, Source
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
    ForecastEnsemble,
    ForecastMatrix,
    ForecastModel,
    ToExog,
)
from .prediction import ARX, RRR, WLS, BackShift

__all__ = [
    "__version__",
    "Source",
    "MEMORY",
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
    "WLS",
    "RRR",
    "ARX",
    "ForecastMatrix",
    "ForecastModel",
    "ForecastEnsemble",
    "ToExog",
    "Concat",
]
