"""Utilities for working with forecast data and creating forecast models."""

from __future__ import annotations

import functools
import re
from typing import Literal

import numpy as np

try:
    import pandas as pd
except ImportError:
    raise ImportError("Error importing pandas. Please make sure it is installed.")

from .core import DEFAULT_SOURCE, MEMORY, Format, Transformation, register_format_like
from .features import (
    DesignMatrix,
    ForgettingMean,
    ForgettingVariance,
    Lag,
    LowPass,
    SlidingMean,
    SlidingSum,
    ToArray,
)
from .hierarchies import RidgeReconciliation
from .prediction import ARX, RRR, WLS

### Numpy / Pandas conversion utilities


def _to_pandas(val, index=None, columns=None):

    if isinstance(val, pd.DataFrame):
        val = val.to_numpy()

    if val.ndim == 2:
        return pd.DataFrame(val, index=index, columns=columns)
    else:
        # Try to reshape to 2D
        reshaped = val.reshape((val.shape[0], -1))
        if columns is not None and len(columns) != reshaped.shape[1]:
            columns = [
                f"{var}_{i}"
                for var in columns
                for i in range(reshaped.shape[1] // len(columns))
            ]
        return pd.DataFrame(reshaped, index=index, columns=columns)


def _to_pandas_like(val, ref_val):
    return _to_pandas(val, index=ref_val.index, columns=ref_val.columns)


def assert_fc_columns(columns: pd.MultiIndex):
    """Assert that columns are in forecast matrix format."""
    assert isinstance(columns, pd.MultiIndex), "Columns must be a MultiIndex."
    assert (
        columns.names == ForecastMatrix.names
    ), f"Column names must be {ForecastMatrix.names}."
    assert all(
        isinstance(h, int) for h in columns.get_level_values("Horizon")
    ), "Horizon level must be integers."


def assert_fc_index(index: pd.Index):
    """Assert that index is in forecast matrix format."""
    assert isinstance(index, pd.Index), "Index must be a pandas Index."
    assert index.is_monotonic_increasing, "Index must be monotonic increasing."
    assert not index.has_duplicates, "Index must not have duplicates."

    # If DatetimeIndex, check frequency
    if isinstance(index, pd.DatetimeIndex):
        assert index.freq is not None, "DatetimeIndex must have frequency set."


@pd.api.extensions.register_dataframe_accessor("fc")
class ForecastMatrix:
    r"""Acessor with utilities for working with forecast data.

    The accessor provides methods for checking, subsetting and manipulating data obeying
    a \"forecast matrix\" structure. Any such data should be stored as a pandas
    DataFrame with a two level MultiIndex for columns, where the first level contains
    variable names and the second level contains integer forecast horizons. Dataframes
    can be converted to this format using the ``convert`` method.
    """

    names = ["Variable", "Horizon"]

    def __init__(self, data: pd.DataFrame):
        self._obj: pd.DataFrame = data

    def subset(self, *variables, horizons=None) -> pd.DataFrame:
        """Return a subset of the data matching specified variables and horizons."""
        selected_cols = subset_columns(
            self._obj.columns, *variables, horizons=horizons, return_index=True
        )
        subset_data = pd.DataFrame(
            self._obj.values[:, selected_cols],
            index=self._obj.index,
            columns=self._obj.columns[selected_cols],
        )
        return subset_data

    def assert_format(self):
        """Assert that the data is in forecast matrix format."""
        assert_fc_columns(self._obj.columns)
        assert_fc_index(self._obj.index)

    def convert(self, separator=None, fill_method=None):
        """Return a copy of the data converted to forecast matrix format."""
        data = self._obj.copy()

        if not isinstance(data.columns, pd.MultiIndex):

            if separator is None:
                new_columns = [(col, 0) for col in data.columns]

            else:
                # Try to parse columns using separator
                input_name_pattern = re.compile(rf"^(.*?){re.escape(separator)}(\d+)$")
                new_columns = []
                for col in data.columns:
                    if isinstance(col, str):
                        match = input_name_pattern.match(col)
                        if match:
                            name = match.group(1)
                            if not match.group(2).isdigit():
                                raise ValueError(
                                    f"Horizon must be an integer, got {match.group(2)}."
                                )
                            horizon = int(match.group(2))
                            new_columns.append((name, horizon))
                        else:
                            new_columns.append((col, 0))
                    else:
                        new_columns.append((col, 0))
        else:

            # Flatten levels if more than 2
            n_levels = len(data.columns.levels)
            if n_levels > 2:
                horizons = data.columns.get_level_values(-1)
                flat_level = data.columns.droplevel(-1).to_flat_index()
                new_columns = [(l, h) for l, h in zip(flat_level, horizons)]
            else:
                new_columns = list(data.columns)

            # Convert horizons to integers if possible, otherwise set to 0
            for i, col in enumerate(new_columns):
                if not isinstance(col[1], int):
                    if isinstance(col[1], str) and col[1].isdigit():
                        new_columns[i] = (col[0], int(col[1]))
                    else:
                        new_columns[i] = (col[0], 0)

        data.columns = pd.MultiIndex.from_tuples(new_columns, names=self.names)

        if isinstance(data.index, pd.DatetimeIndex):
            # Infer frequency
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq is not None:
                data = data.asfreq(inferred_freq, method=fill_method)

        return data

    @property
    def variables(self):
        """Return list of unique variable names in the data."""
        return self._obj.columns.get_level_values(0).unique().tolist()

    @property
    def horizons(self):
        """Return list of unique horizons in the data."""
        res = self._obj.columns.get_level_values(1).unique().tolist()
        return tuple(res)

    def lag(self, reverse=False) -> pd.DataFrame:
        """Return a copy of the data with columns shifted accoring to horizons."""
        result = self._obj.copy()
        for i, col in enumerate(result.columns):
            if col[1] not in [None, ""]:
                k = -col[1] if reverse else col[1]
                result.iloc[:, [i]] = result.iloc[:, [i]].shift(
                    k, fill_value=float("nan")
                )
        return result


# %%
@functools.wraps(pd.read_csv)
def read_forecast_csv(*args, horizon_pattern=None, **kwargs) -> pd.DataFrame:
    """Read a CSV file and convert to forecast matrix format if necessary."""
    if horizon_pattern is None:
        kwargs = {"header": [0, 1]} | kwargs
    df = pd.read_csv(*args, **kwargs)

    try:
        df.fc.assert_format()
    except AssertionError:
        df = df.fc.convert(separator=horizon_pattern)
    # TODO: add check that columns are correctly loaded
    df.index.name = None
    return df


def subset_columns(columns, *variables, horizons=None, return_index=False):
    """Return index or columns matching variables and horizons.

    Parameters
    ----------
    columns : pd.Index
        Columns to subset, typically the columns of a forecast matrix DataFrame.
    *variables : str
        Variable names to subset. If empty, all variables are included.
    horizons : list of int, optional
        Forecast horizons to subset. If ``None``, all horizons are included.
    return_index : bool, optional
        If ``True``, return a boolean indexer instead of the subset of columns.

    Returns
    -------
    pd.Index or np.ndarray
        Subset of columns matching the specified variables and horizons, or a boolean
        indexer if ``return_index`` is ``True``.
    """
    if len(variables) == 0 and horizons is None:
        index = [True] * len(columns)
    elif len(variables) == 0:
        if isinstance(columns, pd.MultiIndex):
            index = columns.get_level_values(1).isin(horizons)
        else:
            raise ValueError(
                "Cannot subset by horizon when columns are not MultiIndex."
            )
    elif horizons is None:
        index = [col[0] in variables for col in columns]
    else:
        index = [(col[0] in variables) and (col[1] in horizons) for col in columns]

    if return_index:
        return index

    return columns[index]


def fc_columns_from_tuples(tuples: list | tuple) -> pd.MultiIndex:
    """Return forecast matrix columns from list of (variable, horizon) tuples."""
    return pd.MultiIndex.from_tuples(tuples, names=ForecastMatrix.names)


def fc_columns_from_product(
    variables: list | tuple, horizons: list | tuple, group_by_horizon: bool = False
) -> pd.MultiIndex:
    """Return forecast matrix columns from product of variables and horizons.

    Parameters
    ----------
    variables : list of str
        Variable names to include in columns.
    horizons : list of int
        Forecast horizons to include in columns.
    group_by_horizon : bool, optional
        If ``True``, group columns by horizon instead of variable, i.e. order columns as
        (var1, h1), (var2, h1), ..., (var1, h2), (var2, h2), ... If ``False``, order
        columns as (var1, h1), (var1, h2), ..., (var2, h1), (var2, h2), ...

    Returns
    -------
    pd.MultiIndex
        Forecast matrix columns with specified variables and horizons.
    """
    if group_by_horizon:
        # Sort according to (var1, h1), (var2, h1), ..., (var1, h2), (var2, h2), ...
        tuples = [(var, h) for h in horizons for var in variables]
        return fc_columns_from_tuples(tuples)
    else:
        return pd.MultiIndex.from_product(
            [variables, horizons], names=ForecastMatrix.names
        )


def _make_fc_columns(names, horizons, outer_prod=False):
    if outer_prod:
        names = [(n1, n2) for n1 in names for n2 in names]
        horizons = [max(h1, h2) for h1 in horizons for h2 in horizons]
    return pd.MultiIndex.from_tuples(
        [(name, horizon) for name, horizon in zip(names, horizons)],
        names=ForecastMatrix.names,
    )


def _get_forecast_columns(names, horizon, outer_prod=False):
    horizons = [horizon] * len(names)
    return _make_fc_columns(names, horizons, outer_prod=outer_prod)


class Subset(Transformation):
    """Subset data by variable names and horizons.

    Parameters
    ----------
    data : Source
        Source providing data to subset. Output should be a pandas DataFrame conforming
        to the forecast matrix format (see ``ForecastMatrix``).
    *variables : object
        Variable names to include in the subset.
    horizons : list of int, optional
        Forecast horizons to include in the subset. If ``None``, all horizons are
        included.
    """

    def __init__(self, data, *variables, horizons=None):
        self.horizons = horizons
        self.variables = list(variables)
        super().__init__(data, indices=MEMORY)

    def evaluate(self, data, indices=None):
        """Return subset of data matching specified variables and horizons."""
        if isinstance(data, pd.DataFrame):
            columns = data.columns
        else:
            return data, None

        if indices is None:
            indices = subset_columns(
                columns, *self.variables, horizons=self.horizons, return_index=True
            )

        return data.iloc[:, indices], indices

    def __repr__(self):
        """Return string representation of Subset transformation."""
        return super().__repr__() + f"({self.variables}, horizons={self.horizons})"


class _GetHorizons(Transformation):

    def __init__(self, data, *horizons):
        self.horizons = horizons
        super().__init__(data, indices=MEMORY)

    def evaluate(self, data, indices=None):
        if isinstance(data, pd.DataFrame):
            if indices is None:
                indices = subset_columns(
                    data.columns, horizons=self.horizons, return_index=True
                )
            return data.iloc[:, indices], indices
        elif isinstance(data, dict):
            result = [data[h] for h in data if h in self.horizons]
            # Concatenate result
            if isinstance(result[0], pd.DataFrame):
                result = pd.concat(result, axis=1)
            else:
                # Ensure all are 2D, keeping first dimension as time
                result = [r.reshape(r.shape[0], -1) for r in result]
                result = np.hstack(result)
            return result, indices
        else:
            return data, indices


class Reindexer(Transformation):
    """Reindex data to specified frequency, filling missing values with NaN.

    Parameters
    ----------
    freq : str
        Pandas frequency string to reindex data, e.g. "h" for hourly.
    data : Source, optional
        Source providing data to reindex. Output should be a pandas DataFrame with a
        DatetimeIndex. If ``None`` (default) use ``DEFAULT_SOURCE``.
    """

    def __init__(self, freq, data=DEFAULT_SOURCE):
        super().__init__(data)
        self.freq = freq

    def evaluate(self, data):
        """Return reindexed data."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Reindexer can only be applied to pandas DataFrame.")

        if data.index.freq != self.freq:
            expected_index = pd.date_range(
                start=data.index.min(), end=data.index.max(), freq=self.freq
            )
            data = data.reindex(expected_index)

        return data


class DataCleaner(Transformation):
    """Clean data by removing outliers and filling missing values.

    Parameters
    ----------
    forgetting : float
        Forgetting factor for estimating mean and variance of the data, see
        ``ForgettingMean`` and ``ForgettingVariance``.
    data : Source, optional
        Source providing data to clean. Output should be a pandas DataFrame. If ``None``
        (default) use ``DEFAULT_SOURCE``.
    z_thresh : float, optional
        Z-score threshold for identifying outliers. Default is 3.
    forward_fill : bool, optional
        Whether to forward fill missing values after removing outliers. Default is
        ``True``.
    track_memory : bool, optional
        Whether to track memory of mean and variance estimates for outlier detection.
        See ``ForgettingMean`` and ``ForgettingVariance``. Default is ``True``.
    freq : str, optional
        If specified, reindex data to this frequency before cleaning, using
        ``Reindexer``.
    """

    def __init__(
        self,
        forgetting,
        data=DEFAULT_SOURCE,
        z_thresh=3,
        forward_fill=True,
        track_memory=True,
        freq: str = None,
    ):

        if freq is not None:
            data = Reindexer(freq, data)

        mean = ForgettingMean(forgetting, track_memory=track_memory, data=data)
        variance = ForgettingVariance(
            forgetting,
            track_memory=track_memory,
            center=mean,
            covariance=False,
            data=data,
        )
        super().__init__(data, variance=variance, mean=mean, last_state=MEMORY)
        self.z_thresh = z_thresh
        self.forward_fill = forward_fill
        self.freq = freq

    def evaluate(self, data, variance, mean, last_state=None):
        """Return cleaned data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to clean.
        variance : np.ndarray
            Variance estimates for the data.
        mean : np.ndarray
            Mean estimates for the data.
        last_state : pd.Series, optional
            Last row of the cleaned data from the previous steps.

        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("DataCleaner can only be applied to pandas DataFrame.")

        std = np.sqrt(variance)
        z_scores = np.abs((data - mean) / std)

        # Identify outliers
        outliers = z_scores > self.z_thresh

        # Replace outliers with NaN
        data_cleaned = data.mask(outliers)

        if last_state is not None:
            # Fill first row with last state
            filled_first_row = data_cleaned.iloc[[0]].fillna(value=last_state)
            data_cleaned.iloc[0] = filled_first_row.iloc[0]

        if self.forward_fill:
            data_cleaned = data_cleaned.ffill()

        # Store last state
        last_state = data_cleaned.iloc[-1]

        return data_cleaned, last_state


class RenameColumns(Transformation):
    """Rename columns of a DataFrame."""

    def __init__(self, data, new_columns):
        super().__init__(data=data)
        self.new_columns = new_columns

    def evaluate(self, data):
        """Return data with renamed columns."""
        data.columns = self.new_columns
        return data


class Align(Transformation):
    """Align dataframes to expected index and concatenate.

    Parameters
    ----------
    expected_index : pd.Index
        Index to align dataframes to.
    *data : list of pd.DataFrame
        DataFrames to align and concatenate.
    method : str, optional
        Method to use for filling missing values when aligning, passed to pandas
        reindex method. Default is ``None``, which means missing values will be filled
        with NaN.
    """

    def __init__(self, expected_index, *data, method: str = None):
        super().__init__(expected_index, *data)
        self.method = method

    def evaluate(self, expected_index, *data):
        """Return aligned dataframe."""
        aligned_data = []
        for df in data:
            if not isinstance(df, pd.DataFrame):
                raise ValueError(
                    "AlignIndex can only be applied to pandas DataFrame or Series."
                )
            aligned_df = df.reindex(expected_index, method=self.method)
            aligned_data.append(aligned_df)
        if len(aligned_data) == 1:
            result = aligned_data[0]
        result = aligned_data

        # Concatenate if multiple dataframes
        result = pd.concat(result, axis=1)

        return result


class Concat(Transformation):
    """Concatenate dataframes along specified axis."""

    def __init__(self, *data, axis=0):
        super().__init__(*data)
        self.axis = axis

    def evaluate(self, *data):
        """Return concatenated dataframe."""
        return pd.concat(data, axis=self.axis)


class Scaler(Transformation):
    """Multiply specified variables by given floats.

    Parameters
    ----------
    data : Source
        Source providing data to scale. Output should be a pandas DataFrame or Series.
    var_scales : dict
        Dictionary mapping variable names to floats. Variables in the data but not in
        the dictionary will not be scaled.
    """

    def __init__(self, data, var_scales: dict[str, float]):
        super().__init__(data, state=MEMORY)
        self.var_scales = var_scales

    def evaluate(self, data, state=None):
        """Return scaled data and state containing indexer.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            Data to scale.
        state : tuple, optional
            If provided, should be a tuple of (indexer, scales) where indexer is a
            boolean or integer indexer for the columns to scale, and scales is an array
            of the corresponding scale factors.
        """
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise ValueError(
                "DataScaler can only be applied to pandas DataFrame or Series."
            )

        if state is None:
            data_cols = data.columns if isinstance(data, pd.DataFrame) else data.index
            indexer = subset_columns(
                data_cols, *list(self.var_scales.keys()), return_index=True
            )

            # Translate bool indexer to location index
            indexer = np.where(indexer)[0]

            cols = data_cols[indexer]
            if isinstance(cols, pd.MultiIndex):
                scales = np.array([self.var_scales[col[0]] for col in cols])
            else:
                scales = np.array([self.var_scales[col] for col in cols])
            state = (indexer, scales)

        indexer, scales = state

        scaled_data = data.copy()

        if isinstance(data, pd.Series):
            scaled_data.iloc[indexer] = data.iloc[indexer] * scales

        else:

            scaled_data.iloc[:, indexer] = data.iloc[:, indexer] * scales

        return scaled_data, state


class Aggregator(Transformation):
    """Aggregate data by resampling to specified frequency and applying sum or mean.

    Parameters
    ----------
    freq : str
        Pandas frequency string to resample data.
    data : Source, optional
        Source providing data to aggregate. Output should be a pandas DataFrame.
    agg_type : str, default "mean"
        Type of aggregation to apply. Must be either "sum" or "mean".
    """

    def __init__(
        self, freq, data=DEFAULT_SOURCE, agg_type: Literal["sum", "mean"] = "mean"
    ):
        self.freq = freq
        if agg_type not in ["sum", "mean"]:
            raise ValueError("agg_type must be either 'sum' or 'mean'.")
        self.agg_type = agg_type
        super().__init__(data)

    def evaluate(self, data):
        """Return aggregated data."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Aggregator can only be applied to pandas dataframe.")

        if self.agg_type == "sum":
            aggregated_data = data.resample(
                self.freq, closed="right", label="right"
            ).sum()
        else:
            aggregated_data = data.resample(
                self.freq, closed="right", label="right"
            ).mean()

        return aggregated_data


class FillMissing(Transformation):
    """Fill missing values in data with specified value."""

    def __init__(self, data, fill_value=0):
        super().__init__(data=data)
        self.fill_value = fill_value

    def evaluate(self, data):
        """Return data with missing values filled."""
        if isinstance(data, np.ndarray):
            return np.nan_to_num(data, nan=self.fill_value)
        else:
            return data.fillna(self.fill_value)


class ToPandas(Transformation):
    """Convert data to pandas DataFrame with specified columns and index.

    Parameters
    ----------
    data : Source
        Source providing data to convert. Output should be a numpy array or similar.
    columns : list of str or pd.Index
        Column names for the resulting DataFrame.
    index : pd.Index, optional
        Index for the resulting DataFrame. If ``None``, a default integer index will be
        used.
    """

    def __init__(self, data, columns, index):
        super().__init__(data=data, index=index)
        self.new_columns = columns

    def evaluate(self, data, index=None):
        """Return data converted to pandas DataFrame."""
        return _to_pandas(data, index=index, columns=self.new_columns)


class Disruption(Transformation):
    """Model hourly disruptions at specific times by generating indicator features.

    Parameters
    ----------
    index : Source
        Source providing a datetime index to evaluate disruptions on.
    hour : int
        Hour of the day when the disruption occurs (0-23).
    dayofweek : int, optional
        Day of the week when the disruption occurs (0=Monday, 6=Sunday). If ``None``,
        disruption occurs on all days.
    duration : int, optional
        Duration of the disruption in hours. If ``None``, disruption is assumed to last
        for one hour.
    horizons : int or list of int, optional
        Forecast horizons to generate disruption features for. Outputs are shifted
        to match the specified horizons, so that the feature activates on the time for
        which the forecast is made.
    as_dict : bool, optional
        If ``True``, return a dictionary mapping horizons to disruption features instead
        of concatenating into a single DataFrame. Default is ``True``.
    """

    def __init__(
        self,
        index,
        hour,
        dayofweek=None,
        duration=None,
        horizons: int | list = 0,
        as_dict=True,
    ):
        super().__init__(index=index)
        self.hour = hour
        self.dayofweek = dayofweek
        self.duration = duration
        self.end_hour = (hour + duration) % 24 if duration is not None else None
        self.horizons = horizons if isinstance(horizons, list) else [horizons]
        self.columns = _make_fc_columns(
            ["Disruption"] * len(self.horizons), self.horizons
        )
        self.as_dict = as_dict

    def evaluate(self, index):
        """Return disruption features for specified index."""
        result = {}
        for h in self.horizons:
            result[h] = self._evaluate_horizon(index, h)

        if self.as_dict:
            return result

        result = list(result.values())

        # Concatenate into forecast matrix format
        result = np.array(result).T
        result = _to_pandas(result, index=index, columns=self.columns)

        return result

    def _evaluate_horizon(self, index, horizon):
        """Return disruption feature for specified index and horizon."""
        # TODO: consider pre-computing for efficiency
        pred_time = index + pd.Timedelta(hours=horizon)
        if self.duration is not None:
            if self.hour < self.end_hour:
                cond = (pred_time.hour >= self.hour) & (pred_time.hour < self.end_hour)
                if self.dayofweek is not None:
                    cond = cond & (pred_time.dayofweek == self.dayofweek)
            else:
                cond1 = pred_time.hour >= self.hour
                cond2 = pred_time.hour < self.end_hour
                if self.dayofweek is None:
                    cond = cond1 | cond2
                else:
                    d1 = pred_time.dayofweek == self.dayofweek
                    d2 = pred_time.dayofweek == (self.dayofweek + 1) % 7
                    cond = (cond1 & d1) | (cond2 & d2)
        else:
            cond = pred_time.hour == self.hour
            if self.dayofweek is not None:
                cond = cond & (pred_time.dayofweek == self.dayofweek)

        if not isinstance(cond, (pd.Index, np.ndarray)):
            cond = pd.Index([cond])
        data = cond.astype(float)

        return data


class TimeOfDay(Transformation):
    """Time of day feature, expressed as the fraction of the day passed.

    Parameters
    ----------
    index : Source
        Source providing a datetime index to evaluate time of day on.
    """

    def __init__(self, t):
        super().__init__(t=t)

    def evaluate(self, t):
        """Return the fraction of day passed for each timestamp in the index."""
        if isinstance(t, pd.DatetimeIndex):
            t = pd.Series(t)
        elif isinstance(t, pd.Timestamp):
            t = pd.Series([t])
        else:
            raise ValueError("Input t must be a pd.DatetimeIndex or pd.Timestamp.")
        delta = t - t.dt.floor("D")
        seconds = delta.dt.total_seconds()
        time_of_day_float = seconds / 86400
        time_of_day_float = time_of_day_float.to_numpy()

        return time_of_day_float


class TimeOfYear(Transformation):
    """Time of year feature, expressed as the fraction of the year passed.

    Parameters
    ----------
    index : Source
        Source providing a datetime index to evaluate time of year on.
    """

    def __init__(self, t):
        super().__init__(t=t)

    def evaluate(self, t):
        """Return the fraction of year passed for each timestamp in the index."""
        if isinstance(t, pd.DatetimeIndex):
            t = pd.Series(t)
        elif isinstance(t, pd.Timestamp):
            t = pd.Series([t])
        else:
            raise ValueError("Input t must be a pd.DatetimeIndex or pd.Timestamp.")

        start = pd.to_datetime(t.dt.year, format="%Y")
        end = start + pd.DateOffset(years=1)

        delta = (t - start).dt.total_seconds()
        total = (end - start).dt.total_seconds()
        result = delta / total
        result = result.to_numpy()

        return result


class TimeOfWeek(TimeOfDay):
    """Time of week feature, expressed as the fraction of the week passed.

    Parameters
    ----------
    index : Source
        Source providing a datetime index to evaluate time of week on.
    """

    def __init__(self, t):
        super().__init__(t=t)

    def evaluate(self, t):
        """Return the fraction of week passed for each timestamp in the index."""
        tod = super().evaluate(t)
        if isinstance(t, pd.DatetimeIndex):
            t = pd.Series(t)
        elif isinstance(t, pd.Timestamp):
            t = pd.Series([t])
        else:
            raise ValueError("Input t must be a pd.DatetimeIndex or pd.Timestamp.")

        tow = (t.dt.dayofweek + tod) / 7
        tow = tow.to_numpy()
        return tow


class Index(Transformation):
    """Return the index of the data as a feature."""

    def __init__(self, data=DEFAULT_SOURCE):
        super().__init__(data=data)

    def evaluate(self, data):
        """Return the index of the data."""
        return data.index


def _get_horizons(columns: pd.MultiIndex | tuple):
    if isinstance(columns, pd.MultiIndex):
        return columns.get_level_values(1).unique().tolist()
    if isinstance(columns, tuple) and len(columns) == 2 and isinstance(columns[1], int):
        return columns[1]
    else:
        raise ValueError("Columns do not have MultiIndex format.")


class Horizons(Transformation):
    """Extract forecast horizons from columns of a DataFrame or keys of a dictionary.

    Parameters
    ----------
    data : Source
        Source providing data to extract horizons from. Output should be a pandas
        DataFrame in the forecast matrix format (see ``ForecastMatrix``) or a dictionary
        with integer keys representing horizons.
    """

    def __init__(self, data=DEFAULT_SOURCE):
        super().__init__(data=data)

    def evaluate(self, data):
        """Return the forecast horizons."""
        if isinstance(data, pd.DataFrame):
            vars = data.columns
        elif isinstance(data, pd.Series):
            vars = data.name
        return _get_horizons(vars)


class ToExog(Transformation):
    """Convert data to exogenous format for use with ARX predictor.

    Parameters
    ----------
    horizon : int
        Number of forecast horizons to include in the exogenous variables. All horizons
        from 1 to this value will be extracted from the data.
    data : Source, optional
        Source providing data to convert. Output should be a pandas DataFrame in the
        forecast matrix format (see ``ForecastMatrix``) or a 3D numpy array of shape
        (t, n_horizon, n_var).  If ``None`` (default) use ``DEFAULT_SOURCE``.
    """

    def __init__(self, horizon, data=DEFAULT_SOURCE):
        self.horizons = np.arange(1, horizon + 1)
        super().__init__(data=data, indices=MEMORY)

    def evaluate(self, data, indices=None):
        """Return data converted to exogenous format for ARX predictor.

        If the input data is a DataFrame in forecast matrix format, it will be reshaped
        to a 3D array. If the input data is already a 3D array, it will be checked for
        correct shape and returned as is.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            Data to convert. Should be either a pandas DataFrame in the forecast matrix
            format (see ``ForecastMatrix``) or a 3D numpy array of shape
            (t, n_horizon, n_var).
        indices : list of int, optional
            If provided, should be a list of column indices to extract from the
            DataFrame. Will be inferred from the DataFrame columns if not provided.
            Ignored if data is a numpy array.

        Returns
        -------
        np.ndarray
            Data converted to exogenous format for ARX predictor, as a 3D array of time
            steps, horizons and variables. The result is a 3D array of shape
            (t, n_horizon, n_var), where t is the number of time steps, n_horizon is the
            number of forecast horizons, and n_var is the number of variables.
        """
        if isinstance(data, pd.DataFrame):
            # Convert to 3D array

            # Fetch indices for each horizon
            if indices is None:
                indices = [
                    subset_columns(data.columns, horizons=[h], return_index=True)
                    for h in self.horizons
                ]

            # Stack 3D array by horizon
            result = np.full(
                (data.shape[0], len(indices), data.shape[1] // len(self.horizons)),
                np.nan,
            )
            for i, idx in enumerate(indices):
                result[:, i, :] = data.iloc[:, idx]

        elif isinstance(data, np.ndarray):
            if data.ndim != 3:
                raise ValueError(
                    f"Data must be a 3D array of shape (t, n_horizon, n_var), but has shape {data.shape}."
                )

            # Check that data has correct shape
            if data.shape[1] != len(self.horizons):
                raise ValueError(
                    f"Data has {data.shape[1]} horizons, but expected {len(self.horizons)} horizons based on initialization."
                )

            result = data

        else:
            raise ValueError(
                f"Data must be either a pandas DataFrame or a numpy array, but is of type {type(data)}."
            )

        return result, indices


# Register methods for format_like
@register_format_like(pd.DataFrame, np.ndarray)
def _(source: pd.DataFrame, target: np.ndarray):
    return source.to_numpy()


@register_format_like(np.ndarray, pd.DataFrame)
def _(source: np.ndarray, target: pd.DataFrame):
    return _to_pandas_like(source, target)


# Make new format
class ForecastFormat(Format):
    """Format for handling data stored in dataframes obeying the forecast matrix format."""

    pass


# Register some common transformations
@ForecastFormat.register_resolver(Lag)
def _(source: Lag):
    return ForecastFormat.get_formatter(source.apply_args[0])


@ForecastFormat.register_resolver(LowPass)
def _(source: LowPass):
    return ForecastFormat.get_formatter(source.apply_args[0])


@ForecastFormat.register_resolver(ToArray)
def _(source: ToArray):
    return ForecastFormat.get_formatter(source.apply_args[0])


@ForecastFormat.register_resolver(RRR)
def _(source: RRR):
    def formatter(value_source, state, memory=None):
        value = state[value_source]
        Y = state[source.Y]
        formatted_value, cols = format_forecast(
            value, Y, horizon=source.horizon, cols=memory
        )
        return formatted_value, cols

    return formatter


@ForecastFormat.register_resolver(WLS)
def _(source: WLS):
    def formatter(value_source, state, memory=None):
        value = state[value_source]
        Y = state[source.Y]
        formatted_value, cols = format_forecast(
            value, Y, horizon=source.X_train.amount, cols=memory
        )
        return formatted_value, cols

    return formatter


@ForecastFormat.register_resolver(ARX)
def _(source: ARX):
    # TODO: fix
    def formatter(value_source, state, memory=None):
        value = state[value_source]
        Y = state[source.Y]

        if memory is None:
            # Construct columns based on Y
            name = Y.fc.variables[0]
            memory = fc_columns_from_product(
                [name], [h + 1 for h in range(source.horizon)]
            )

        mean = _to_pandas(value["mean"], index=Y.index, columns=memory)
        var = _to_pandas(value["var"], index=Y.index, columns=memory)

        return {"mean": mean, "var": var}, memory
    
    return formatter


@ForecastFormat.register_resolver(RidgeReconciliation)
def _(source: RidgeReconciliation):
    def formatter(value_source, state, memory=None):
        value = state[value_source]
        Y = state[source.Y_hat]

        result = {}
        # Format value["mean"] like Y
        result["mean"] = _to_pandas(value["mean"], index=Y.index, columns=Y.columns)

        # Format value["cov"] like Y, but with outer product of columns if cov is 3D
        outer_prod = value["cov"].ndim == 3

        if outer_prod:
            if memory is None:
                horizons = Y.columns.get_level_values(1).to_list()
                memory = _make_fc_columns(Y.columns, horizons, outer_prod=True)
            result["cov"] = _to_pandas(value["cov"], index=Y.index, columns=memory)
        else:
            result["cov"] = _to_pandas(value["cov"], index=Y.index, columns=Y.columns)

        return result, memory

    return formatter


@ForecastFormat.register_resolver(SlidingSum)
def _(source: SlidingSum):
    return ForecastFormat.get_formatter(source.apply_args[0])


@ForecastFormat.register_resolver(SlidingMean)
def _(source: SlidingMean):
    return ForecastFormat.get_formatter(source.apply_args[0])


@ForecastFormat.register_resolver(ForgettingMean)
def _(source: ForgettingMean):
    return ForecastFormat.get_formatter(source.data)


@ForecastFormat.register_resolver(ForgettingVariance)
def _(source: ForgettingVariance):
    mean_formatter = ForecastFormat.get_formatter(source.apply_kwargs["data"])
    if source.covariance:

        def formatter(value_source, state, memory=None):
            formatted_mean = mean_formatter(source.apply_kwargs["data"], state, None)

            if memory is None:

                # Construct outer product format
                names = formatted_mean.columns.to_list()
                horizons = formatted_mean.columns.get_level_values(1).to_list()
                memory = _make_fc_columns(names, horizons, outer_prod=True)

            # Format using memory columns and index as mean formatter
            formatted_cov = _to_pandas(
                state[value_source], index=formatted_mean.index, columns=memory
            )

            return formatted_cov, memory

        return formatter
    else:
        return mean_formatter


def format_forecast(prediction: dict, Y: pd.DataFrame, horizon: int = None, cols=None):
    """Format forecast output from WLS and RRR to match target variable Y."""
    # Fetch format if not provided
    if cols is None:
        cols = {}

        if not isinstance(Y, pd.DataFrame):
            raise ValueError("Y must be a pandas DataFrame to format forecast output.")

        y_vars = Y.columns

        if isinstance(y_vars, pd.MultiIndex):
            # Drop horizon level if present
            if "Horizon" in y_vars.names:
                y_vars = y_vars.droplevel("Horizon")

        # Attach horizon
        cols["mean"] = _get_forecast_columns(y_vars, horizon)

        if "cov" in prediction:
            cov_outer_prod = prediction["cov"].ndim == 3
            cols["cov"] = _get_forecast_columns(
                y_vars, horizon, outer_prod=cov_outer_prod
            )

    # Use same index as Y
    index = Y.index

    result = {}

    result["mean"] = _to_pandas(prediction["mean"], index=index, columns=cols["mean"])
    if "cov" in prediction:
        result["cov"] = _to_pandas(prediction["cov"], index=index, columns=cols["cov"])

    if "score" in prediction:
        result["score"] = prediction["score"]

    return result, cols


class ForecastModel(Transformation):
    """Linear forecast model using ridge regression.

    Parameters
    ----------
    X : list of Source or Source
        Source or list of sources providing input features for the forecast model.
        Sources that output dataframes in the forecast matrix format
        (see ``ForecastMatrix``) will be automatically subset to the specified input
        horizons.
    Y : Source
        Source providing target variable for the forecast model. Output should be a
        pandas DataFrame in the forecast matrix format (see ``ForecastMatrix``).
    horizon : int
        Forecast horizon to predict.
    input_horizons : tuple of int or "auto", optional
        Forecast horizons to include as input features for the forecast model. If a
        tuple of int, should specify the horizons to include. If ``"auto"``, will
        use (0, horizon).
    resolve_format : bool, optional
        Whether to apply formatting transformations to the input features before
        subsetting. If ``True`` (default), will apply ``ForecastFormat`` to each input.
    **kwargs
        Additional keyword arguments to pass to the ridge regression model, see
        ``prediction.RRR``.
    """

    def __init__(
        self,
        X,
        Y,
        horizon: int,
        input_horizons: tuple | Literal["auto"] = None,
        resolve_format=True,
        **kwargs,
    ):

        self.horizon = horizon

        # Ensure X is a list
        if not isinstance(X, (tuple, list)):
            X = [X]

        # If resolve_format is True, apply ForecastFormat transform to each X_i
        if resolve_format:
            X = [ForecastFormat(X_i) for X_i in X]

        # If input_horizons is "auto", set to include observation (0) and matching horizon (h)
        if input_horizons == "auto":
            input_horizons = (0, horizon)

        # Subset X according to input horizons
        if input_horizons is not None:
            X = [_GetHorizons(X_i, *input_horizons) for X_i in X]

        # Form design matrix for RRR
        X = DesignMatrix(*X)

        # Make prediction
        self.prediction = RRR(X, Y, horizon, **kwargs)

        # Set format
        self.prediction.set_format(ForecastFormat)

        super().__init__(self.prediction)

    @property
    def predictor(self):
        """Return the current ``RRR`` predictor object if any."""
        return self.recursion_pars[self.prediction][0]

    def set_score_mode(self):
        """Set score mode for the prediction."""
        self.prediction.set_score_mode()

    def unset_score_mode(self):
        """Unset score mode for the prediction."""
        self.prediction.unset_score_mode()

    def evaluate(self, prediction):
        """Return the prediction."""
        return prediction

    def update(self, data, update_predictor=True, **params):
        """Return model outputs from applying model transformations.

        Parameters
        ----------
        data : dict
            Dictionary containing data to call transformations on.
        update_predictor : bool, optional
            Whether to update model parameters using the provided data. If ``True``
            (default), will update the predictor.
        **params
            Additional parameters to pass to transformations.
        """
        return self(data, update_predictor=update_predictor, track_state=True, **params)

    def fit(self, data, update_predictor=True, **params):
        """Fit the model by resetting state and calling update."""
        self.reset_state()
        return self.update(data, update_predictor=update_predictor, **params)


class ForecastEnsemble(Transformation):
    """Ensemble of forecast models for multiple horizons.

    The ensemble consists of a separate ``ForecastModel`` for each specified horizon,
    which are called in parallel and their outputs are combined.

    Parameters
    ----------
    X : list of Source or Source
        Source or list of sources providing input features for the forecast model.
        Sources that output dataframes in the forecast matrix format
        (see ``ForecastMatrix``) will be automatically subset to the specified input
        horizons.
    Y : Source
        Source providing target variable for the forecast model. Output should be a
        pandas DataFrame in the forecast matrix format (see ``ForecastMatrix``).
    horizons : tuple of int
        Forecast horizons to predict. A separate forecast model will be created for each
        horizon.
    input_horizons : dict of {int: tuple of int} or "auto", optional
        Forecast horizons to include as input features for each forecast model. If a
        dict, should map each horizon to a tuple of int specifying the horizons to
        include for that model. If ``"auto"``, will use (0, h) for each horizon h.
    **kwargs
        Additional keyword arguments to pass to each forecast model, see
        ``ForecastModel``.
    """

    def __init__(
        self,
        X,
        Y,
        horizons: tuple[int],
        *args,
        input_horizons: dict | Literal["auto"] = None,
        **kwargs,
    ):
        self.models = []
        self.horizons = horizons
        for h in horizons:
            input_horizons_h = (
                input_horizons[h]
                if isinstance(input_horizons, dict)
                else input_horizons
            )
            m = ForecastModel(X, Y, h, *args, input_horizons=input_horizons_h, **kwargs)
            self.models.append(m)

        super().__init__(*self.models)

    def evaluate(self, *predictions):
        """Return dict of combined predictions."""
        result = {}
        # Concatenate predictions
        for key in ["mean", "cov"]:
            if key in predictions[0]:
                result[key] = pd.concat([pred[key] for pred in predictions], axis=1)

        # Combine scores in a dict per horizon
        if "score" in predictions[0]:
            result["score"] = {
                h: pred["score"] for h, pred in zip(self.horizons, predictions)
            }

        return result

    def set_score_mode(self):
        """Set score mode for all models in the ensemble."""
        for m in self.models:
            m.set_score_mode()

    def unset_score_mode(self):
        """Unset score mode for all models in the ensemble."""
        for m in self.models:
            m.unset_score_mode()

    def update(self, data, update_predictor=True, **params):
        """Return model outputs from applying model transformations for all models."""
        return self(data, update_predictor=update_predictor, track_state=True, **params)

    def fit(self, data, update_predictor=True, **params):
        """Fit the ensemble by resetting state and calling update."""
        self.reset_state()
        return self.update(data, update_predictor=update_predictor, **params)
