from __future__ import annotations
from unittest import result
import pandas as pd
import numpy as np
#from .core import Transformation, DEFAULT_SOURCE, DEFAULT_INDEX, MEMORY
#from .features import ForgettingMean, ForgettingVariance, Combine
from typing import Literal
import re
import functools

from pyparsing import Combine
from .core import *
from .prediction import *
from .hierarchies import *

### Numpy / Pandas conversion utilities

def to_pandas(val, index = None, columns = None):
    
    if isinstance(val, pd.DataFrame):
        val = val.to_numpy()

    if val.ndim == 2:
        return pd.DataFrame(val, index = index, columns = columns)
    else:
        # Try to reshape to 2D
        reshaped = val.reshape((val.shape[0], -1))
        if columns is not None and len(columns) != reshaped.shape[1]:
            columns = [f"{var}_{i}" for var in columns for i in range(reshaped.shape[1] // len(columns))]
        return pd.DataFrame(reshaped, index = index, columns = columns)
            

def to_pandas_like(val, ref_val):
    return to_pandas(val, index = ref_val.index, columns = ref_val.columns)


@pd.api.extensions.register_dataframe_accessor("fc")
class ForecastMatrix:
    
    names = ["Variable", "Horizon"]

    def __init__(self, data: pd.DataFrame):
        self._obj: pd.DataFrame = data
     
    def append(self, data: pd.DataFrame | pd.Series, ignore_column_names = False, sort = False):
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        else:
            data = data.copy()

        # Append time data of same shape
        if ignore_column_names:
            data.columns = self._obj.columns

        if sort:
            if not data.index.is_monotonic_increasing:
                data = data.sort_index()

        # Check that frequency matches (if existing data has datetime index)
        if isinstance(self._obj.index, pd.DatetimeIndex):
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Cannot append data with non-datetime index to data with datetime index.")
            # Check that data is contiguous
            expected_index = pd.date_range(start=self._obj.index[-1] + self._obj.index.freq, periods=len(data), freq=self._obj.index.freq)
            if not data.index.equals(expected_index):
                raise ValueError("Appended data is not contiguous with existing data.")
            
        return pd.concat([self._obj, data], axis = 0)

    def join(self, data: pd.DataFrame, how = "left"):
        if not data.fc.check():
            data = data.copy()
            data = data.fc.convert()
        return self._obj.join(data, how = how)

    def drop_data(self, *variables):
        for v in variables:
            if v in self._obj.columns.get_level_values(0):
                self._obj.drop(columns=[v], level=0, inplace=True)
            else:
                raise ValueError(f"Variable '{v}' not found in data.")

    def _get_columns(self, *variables, horizons = None):
        return subset_columns(self._obj.columns, *variables, horizons = horizons, return_index = True)

    def subset(self, *variables, start_time=None, end_time=None, start_index=None, end_index=None, horizons=None, time_col = None) -> pd.DataFrame:
        if start_time is not None:
            start_time = pd.to_datetime(start_time)
        if end_time is not None:
            end_time = pd.to_datetime(end_time)
        row_mask = np.ones(len(self._obj), dtype=bool)

        if start_time is not None or end_time is not None:
            if isinstance(self._obj.index, pd.DatetimeIndex):
                comp = self._obj.index
            elif not time_col is None and time_col in self._obj.columns:
                comp = self._obj[time_col]
            else:
                raise ValueError("No time index found in data.")
        if start_time is not None:
            row_mask &= (comp >= start_time).flatten()
        if end_time is not None:
            row_mask &= (comp < end_time).flatten()
        if start_index is not None:
            row_mask[:start_index] = False
        if end_index is not None:
            row_mask[end_index:] = False

        selected_cols = self._get_columns(*variables, horizons = horizons)
        subset_data = pd.DataFrame(self._obj.values[:, selected_cols], index=self._obj.index, columns=self._obj.columns[selected_cols])

        # Only apply row mask if any corresponding function arguments are not none
        if any([arg is not None for arg in [start_time, end_time, start_index, end_index]]):
            subset_data = subset_data.loc[row_mask]

        return subset_data

    def get_data(self, *variables, horizons=None, include_shape=False):
        result = {}
        if include_shape:
            result["n_t"] = self.n_t
    
        selected_cols = self._get_columns(*variables, horizons = horizons)

        subset_data = self._obj.iloc[:,selected_cols]

        for v in variables:
            result[v] = subset_data.fc[[v]]

        return result

    def check(self, silent = True):
        # Check if self._obj conforms to forecast structure
        if not isinstance(self._obj.columns, pd.MultiIndex):
            if not silent:
                print("DataFrame does not have MultiIndex columns.")
            return False
        elif len(self._obj.columns.levels) != 2:
            if not silent:
                print("DataFrame does not have 2 levels in MultiIndex columns.")
            return False
        elif not self._obj.columns.names == ['Variable', 'Horizon']:
            if not silent:
                print("DataFrame does not have correct column names.")
            return False
        # Check if all horizons are integers
        elif not all(isinstance(h, int) for h in self.horizons):
            if not silent:
                print("Not all horizons are integers.")
            return False
        # Check if index is datetime
        elif isinstance(self._obj.index, pd.DatetimeIndex):
            # Check if freq is set
            if self._obj.index.freq is None:
                if not silent:
                    print("DatetimeIndex does not have frequency set.")
                return False
        return True

    def convert(self, separator = ".k", fill_method = None):
        data = self._obj.copy()

        if not isinstance(data.columns, pd.MultiIndex):
            
            input_name_pattern = re.compile(rf'^(.*?){re.escape(separator)}(\d+)$')
            new_columns = []
            for col in data.columns:
                if isinstance(col, str):
                    match = input_name_pattern.match(col)
                    if match:
                        name = match.group(1)
                        if not match.group(2).isdigit():
                            raise ValueError(f"Horizon must be an integer, got {match.group(2)}.")
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

            # Rename horizons if they are not integers
            for i, col in enumerate(new_columns):
                if not isinstance(col[1], int):
                    if isinstance(col[1], str):
                        if col[1].isdigit():
                            new_columns[i] = (col[0], int(col[1]))
                        else:
                            match = col[1].rsplit(separator, 1)
                            if len(match) == 2 and match[1].isdigit():
                                new_columns[i] = (col[0], int(match[1]))
                            else:
                                new_columns[i] = (col[0], 0)
                    else:
                        new_columns[i] = (col[0], 0)

        data.columns = pd.MultiIndex.from_tuples(new_columns, names=self.names)

        if isinstance(data.index, pd.DatetimeIndex):
            # Infer frequency
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq is not None:
                data = data.asfreq(inferred_freq, method=fill_method)
        
        return data

    def remove_old_data(self, n_keep: int = None):
        if n_keep is None:
            n_keep = max(self.horizons)
        # Keep only required data
        n_drop = self.n_t - n_keep
        if n_drop > 0:
            self._obj.drop(self._obj.index[:n_drop], inplace=True)

    @property
    def n_t(self):
        return len(self._obj)
    
    @property
    def variables(self):
        return self._obj.columns.get_level_values(0).unique().tolist()
    
    @property
    def horizons(self):
        res = self._obj.columns.get_level_values(1).unique().tolist()
        return tuple(res)

    def get_horizon_tail(self, horizon: str | int, n_t):
        col_filter = self._obj.columns.get_level_values('Horizon').isin([horizon, None])
        data = self._obj.loc[:, col_filter].copy()
        return data[-(n_t+horizon):]


    def lag(self, *args, reverse = False, **kwargs) -> pd.DataFrame:
        """
        Retrieves a subset of the data lagged according to column name forecast horizons.
        """
        subset = self.subset(*args, **kwargs)
        for i, col in enumerate(subset.columns):
            if col[1] not in [None, ""]:
                k = -col[1] if reverse else col[1]
                subset.iloc[:,  [i]] = subset.iloc[:, [i]].shift(k, fill_value=float("nan"))
        return subset

    def join_variable(self, var, data: pd.DataFrame, how = "right") -> pd.DataFrame:
        data = data.copy()
        new_cols = pd.MultiIndex.from_product([[var], data.columns], names = self._obj.columns.names)
        data.columns = new_cols
        return self.join(data, how = how)

    def join_horizon(self, horizon, data: pd.DataFrame, how = "right") -> pd.DataFrame:
        data = data.copy()
        new_cols = pd.MultiIndex.from_product([data.columns.get_level_values(0), [horizon]], names = self._obj.columns.names)
        data.columns = new_cols
        return self.join(data, how = how)

    def __getitem__(self, key):
        match = next((col for col in self._obj.columns if col[0] == key), None)
        if match and match[1] == None:
            result = self._obj[key][None]
            if isinstance(result, pd.Series):
                result.name = key
            return result
        else:
            return self._obj[key]

def new_fc(data = None, index = None, columns = None, dtype = None, copy = None, separator = '.k') -> pd.DataFrame:
    result = pd.DataFrame(data, index, columns, dtype = dtype, copy = copy)
    if not result.fc.check():
        result = result.fc.convert(separator = separator)
    return result

def forecast_matrix_from_product(names: list | tuple, horizons: tuple) -> pd.DataFrame:
    """
    Create a DataFrame with MultiIndex columns from the product of names and horizons.
    """
    cols = pd.MultiIndex.from_product([names, horizons], names = ForecastMatrix.names)
    return new_fc(columns = cols)

#%%
@functools.wraps(pd.read_csv)
def read_forecast_csv(*args, horizon_pattern = None, **kwargs) -> pd.DataFrame:
    if horizon_pattern is None:
       kwargs = {"header": [0, 1]} | kwargs
    df = pd.read_csv(*args, **kwargs)
    if not df.fc.check():
        df = df.fc.convert(separator = horizon_pattern)
    # TODO: add check that columns are correctly loaded
    df.index.name = None
    return df


def subset_columns(columns, *variables, horizons = None, return_index = False):

    if len(variables) == 0 and horizons is None:
        index = [True]*len(columns)
    elif len(variables) == 0:
        if isinstance(columns, pd.MultiIndex):
            index = columns.get_level_values(1).isin(horizons)
        else:
            raise ValueError("Cannot subset by horizon when columns are not MultiIndex.")
    elif horizons is None:
        index = [col[0] in variables for col in columns]
    else:
        index = [(col[0] in variables) and (col[1] in horizons) for col in columns]

    if return_index:
        return index
    
    return columns[index]

def fc_columns_from_tuples(tuples: list | tuple) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(tuples, names = ForecastMatrix.names)

def fc_columns_from_product(variables: list | tuple, horizons: list | tuple, group_by_horizon: bool = False) -> pd.MultiIndex:
    if group_by_horizon:
        # Sort according to (var1, h1), (var2, h1), ..., (var1, h2), (var2, h2), ...
        tuples = [(var, h) for h in horizons for var in variables]
        return fc_columns_from_tuples(tuples)
    else:
        return pd.MultiIndex.from_product([variables, horizons], names = ForecastMatrix.names)

def check_fc_format(index: pd.Index | list | tuple, as_list = False) -> bool:
    if as_list:
        # Check only list of tuples
        for col in index:
            if not (isinstance(col, tuple) and len(col) == 2 and isinstance(col[1], int)):
                return False

    if not isinstance(index, pd.MultiIndex):
        return False
    if index.names != ForecastMatrix.names:
        return False
    if not all(isinstance(h, int) for h in index.get_level_values('Horizon')):
        return False
    return True

class Subset(Transformation):

    def __init__(self, data, *variables, horizons = None):
        self.horizons = horizons
        self.variables = list(variables)
        super().__init__(data, indices = MEMORY)


    def evaluate(self, data, indices = None):
        # Use fc.subset_columns to get the subset of columns
        if isinstance(data, pd.DataFrame):
            columns = data.columns
        else:
            return data, None

        if indices is None:
            indices = subset_columns(columns, *self.variables, horizons = self.horizons, return_index = True)

        return data.iloc[:, indices], indices

    def __repr__(self):
        return super().__repr__() + f"({self.variables}, horizons={self.horizons})"

class GetHorizons(Transformation):

    def __init__(self, data, *horizons):
        self.horizons = horizons
        super().__init__(data, indices = MEMORY)

    def evaluate(self, data, indices = None):
        if isinstance(data, pd.DataFrame):
            if indices is None:
                indices = subset_columns(data.columns, horizons = self.horizons, return_index = True)
            return data.iloc[:, indices], indices
        elif isinstance(data, dict):
            result = [data[h] for h in data if h in self.horizons]
            # Concatenate result
            if isinstance(result[0], pd.DataFrame):
                result = pd.concat(result, axis = 1)
            else:
                # Ensure all are 2D, keeping first dimension as time
                result = [r.reshape(r.shape[0], -1) for r in result]
                result = np.hstack(result)
            return result, indices
        else:
            return data, indices
        
class Reindexer(Transformation):

    def __init__(self, freq, data = DEFAULT_SOURCE):
        super().__init__(data)
        self.freq = freq

    def evaluate(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Reindexer can only be applied to pandas DataFrame.")

        if data.index.freq != self.freq:
            expected_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=self.freq)
            data = data.reindex(expected_index)

        return data

class DataCleaner(Transformation):

    def __init__(self, forgetting, data = DEFAULT_SOURCE, z_thresh = 3, forward_fill = True, track_memory = True, freq: str = None):

        if freq is not None:
            data = Reindexer(freq, data)

        mean = ForgettingMean(forgetting, track_memory = track_memory, data = data)
        variance = ForgettingVariance(forgetting, track_memory = track_memory, center = mean, covariance = False, data = data)
        super().__init__(data, variance = variance, mean = mean, last_state = MEMORY)
        self.z_thresh = z_thresh
        self.forward_fill = forward_fill
        self.freq = freq

    def evaluate(self, data, variance, mean, last_state = None):
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

    def __init__(self, data, new_columns):
        super().__init__(data = data)
        self.new_columns = new_columns

    def evaluate(self, data):
        data.columns = self.new_columns
        return data

class Align(Transformation):
    # Align dataframes/series to expected index and concatenate

    def __init__(self, expected_index, *data, method: str = None):
        super().__init__(expected_index, *data)
        self.method = method
    
    def evaluate(self, expected_index, *data):
        aligned_data = []
        for df in data:
            if not isinstance(df, (pd.DataFrame, pd.Series)):
                raise ValueError("AlignIndex can only be applied to pandas DataFrame or Series.")
            aligned_df = df.reindex(expected_index, method=self.method)
            aligned_data.append(aligned_df)
        if len(aligned_data) == 1:
            result = aligned_data[0]
        result = aligned_data

        # Concatenate if multiple dataframes
        result = pd.concat(result, axis = 1)

        return result

class Concat(Transformation):
    def __init__(self, *data, axis = 0):
        super().__init__(*data)
        self.axis = axis

    def evaluate(self, *data):
        return pd.concat(data, axis = self.axis)

class Scaler(Transformation):

    def __init__(self, data = DEFAULT_SOURCE, var_scales: dict[str, float] = None):
        super().__init__(data, state = MEMORY)
        self.var_scales = var_scales

    def evaluate(self, data, state = None):
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise ValueError("DataScaler can only be applied to pandas DataFrame or Series.")

        if state is None:
            data_cols = data.columns if isinstance(data, pd.DataFrame) else data.index
            indexer = subset_columns(data_cols, *list(self.var_scales.keys()), return_index = True)

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

        return scaled_data

class Aggregator(Transformation):

    def __init__(self, freq, data = DEFAULT_SOURCE, agg_type: Literal["sum", "mean"] = "mean"):
        self.freq = freq
        if agg_type not in ["sum", "mean"]:
            raise ValueError("agg_type must be either 'sum' or 'mean'.")
        self.agg_type = agg_type
        super().__init__(data)

    def evaluate(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Aggregator can only be applied to pandas dataframe.")

        if self.agg_type == "sum":
            aggregated_data = data.resample(self.freq, closed = "right", label = "right").sum()
        else:
            aggregated_data = data.resample(self.freq, closed = "right", label = "right").mean()
                
        return aggregated_data

class FillMissing(Transformation):

    def __init__(self, data, fill_value=0):
        super().__init__(data=data)
        self.fill_value = fill_value

    def evaluate(self, data):
        if isinstance(data, np.ndarray):
            return np.nan_to_num(data, nan=self.fill_value)
        else:
            return data.fillna(self.fill_value)

class ToPandas(Transformation):

    def __init__(self, data, columns, index):
        super().__init__(data = data, index = index)
        self.new_columns = columns

    def evaluate(self, data, index = None):
        return to_pandas(data, index = index, columns = self.new_columns)


class Disruption(Transformation):
    # A class for modelling disruptions at a specific day and hour

    def __init__(self, index, hour, dayofweek = None, duration = None, horizons: int | list = 0, as_dict = True):
        super().__init__(index = index)
        self.hour = hour
        self.dayofweek = dayofweek
        self.duration = duration
        self.end_hour = (hour + duration) % 24 if duration is not None else None
        self.horizons = horizons if isinstance(horizons, list) else [horizons]
        self.columns = make_fc_columns(["Disruption"]*len(self.horizons), self.horizons)
        self.as_dict = as_dict

    def evaluate(self, index):
        result = {}
        for h in self.horizons:
            result[h] = self.evaluate_horizon(index, h)

        if self.as_dict:
            return result

        result = list(result.values())

        # Concatenate into forecast matrix format
        result = np.array(result).T
        result.columns = self.columns
        result.index = index

        return result

    def evaluate_horizon(self, index, horizon):
        # TODO: consider pre-computing for efficiency
        pred_time = index + pd.Timedelta(hours=horizon)
        if self.duration is not None:
            if self.hour < self.end_hour:
                cond = (pred_time.hour >= self.hour) & (pred_time.hour < self.end_hour)
                if self.dayofweek is not None:
                    cond = cond & (pred_time.dayofweek == self.dayofweek)                
            else:
                cond1 = (pred_time.hour >= self.hour)
                cond2 = (pred_time.hour < self.end_hour)
                if self.dayofweek is None:
                    cond = cond1 | cond2
                else:
                    d1 = (pred_time.dayofweek == self.dayofweek)
                    d2 = (pred_time.dayofweek == (self.dayofweek + 1) % 7)
                    cond = (cond1 & d1) | (cond2 & d2)
        else:
            cond = (pred_time.hour == self.hour)
            if self.dayofweek is not None:
                cond = cond & (pred_time.dayofweek == self.dayofweek)

        if not isinstance(cond, (pd.Index, np.ndarray)):
            cond = pd.Index([cond])
        data = cond.astype(float)

        return data

class TimeOfDay(Transformation):

    def __init__(self, t):
        super().__init__(t = t)

    def evaluate(self, t):
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

    def __init__(self, t):
        super().__init__(t=t)

    def evaluate(self, t):
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

    def __init__(self, t):
        super().__init__(t=t)

    def evaluate(self, t):
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

    def __init__(self, data = DEFAULT_SOURCE):
        super().__init__(data = data)
    
    def evaluate(self, data):
        return data.index

def get_horizons(columns: pd.MultiIndex | tuple):
    if isinstance(columns, pd.MultiIndex):
        return columns.get_level_values(1).unique().tolist()
    if isinstance(columns, tuple) and len(columns) == 2 and isinstance(columns[1], int):
        return columns[1]
    else:
        raise ValueError("Columns do not have MultiIndex format.")

class Horizons(Transformation):
    
    def __init__(self, data = DEFAULT_SOURCE):
        super().__init__(data = data)

    def evaluate(self, data):
        if isinstance(data, pd.DataFrame):
            vars = data.columns
        elif isinstance(data, pd.Series):
            vars =  data.name
        return get_horizons(vars)
    

class ToExog(Transformation):
    """
    Ensures correct format of exogenous variables for use with the ARX predictor. Checks that inputs are 3D arrays of correct shape, or converts dataframes with fc format.
    The result shape is (t, n_horizon, n_var), where t is the number of time steps, n_horizon is the number of forecast horizons, and n_var is the number of variables.
    """

    def __init__(self, horizon, data = DEFAULT_SOURCE):
        self.horizons = np.arange(1, horizon + 1)
        super().__init__(data = data, indices = MEMORY)

    def evaluate(self, data, indices = None):

        if isinstance(data, pd.DataFrame):
            # Convert to 3D array

            # Fetch indices for each horizon
            if indices is None:
                indices = [subset_columns(data.columns, horizons=[h], return_index = True) for h in self.horizons]

            # Stack 3D array by horizon
            result = np.full((data.shape[0], len(indices), data.shape[1] // len(self.horizons)), np.nan)
            for i, idx in enumerate(indices):
                result[:, i, :] = data.iloc[:, idx]


        elif isinstance(data, np.ndarray):
            if data.ndim != 3:
                raise ValueError(f"Data must be a 3D array of shape (t, n_horizon, n_var), but has shape {data.shape}.")
            
            # Check that data has correct shape
            if data.shape[1] != len(self.horizons):
                raise ValueError(f"Data has {data.shape[1]} horizons, but expected {len(self.horizons)} horizons based on initialization.")
            
            result = data

        else:
            raise ValueError(f"Data must be either a pandas DataFrame or a numpy array, but is of type {type(data)}.")

        return result, indices


# Register methods for format_like
@register_format_like(pd.DataFrame, np.ndarray)
def _(source: pd.DataFrame, target: np.ndarray):
    return source.to_numpy()

@register_format_like(np.ndarray, pd.DataFrame)
def _(source: np.ndarray, target: pd.DataFrame):
    return to_pandas_like(source, target)

# Make new format
class ForecastFormat(Format):
    pass

# Register some common transformations
@ForecastFormat.register_resolver(Lag)
def _(source: Lag):
    return ForecastFormat.get_formatter(source.apply_kwargs["data"])

@ForecastFormat.register_resolver(LowPass)
def _(source: LowPass):
    return ForecastFormat.get_formatter(source.apply_kwargs["data"])

@ForecastFormat.register_resolver(ToArray)
def _(source: ToArray):
    return ForecastFormat.get_formatter(source.apply_kwargs["data"])

@ForecastFormat.register_resolver(RRR)
def _(source: RRR):
    def formatter(transform, state, memory = None):
        value = state[transform]
        Y = state[source.Y]
        formatted_value, cols = format_forecast(value, Y, horizon = source.horizon, cols = memory)
        return formatted_value, cols
    return formatter

@ForecastFormat.register_resolver(WLS)
def _(source: WLS):
    def formatter(transform, state, memory = None):
        value = state[transform]
        Y = state[source.Y]
        formatted_value, cols = format_forecast(value, Y, horizon = source.X_train.amount, cols = memory)
        return formatted_value, cols
    return formatter

@ForecastFormat.register_resolver(ARX)
def _(source: ARX):
    # TODO: fix
    def formatter(transform, state, memory = None):
        value = state[transform]
        Y = state[source.Y]
        formatted_value, cols = format_forecast(value, Y, horizon = source.horizon, cols = memory)
        return formatted_value, cols
    return formatter

@ForecastFormat.register_resolver(RidgeReconciliation)
def _(source: RidgeReconciliation):
    def formatter(transform, state, memory = None):
        value = state[transform]
        Y = state[source.Y_hat]

        result = {}
        # Format value["mean"] like Y
        result["mean"] = to_pandas(value["mean"], index = Y.index, columns = Y.columns)

        # Format value["cov"] like Y, but with outer product of columns if cov is 3D
        outer_prod = value["cov"].ndim == 3

        if outer_prod:
            if memory is None:
                horizons = Y.columns.get_level_values(1).to_list()
                memory = make_fc_columns(Y.columns, horizons, outer_prod = True)
            result["cov"] = to_pandas(value["cov"], index = Y.index, columns = memory)
        else:
            result["cov"] = to_pandas(value["cov"], index = Y.index, columns = Y.columns)

        return result, memory

    return formatter

@ForecastFormat.register_resolver(SlidingSum)
def _(source: SlidingSum):
    return ForecastFormat.get_formatter(source.apply_kwargs["data"])

@ForecastFormat.register_resolver(SlidingMean)
def _(source: SlidingMean):
    return ForecastFormat.get_formatter(source.apply_kwargs["data"])

@ForecastFormat.register_resolver(ForgettingMean)
def _(source: ForgettingMean):
    return ForecastFormat.get_formatter(source.data)

@ForecastFormat.register_resolver(ForgettingVariance)
def _(source: ForgettingVariance):
    mean_formatter = ForecastFormat.get_formatter(source.apply_kwargs["data"])
    if source.covariance:
        def formatter(transform, state, memory = None):
            formatted_mean = mean_formatter(source.apply_kwargs["data"], state, None)

            if memory is None:

                # Construct outer product format
                names = formatted_mean.columns.to_list()
                horizons = formatted_mean.columns.get_level_values(1).to_list()
                memory = make_fc_columns(names, horizons, outer_prod = True)
                            
            # Format using memory columns and index as mean formatter
            formatted_cov = to_pandas(state[transform], index = formatted_mean.index, columns = memory)

            return formatted_cov, memory
        return formatter
    else:
        return mean_formatter
        

def make_fc_columns(names, horizons, outer_prod = False):
    if outer_prod:
        names = [(n1, n2) for n1 in names for n2 in names]
        horizons = [max(h1, h2) for h1 in horizons for h2 in horizons]
    return pd.MultiIndex.from_tuples([(name, horizon) for name, horizon in zip(names, horizons)], names = ForecastMatrix.names)

def get_forecast_columns(names, horizon, outer_prod = False):
    horizons = [horizon]*len(names)
    return make_fc_columns(names, horizons, outer_prod = outer_prod)

def format_forecast(prediction: dict, Y: pd.DataFrame, horizon: int = None, cols = None):
    """
    Convenience function for formatting some special prediction dicts pf WLS and RRR
    """

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
        cols["mean"] = get_forecast_columns(y_vars, horizon)

        if "cov" in prediction:
            cov_outer_prod = prediction["cov"].ndim == 3
            cols["cov"] = get_forecast_columns(y_vars, horizon, outer_prod = cov_outer_prod)

    # Use same index as Y
    index = Y.index

    result = {}

    result["mean"] = to_pandas(prediction["mean"], index = index, columns = cols["mean"])
    if "cov" in prediction:
        result["cov"] = to_pandas(prediction["cov"], index = index, columns = cols["cov"])

    if "score" in prediction:
        result["score"] = prediction["score"]

    return result, cols

class ForecastModel(Transformation):


    def __init__(self, X, Y, horizon: int, *args, input_horizons: tuple | Literal["auto"] = None, resolve_format = True, **kwargs):

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
            X = [GetHorizons(X_i, *input_horizons) for X_i in X]

        # Form design matrix for RRR
        X = DesignMatrix(*X)

        # Make prediction
        self.prediction = RRR(X, Y, horizon, *args, **kwargs)

        # Set format
        self.prediction.set_format(ForecastFormat)

        super().__init__(self.prediction)


    @property
    def predictor(self):
        return self.recursion_pars[self.prediction][0]

    def set_score_mode(self):
        self.prediction.set_score_mode()

    def unset_score_mode(self):
        self.prediction.unset_score_mode()

    def evaluate(self, prediction):
        return prediction

    def update(self, data, update_predictor = True, **params):
        return self(data, update_predictor = update_predictor, track_state = True, **params)

    def fit(self, data, update_predictor = True, **params):
        self.reset_state()
        return self.update(data, update_predictor = update_predictor, **params)

class ForecastEnsemble(Transformation):

    def __init__(self, X, Y, horizons: tuple[int], *args, input_horizons: dict | Literal["auto"] = None, **kwargs):
        self.models = []
        self.horizons = horizons
        for h in horizons:
            input_horizons_h = input_horizons[h] if isinstance(input_horizons, dict) else input_horizons
            m = ForecastModel(X, Y, h, *args, input_horizons = input_horizons_h, **kwargs)
            self.models.append(m)

        super().__init__(*self.models)

    def evaluate(self, *predictions):
        result = {}
        # Concatenate predictions
        for key in ["mean", "cov"]:
            if key in predictions[0]:
                result[key] = pd.concat([pred[key] for pred in predictions], axis = 1)

        # Combine scores in a dict per horizon
        if "score" in predictions[0]:
            result["score"] = {h: pred["score"] for h, pred in zip(self.horizons, predictions)}

        return result

    def set_score_mode(self):
        for m in self.models:
            m.set_score_mode()

    def unset_score_mode(self):
        for m in self.models:
            m.unset_score_mode()

    def update(self, data, update_predictor = True, **params):
        return self(data, update_predictor = update_predictor, track_state = True, **params)

    def fit(self, data, update_predictor = True, **params):
        self.reset_state()
        return self.update(data, update_predictor = update_predictor, **params)
        
