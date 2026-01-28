#%%
from __future__ import annotations
from unittest import result
import numpy as np
import pandas as pd
import re
from typing import Literal
import inspect
import os
import functools
import pickle
from abc import ABC, abstractmethod
from numpy.lib.stride_tricks import sliding_window_view

# Get the directory of the current module
module_dir = os.path.dirname(__file__)

# Construct the path to the data file
data_folder = os.path.join(module_dir, 'data')

def subset_columns(columns, *variables, horizons = None, return_index = False):

    if len(variables) == 0 and horizons is None:
        index = [True]*len(columns)
    elif len(variables) == 0:
        index = [col[1] in horizons for col in columns]
    elif horizons is None:
        index = [col[0] in variables for col in columns]
    else:
        index = [(col[0] in variables) and (col[1] in horizons) for col in columns]

    if return_index:
        return index
    
    return columns[index]

def fc_columns_from_tuples(tuples: list | tuple) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(tuples, names = ['Variable', 'Horizon'])

def fc_columns_from_product(variables: list | tuple, horizons: list | tuple, group_by_horizon: bool = False) -> pd.MultiIndex:
    if group_by_horizon:
        # Sort according to (var1, h1), (var2, h1), ..., (var1, h2), (var2, h2), ...
        tuples = [(var, h) for h in horizons for var in variables]
        return fc_columns_from_tuples(tuples)
    else:
        return pd.MultiIndex.from_product([variables, horizons], names = ['Variable', 'Horizon'])

def check_fc_format(index: pd.Index | list | tuple, as_list = False) -> bool:
    if as_list:
        # Check only list of tuples
        for col in index:
            if not (isinstance(col, tuple) and len(col) == 2 and isinstance(col[1], int)):
                return False

    if not isinstance(index, pd.MultiIndex):
        return False
    if index.names != ['Variable', 'Horizon']:
        return False
    if not all(isinstance(h, int) for h in index.get_level_values('Horizon')):
        return False
    return True

@pd.api.extensions.register_dataframe_accessor("fc")
class ForecastMatrix:
    
    def __init__(self, data: pd.DataFrame):
        self._obj = data
     
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

    def subset(self, *variables, start_time=None, end_time=None, start_index=None, end_index=None, horizons=None, time_col = None) -> DataFrame:
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

        data.columns = pd.MultiIndex.from_tuples(new_columns, names=('Variable', "Horizon"))

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


    def lag(self, *args, reverse = False, **kwargs) -> DataFrame:
        """
        Retrieves a subset of the data lagged according to column name forecast horizons.
        """
        subset = self.subset(*args, **kwargs)
        for i, col in enumerate(subset.columns):
            if col[1] not in [None, ""]:
                k = -col[1] if reverse else col[1]
                subset.iloc[:,  [i]] = subset.iloc[:, [i]].shift(k, fill_value=float("nan"))
        return subset

    def join_variable(self, var, data: pd.DataFrame, how = "right") -> DataFrame:
        data = data.copy()
        new_cols = pd.MultiIndex.from_product([[var], data.columns], names = self._obj.columns.names)
        data.columns = new_cols
        return self.join(data, how = how)

    def join_horizon(self, horizon, data: pd.DataFrame, how = "right") -> DataFrame:
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

class DataFrame(pd.DataFrame):
    fc: ForecastMatrix

def new_fc(data = None, index = None, columns = None, dtype = None, copy = None, separator = '.k') -> DataFrame:
    result = pd.DataFrame(data, index, columns, dtype = dtype, copy = copy)
    if not result.fc.check():
        result = result.fc.convert(separator = separator)
    return result

def forecast_matrix_from_product(names: list | tuple, horizons: tuple) -> DataFrame:
    """
    Create a DataFrame with MultiIndex columns from the product of names and horizons.
    """
    cols = pd.MultiIndex.from_product([names, horizons], names = ['Variable', 'Horizon'])
    return new_fc(columns = cols)

#%%
@functools.wraps(pd.read_csv)
def read_forecast_csv(*args, horizon_pattern = None, **kwargs) -> DataFrame:
    if horizon_pattern is None:
       kwargs = {"header": [0, 1]} | kwargs
    df = pd.read_csv(*args, **kwargs)
    if not df.fc.check():
        df = df.fc.convert(separator = horizon_pattern)
    # TODO: add check that columns are correctly loaded
    df.index.name = None
    return df

if os.path.exists(data_folder) and 'simulated_data.csv' in os.listdir(data_folder):
    sample_data = read_forecast_csv(data_folder + '/simulated_data.csv', horizon_pattern=".k",parse_dates=['t'], index_col = "t")
    
class Source:
    """
    Placeholder class for specifying data sources in transformations.
    """
    _name = None

    def __init__(self, name = None):
        self._name = name

    def __repr__(self):
        return self._name

    def __add__(self, other):
        return SumTransformation(self, other)

    def __radd__(self, other):
        return SumTransformation(other, self)

    def __sub__(self, other):
        return SubTransformation(self, other)
    
    def __rsub__(self, other):
        return SubTransformation(other, self)
    
    def __mul__(self, other):     
        return MulTransformation(self, other)

    def __rmul__(self, other):
        return MulTransformation(self, other)

    def __truediv__(self, other):
        return DivTransformation(self, other)
    
    def __rtruediv__(self, other):
        return DivTransformation(other, self)

    def __pow__(self, other):
        return PowTransformation(self, other)

    def __rpow__(self, other):
        return PowTransformation(other, self)

# NOTE: a special class and __reduce__ method is required for keywords for serialization with pickle
class KeywordSource(Source):
    _registry = {}
    def __reduce__(self):
        return make_keyword, (self._name,)

def make_keyword(name: str) -> Source:
    if name not in KeywordSource._registry:
        KeywordSource._registry[name] = KeywordSource(name)
    return KeywordSource._registry[name]
    
Memory = make_keyword("Memory")
DefaultSource = make_keyword("DefaultSource")
DefaultIndex = make_keyword("DefaultIndex")
PredictorParameters = make_keyword("PredictorParameters")
UpdatePredictor = make_keyword("UpdatePredictor")
X_init = make_keyword("X_init")
Y_init = make_keyword("Y_init")

class Transformation(Source):

    """
    Generic class for transformations. Subclasses should implement the evaluate method.

    The class provides functionality for matching placeholder data sources to data, and evaluating
    the transformation based on the provided data. Subclasses should call super().__init__(*args, **kwargs)
    with keyword names matching the evaluate method parameters to specify how data should be passed.   
    Also enables basic operations between transformations, such as +, -, *, etc.
    """

    # Superclass for transformations
    def __init__(self, *apply_args, **apply_kwargs):

        """
        apply_args: optional positional arguments to specify input data targets for evaluate when called via. apply.
        apply_kwargs: optional dict to specify input data targets for evaluate when called via. apply.
        """
        
        # Iterate args and kwargs, replacing strings with Map
        apply_args = [Map(arg) if isinstance(arg, str) else arg for arg in apply_args]
        apply_kwargs = {key: Map(value) if isinstance(value, str) else value for key, value in apply_kwargs.items()}

        # Check for unused params and whether there is any dependency on Prediction
        self.apply_kwargs = apply_kwargs
        self.apply_args = apply_args
        for key in apply_kwargs:
            if not (key in self._inputs or self._accepts_kwargs):
                raise KeyError(f"{self} does not use input: {key}.")
            if self.apply_kwargs[key] is None:
                self.apply_kwargs[key] = DefaultSource

        # Check that args and kwargs refer to valid inputs
        for val in list(apply_kwargs.values()) + list(apply_args):
            if not isinstance(val, Source):
                raise ValueError(f"Input {val} must be a Source instance: {self}.")

        # Build pairs of (name, value) for args and kwargs combined
        self._apply_pairs = [(None, val) for val in apply_args] + list(apply_kwargs.items())

        # Determine all sources
        self.sources = list(self.apply_kwargs.values()) + list(self.apply_args)
        self.dependencies = [v for v in self.sources if isinstance(v, Transformation)]

        # Fetch evaluate signature
        self._eval_sig = inspect.signature(self.evaluate)

    def __init_subclass__(cls):

        if hasattr(cls, "evaluate"):
            sig = inspect.signature(cls.evaluate)
            cls.evaluate_kwargs = list(sig.parameters.keys())[1:]

            var_positional_arg = next((param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.VAR_POSITIONAL), None)
            var_keyword_arg = next((param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.VAR_KEYWORD), None)

            cls._accepts_args = var_keyword_arg is not None
            cls._accepts_kwargs = var_positional_arg is not None

            cls._inputs = [n for n, p in inspect.signature(cls.evaluate).parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)][1:]

        else:
            raise ValueError("No evaluate method found.")

        # Overwrite init to capture parameters
        original_init = cls.__init__
        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._init_params = inspect.signature(original_init).bind(self, *args, **kwargs).arguments
            del self._init_params['self']

        cls.__init__ = new_init

    def __repr__(self):
        return f"{self.__class__.__name__}"

    #TODO: add check for circular dependencies

    @property
    def _var_positional_arg(self):
        return next((param.name for param in inspect.signature(self.evaluate).parameters.values() if param.kind == inspect.Parameter.VAR_POSITIONAL), None)

    @property
    def _var_keyword_arg(self):
        return next((param.name for param in inspect.signature(self.evaluate).parameters.values() if param.kind == inspect.Parameter.VAR_KEYWORD), None)

    @property
    def _accepts_var_args(self):
        return self._var_positional_arg is not None

    @property
    def _accepts_var_kwargs(self):
        return self._var_keyword_arg is not None

    @property
    def _accepts_any_kwargs(self):
        return self._var_keyword_arg is not None

    def apply(self, data, memory = None, recursion_pars = None, return_recursion_pars = False, ref = None, update_predictor = True, eval_mode = False, copy_data = True, **predictor_params):

        # NOTE: copy_data is used to avoid modifying input data (unless requested). When applied recursively, copy_data should be False, as we do want to update the data with intermediate results.

        # Extract "shared" predictor params
        shared_predictor_params = {k: v for k,v in predictor_params.items() if not isinstance(k, Prediction)}

        data = parse_data(data, ref = ref, copy = copy_data)

        evaluate_kwargs = {}
        evaluate_args = []

        if recursion_pars is None:
            recursion_pars = {}        

        new_recursion_pars = {}

        if memory is None:
            memory = recursion_pars.get(self, None)

        # Check inputs
        for name, val in self._apply_pairs:

            # Already in data?
            if val in data:
                t_val = data[val]

            elif val is Memory:
                # TODO: consider fetching default from evaluate signature
                t_val = memory

            elif val is UpdatePredictor:
                t_val = update_predictor
            
            elif val is PredictorParameters:
                t_val = shared_predictor_params | predictor_params.get(self, {})

            # Attempt to fetch transformation dependencies if not provided directly.
            elif isinstance(val, Transformation):
                t_val, t_rec_pars = val.apply(data = data, recursion_pars = recursion_pars, return_recursion_pars = True, eval_mode=eval_mode, copy_data=False, **predictor_params)

                new_recursion_pars.update(t_rec_pars)

                data[val] = t_val # Store in data for potential reuse

            else:
                raise ValueError(f"Missing data for input: {val} in {self}.")

            # Store transformed value in evaluate args
            if name is None:
                evaluate_args.append(t_val)
            else:
                evaluate_kwargs[name] = t_val

        # Evaluate
        eval_out = self.evaluate(*evaluate_args, **evaluate_kwargs)

        if isinstance(eval_out, tuple):
            result, memory = eval_out
        else:
            result = eval_out
            memory = None

        new_recursion_pars[self] = memory

        if return_recursion_pars:
            return result, new_recursion_pars
        else:
            return result

    def ancestors(self) -> list[Source]:
        # Recursively get the top most dependencies (sources)
        result = set()
        for dep in self.sources:
            if isinstance(dep, Transformation):
                for anc in dep.ancestors():
                    result.add(anc)
            elif dep not in [Memory, PredictorParameters, UpdatePredictor]:
                result.add(dep)
        return list(result)

    def evaluate(self) -> tuple[DataFrame, dict] | DataFrame:
        # In: signature can be freely defined by subclasses.
        # Out: result or tuple of (result, memory), where memory 
        # should be passed to subsequent calls. 

        raise NotImplementedError("This method should be overridden by subclasses")

    def print_dependency_tree(self, level = 0):
        # Recursively print dependency tree until sources
        indent = "  " * level
        print(f"{indent}- {self}")
        for dep in self.dependencies:
            if isinstance(dep, Transformation):
                dep.print_dependency_tree(level + 1)
            else:
                print(f"{indent}  - {dep}")

    def get_all_dependencies(self) -> list[Transformation]:
        # Recursively get all dependencies
        result = set()
        for dep in self.dependencies:
            result.add(dep)
            if isinstance(dep, Transformation):
                for sub_dep in dep.get_all_dependencies():
                    result.add(sub_dep)
        return list(result)

    def get_graph(self, use_names = True) -> dict:
        # Get all unique nodes
        nodes = self.get_all_dependencies() + [self]

        result = {}
        counts = {}
        names = {}
        for node in nodes:
            # Get name; either from _name or using class name + count
            if use_names:
                name = node._name
                if name is None and node not in names:
                    class_name = node.__class__.__name__
                    count = counts.get(class_name, 0) + 1
                    counts[class_name] = count
                    name = f"{class_name}_{count}"
                    names[node] = name
            else:
                names[node] = node

            result[names[node]] = {
                "class": node.__class__.__name__,
                "params": node._init_params,
                }
            
        if use_names:
            # Replace params Sources with their names
            for node_name, node_info in result.items():
                for key, val in node_info["params"].items():
                    if isinstance(val, Source):
                        if val in names:
                            result[node_name]["params"][key] = names[val]

        return result
    
### Numpy / Pandas conversion utilities
def data_decorator(func):
    """
    Decorator to apply a function to each value in a dictionary.
    """
    @functools.wraps(func)
    def wrapper(val, *args, **kwargs):
        if isinstance(val, dict):
            result = {}
            for key, v in val.items():
                _args = [a.get(key, None) if isinstance(a, dict) else a for a in args]
                _kwargs = {k: (v.get(key, None) if isinstance(v, dict) else v) for k, v in kwargs.items()}
                result[key] = func(v, * _args, **_kwargs)
            return result
        else:
            return func(val, *args, **kwargs)
    return wrapper

@data_decorator
def shift(data, amount):
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.shift(amount)
    elif isinstance(data, np.ndarray):
        if amount > 0:
            shifted = np.empty_like(data)
            shifted[:amount] = np.nan
            shifted[amount:] = data[:-amount]
            return shifted
        elif amount < 0:
            shifted = np.empty_like(data)
            shifted[amount:] = np.nan
            shifted[:amount] = data[-amount:]
            return shifted
        else:
            return data
    else:
        raise ValueError(f"Cannot shift type {type(data)}.")

@data_decorator
def empty_like(ref: dict | pd.DataFrame | pd.Series | np.ndarray):
    if isinstance(ref, pd.DataFrame):
        return pd.DataFrame(np.full(ref.to_numpy().shape, np.nan), index = ref.index, columns = ref.columns)
    elif isinstance(ref, pd.Series):
        return pd.Series(np.full(ref.to_numpy().shape, np.nan), index = ref.index, name = ref.name)
    elif isinstance(ref, np.ndarray):
        return np.full(ref.shape, np.nan)
    else:
        raise ValueError(f"Cannot create empty array like type {type(ref)}.")

def get_ndim(data: dict | pd.DataFrame | pd.Series | np.ndarray, share_ndim = True):
    if isinstance(data, dict):
        if share_ndim:
            ref_val = data[next(iter(data) if data is not None else None)]
            return get_ndim(ref_val, share_ndim = True)
        else:
            result = {}
            for key, val in data.items():
                result[key] = get_ndim(val, share_ndim = False)
            return result
    
    else:
        return data.ndim

def get_num_obs(val):
    if isinstance(val, dict):
        val = get_num_obs(next(iter(val.values())))
    else:
        return val.shape[0]

@data_decorator
def get_element(val, i):
    if isinstance(val, np.ndarray):
        return val[i]
    elif isinstance(val, pd.DataFrame):
        return val.iloc[i]
    else:
        return val

@data_decorator
def get_dim(val, axis = 1):
    return val.shape[axis]

def get_index(data: dict | pd.DataFrame | pd.Series, share_index = True):
    if isinstance(data, dict):
        if share_index:
            ref_val = data[next(iter(data) if data is not None else None)]
            return get_index(ref_val, share_index = True)
        else:
            result = {}
            for key, val in data.items():
                result[key] = get_index(val, share_index = False)
            return result
        
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.index
    else:
        return None

def parse_data(data : dict | pd.DataFrame | pd.Series | np.ndarray, ref = None, copy = True):
    if not isinstance(data, dict):
        data = {DefaultSource: data}
    elif copy:
        data = data.copy()
    
    ref_val = data[ref or next(iter(data))]
 
    if not DefaultSource in data:
        data[DefaultSource] = ref_val

    data[DefaultIndex] = get_index(ref_val)

    return data

@data_decorator
def to_numpy(val, ensure_dim: int = None, squeeze_left = False):

    if isinstance(val, (pd.Series, pd.DataFrame)):
        val = val.to_numpy()
    elif not isinstance(val, np.ndarray):
        raise ValueError(f"Cannot convert type {type(val)} to numpy array.")

    if ensure_dim is None:
        return val

    diff = ensure_dim - val.ndim

    if diff == 0:
        return val

    if diff > 0:
        new_shape = val.shape + (1,) * diff
        return val.reshape(new_shape)
    else:
        if squeeze_left:
            squeeze_dims = tuple(range(-diff))
        else:
            squeeze_dims = tuple(range(diff, 0))

        return val.squeeze(axis = squeeze_dims)

@data_decorator
def squeeze_last(val):
    if val.ndim > 1 and val.shape[-1] == 1:
        return val.squeeze(axis = -1)
    else:
        return val

@data_decorator
def reshape_like(val, ref_val):
    if val.shape == ref_val.shape:
        return val
    else:
        return val.reshape(ref_val.shape)

def to_numpy_like(val, ref_val):
    val = to_numpy(val)
    return reshape_like(val, ref_val)

@data_decorator
def to_pandas(val, index = None, columns = None, outer_prod = False):
    
    if isinstance(val, (pd.Series, pd.DataFrame)):
        val = val.to_numpy()

    if outer_prod and columns is not None: # Get outer product of columns?
        # Form output variable names as product of input variable names (for covariances)
        vars = columns.unique().tolist()
        ran = range(len(vars))
        flat_vars = [(vars[i], vars[j]) for i in ran for j in ran]
        columns = flat_vars

    # If columns has no length, format as series
    if not isinstance(columns, (list, tuple, pd.Index)):
        val = val.squeeze()
        if val.ndim <= 1:
            return pd.Series(val, name = columns, index = index)
        else:
            return pd.Series([val], name = columns, index = index)

    else:
        if val.ndim == 2:
            return pd.DataFrame(val, index = index, columns = columns)
        else:
            # Try to reshape to 2D
            reshaped = val.reshape((val.shape[0], -1))
            if columns is not None and len(columns) != reshaped.shape[1]:
                columns = [f"{var}_{i}" for var in columns for i in range(reshaped.shape[1] // len(columns))]
            return pd.DataFrame(reshaped, index = index, columns = columns)

@data_decorator
def to_pandas_like(val, ref_val, outer_prod = False):
    index = None
    columns = None
    if isinstance(ref_val, pd.Series):
        index = ref_val.index
        columns = ref_val.name
    elif isinstance(ref_val, pd.DataFrame):
        index = ref_val.index
        columns = ref_val.columns
    return to_pandas(val, index, columns, outer_prod)

@data_decorator
def format_like(val, ref_val, outer_prod = False):
    if isinstance(ref_val, (pd.Series, pd.DataFrame)):
        return to_pandas_like(val, ref_val, outer_prod = outer_prod)
    
    if isinstance(ref_val, np.ndarray):
        return to_numpy_like(val, ref_val)

    raise ValueError(f"Cannot format like type {type(ref_val)}.")

@data_decorator
def get_vars(val):
    if isinstance(val, pd.Series):
        return val.name
    elif isinstance(val, pd.DataFrame):
        return val.columns
    else:
        return None

@data_decorator
def check_var(val, ref):
    if ref is not None:
        if not isinstance(val, (pd.Series, pd.DataFrame)):
            raise ValueError("Data failed format check: expected pandas Series or DataFrame.")
        data_vars = get_vars(val)
        if not all(var in data_vars for var in ref):
            raise ValueError("Data failed format check: missing variables.")


@data_decorator
def apply_format(val, variables = None, index = None, outer_prod = False, ensure_dim = None):
    if variables is None:
        return to_numpy(val, ensure_dim = ensure_dim)
    else:
        return to_pandas(val, index, variables, outer_prod)

class DataFormat:

    def __init__(self, variables, outer_prod = False):
        self.variables = variables # variable names or a dict of variables for nested data
        self.outer_prod = outer_prod

    def apply(self, data, index = None):
        return to_pandas(data, index, self.variables, outer_prod = self.outer_prod)

    def check(self, data):
        if self.variables is not None:
            check_var(data, self.variables)

    @classmethod
    def from_reference(cls, val: dict | pd.DataFrame | pd.Series | np.ndarray, outer_prod = False):
        vars = get_vars(val)
        return cls(variables = vars, outer_prod = outer_prod)

    def __call__(self, data, index = None):
        return self.apply(data, index)
    

def standardize_wrapper(*inputs, ensure_dim = None, output_as: str | dict = None, outer_prod: bool | dict = False):
    """
    Wrapper factory, for standardizing inputs, and optionally outputs, of functions.
    - inputs: names of input arguments to convert to numpy arrays.
    - apply_to_args: whether to also convert positional arguments to numpy arrays.
    - ensure_dim: whether to convert ensure a certain number of dimensions for input arrays. If int, all specified inputs are converted to have at least that many dimensions.
    - output_as: reference input name(s) for formatting output(s). If str, single output is formatted like the specified input. If dict, key output names are mapped to input names for formatting.
    - outer_prod: whether to use outer product formatting for outputs (e.g. for covariances). Can be a bool or dict mapping output keys to bools.
    """

    def decorator(func):
        # TODO: consider efficiency improvements for repeated calls
        sig = inspect.signature(func)
        # Check that inputs match signature
        for name in inputs:
            if not name in sig.parameters:
                raise ValueError(f"Input name {name} not found in signature.")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            bound_args = sig.bind(*args, **kwargs)

            # If output is to be formatted, fetch format from input data
            if output_as is not None:

                # Get refernece value
                if isinstance(output_as, str):
                    ref_val = bound_args.arguments.get(output_as, None)

                else: # Assuming dict
                    ref_val = {op: bound_args.arguments.get(ref_name, None) for op, ref_name in output_as.items()}   

            # Convert inputs to arrays
            for name in inputs:
                if name in bound_args.arguments:
                    value = bound_args.arguments[name]
                    if value is not None:
                        bound_args.arguments[name] = to_numpy(value, ensure_dim)

            # Call function
            output = func(*bound_args.args, **bound_args.kwargs)

            if isinstance(output, tuple):
                result, memory = output
            else:
                result = output
                memory = None

            # Convert output to desired format
            if output_as is not None:
            
                result = format_like(result, ref_val, outer_prod = outer_prod)
        
            return result, memory

        return wrapper
    
    return decorator

### Transformations

class BackShift(Transformation):

    def __init__(self, shifts: list | dict, data = DefaultSource, skip_duplicates = False, initial_value = np.nan):
        """
        shifts: list of lists or dict. If list of lists, each inner list represents a row of the bacshift matrix, in which case None values are treated as zeros.
        If dict, keys are (i,j) tuples representing the row and column indices of non-zero entries, and values are the corresponding shifts.
        """

        # Check that object is a list of lists of same length
        if not isinstance(shifts, dict):

            if isinstance(shifts, list) and not isinstance(shifts[0], list):
                # Assume "diagonal" structure with single input, i.e. m = 1
                shifts = {(i, 0): s for i, s in enumerate(shifts) if s is not None}

            # Check that all sublists have same length
            elif not all(len(s) == len(shifts[0]) for s in shifts):
                raise ValueError("All sublists must have the same length")
            else:
                shifts = {(i, j): s for i, s in enumerate(shifts) for j, s in enumerate(s) if s is not None}

        self.n = max(i for i, j in shifts.keys()) + 1 # Number of outputs

        if self.n == 0:
            raise ValueError("Shifts cannot be empty")


        self.max_shifts = {i: max(s for (ii, j), s in shifts.items() if ii == i) for i in range(self.n)}
        self.max_shift = max(self.max_shifts.values())
        self.shifts = shifts
        self.skip_duplicates = skip_duplicates
        self.initial_value = initial_value
        super().__init__(data = data, memory = Memory)

    @standardize_wrapper("data", ensure_dim=2)
    def evaluate(self, data, memory = None):
        # Fetch data from memory
        # TODO: use CircularBuffer for efficiency
        if memory is None:
            old_data = np.full((self.max_shift, data.shape[1]), self.initial_value)
            if self.skip_duplicates:
                offset = 0
        else:
            old_data, offset = memory

        all_data = np.vstack((old_data, data))

        shifted_data = {}
        # Input Z (t x n) -> Temp Y -> Output X
        t = data.shape[0]

        # Collect lagged series
        for (i,j), lag in self.shifts.items():
            if (j, lag) not in shifted_data:
                if lag == 0:
                    shifted_data[(j, lag)] = all_data[self.max_shift:self.max_shift + t, j]
                elif lag > 0:
                    shifted_data[(j, lag)] = all_data[-(t + lag):-lag, j]

        # Form output
        if self.skip_duplicates:
            X = np.full((t, self.n), np.nan)

            # Get mask to select every max(lag)'th row, starting from an offset
            mask = np.arange(self.max_shift - offset, t, self.max_shift + 1)
            X[mask, :] = 0

            # Update offset
            offset = (offset + t) % (self.max_shift + 1)

            # Sum contributions
            for (i,j), lag in self.shifts.items():
                X[mask, i] += shifted_data[(j, lag)][mask]
        else:
            X = np.zeros((t, self.n))

            for (i,j), lag in self.shifts.items():

                X[:, i] += shifted_data[(j, lag)]

        if self.skip_duplicates:
            return X, (all_data[-self.max_shift:], offset)
        else:
            return X, (all_data[-self.max_shift:], None)

class Index(Transformation):

    def __init__(self, data = DefaultSource):
        super().__init__(data = data)
    
    def evaluate(self, data):
        return get_index(data)

def get_horizons(columns: pd.MultiIndex | tuple):
    if isinstance(columns, pd.MultiIndex):
        return columns.get_level_values(1).unique().tolist()
    if isinstance(columns, tuple) and len(columns) == 2 and isinstance(columns[1], int):
        return columns[1]
    else:
        raise ValueError("Columns do not have MultiIndex format.")

class Horizons(Transformation):
    
    def __init__(self, data = DefaultSource):
        super().__init__(data = data)

    def evaluate(self, data):
        if isinstance(data, pd.DataFrame):
            vars = data.columns
        elif isinstance(data, pd.Series):
            vars =  data.name
        return get_horizons(vars)

class Dim(Transformation):
    
    def __init__(self, data, axis = 1):
        self.axis = axis
        super().__init__(data = data)

    def evaluate(self, data):
        return get_dim(data, axis = self.axis)
    
DimX = Dim(X_init)
DimY = Dim(Y_init)

class Length(Transformation):
    
    def __init__(self, data):
        super().__init__(data = data)

    def evaluate(self, data):
        return get_num_obs(data)

class One(Transformation):
    
    def __init__(self, index = DefaultIndex):
        super().__init__(index = index)

    def evaluate(self, index):
        if isinstance(index, (pd.Index, np.ndarray)):
            return np.ones((len(index), 1))
        else:
            return np.ones(1)
        
class LowPass(Transformation):
    def __init__(self, var, alpha = 0):
        super().__init__(data=var, prev_value = Memory)
        self.alpha = alpha

    @standardize_wrapper("data", ensure_dim=2, output_as = "data")
    def evaluate(self, data: pd.DataFrame | pd.Series, prev_value=None):
        alpha = self.alpha
        n, m = data.shape

        new_vals = np.full((n + 1, m), np.nan)
        if prev_value is not None:
            new_vals[0] = prev_value

        for i in range(n):
            new_vals[i+1] = alpha* new_vals[i] + (1 - alpha) * data[i]

            # If any NaNs, replace them with new values
            new_vals[i+1] = np.where(np.isnan(new_vals[i+1]), data[i], new_vals[i+1])
            # If any NaNs still present, replace with prevous values
            new_vals[i+1] = np.where(np.isnan(new_vals[i+1]), new_vals[i-1], new_vals[i+1])

        return new_vals[1:], new_vals[-1]

class FourierSeries(Transformation):

    def __init__(self, data, nharmonics = 1):
        self.nharmonics = nharmonics
        super().__init__(data = data)

    @standardize_wrapper("data", ensure_dim=2)
    def evaluate(self, data):            
        results = []
        for i in range(1, self.nharmonics + 1):
            results.append(np.cos(2*np.pi*i*data))
            results.append(np.sin(2*np.pi*i*data))
    
        result = np.hstack(results)

        return result

class SlidingSum(Transformation):

    def __init__(self, data = DefaultSource, window_size = 1, *args, **kwargs):
        self.window_size = window_size
        super().__init__(data = data, old_data = Memory)
        self._args = args
        self._kwargs = kwargs

    @standardize_wrapper("data", ensure_dim=2, output_as = "data")
    def evaluate(self, data, old_data = None):
        if old_data is None:
            n_vars = data.shape[1]
            old_data = np.full((self.window_size - 1, n_vars), np.nan)
                
        extended_data = np.concatenate((old_data, data))
        windows = sliding_window_view(extended_data, self.window_size, *self._args, axis = 0, **self._kwargs)

        result_all = np.sum(windows, axis = -1)

        result = result_all[-data.shape[0]:]
        old_data = extended_data[-(self.window_size - 1):]

        return result, old_data

class SlidingMean(Transformation):

    def __init__(self, data = None, window_size = None, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.window_size = window_size
        super().__init__(data = data, old_data = Memory)

    @standardize_wrapper("data", ensure_dim=2, output_as = "data")
    def evaluate(self, data, old_data = None):
        if old_data is None:
            n_vars = data.shape[1]
            old_data = np.full((self.window_size - 1, n_vars), np.nan)
                
        extended_data = np.concatenate((old_data, data))
        windows = sliding_window_view(extended_data, self.window_size, *self._args, axis = 0, **self._kwargs)

        result_all = np.mean(windows, axis = -1)

        result = result_all[-data.shape[0]:]
        old_data = extended_data[-(self.window_size - 1):]

        return result, old_data

def forgetting_mean(forgetting, data, state, track_memory = False):

    collapse = tuple(i for i in range(1, data.ndim))
    mask = ~np.isnan(data).any(axis = collapse)
    clean_data = data[mask]

    n = clean_data.shape[0]

    if state is None:
        old_est = np.zeros_like(data[0])
        memory = 0
    else:
        old_est, memory = state

    # Return nan array if clean_data is empty
    if n == 0:
        return np.full_like(data, np.nan), (old_est, memory)

    result = np.full_like(clean_data, np.nan)

    if track_memory:
        # Old estimate is unnormalized sum
        w_sum = old_est

        for i in range(n):
            w_sum = forgetting * w_sum + clean_data[i]
            memory = memory * forgetting + 1
            result[i] = w_sum / memory

        # Store data for next iteration
        new_est = w_sum

    else:
        # Assume saturated memory and update mean estimate directly (old_est is mean)
        result[0] = forgetting * old_est + (1 - forgetting) * clean_data[0]
        for i in range(1, n):
            result[i] = forgetting * result[i-1] + (1 - forgetting) * clean_data[i]

        # Store data for next iteration
        new_est = result[-1]
        memory = None

    result_full = np.full_like(data, np.nan)
    result_full[mask] = result

    return result_full, (new_est, memory)

class ForgettingMean(Transformation):

    def __init__(self, forgetting, data = DefaultSource, track_memory = True):
        super().__init__(data, state = Memory)
        self.forgetting = forgetting
        self.track_memory = track_memory
        self.data = data

    @standardize_wrapper("data", ensure_dim=2, output_as = "data")
    def evaluate(self, data, state = None):
        return forgetting_mean(self.forgetting, data, state, self.track_memory)

class ForgettingVariance(Transformation):

    def __init__(self, forgetting, data = DefaultSource, track_memory = True, center = True, covariance = False):
        if center:
            if isinstance(center, Transformation):
                mean = center
            else:
                mean = ForgettingMean(forgetting, data = data, track_memory = track_memory)
            super().__init__(mean = mean, data = data, state = Memory)
        else:
            super().__init__(data = data, state = Memory)

        self.forgetting = forgetting
        self.track_memory = track_memory
        self.covariance = covariance

    def evaluate(self, data, mean = None, state = None):
        if self.covariance:
            return self._eval_covariance(data, mean, state)
        else:
            return self._eval(data, mean, state)

    @standardize_wrapper("data","mean", output_as = "data", outer_prod = True, ensure_dim = 2)
    def _eval_covariance(self, data, mean = None, state = None):
        return self._evaluate(data, mean, state)
    
    @standardize_wrapper("data","mean", output_as = "data", ensure_dim = 2)
    def _eval(self, data, mean = None, state = None):
        return self._evaluate(data, mean, state)

    def _evaluate(self, data, mean = None, state = None):
        
        # Compute unentered variance
        if self.covariance:
            data = np.einsum('ij,ik->ijk', data, data)
        else:
            data = data**2

        var, state = forgetting_mean(self.forgetting, data, state, self.track_memory)

        # Center estimate if mean is provided
        if mean is not None:
            if self.covariance:
                mean_outer = np.einsum('ij,ik->ijk', mean, mean)
                var = var - mean_outer
            else:
                mean_sq = mean**2
                var = var - mean_sq

        return var, state

class FSDay(Transformation):
    
    def __init__(self, freq, t = DefaultIndex, nharmonics = 1, horizons = None):
        self.nharmonics = nharmonics
        self.freq = freq
        self.horizons = [0] if horizons is None else horizons
        super().__init__(t = t, pre_computed = Memory)

    def evaluate(self, t, pre_computed = None):
        if isinstance(t, pd.DataFrame):
            ts = pd.DatetimeIndex(t.iloc[:,0])
        elif isinstance(t, pd.DatetimeIndex):
            ts = t
        elif isinstance(t, pd.Timestamp):
            ts = pd.DatetimeIndex([t])
        else:
            raise ValueError("t must be a pd.Timestamp, pd.DatetimeIndex or DataFrame.")

        if pre_computed is None:

            # Create set of times based on freq and first entry in t
            end = ts[0] + pd.Timedelta(days = 1)
            times = pd.date_range(start = ts[0], end = end, freq = self.freq, inclusive = 'left')

            # Compute times of day,
            tod = TimeOfDay.evaluate(None, times)

            # Prepare dataframe indexed by times
            data = []
            # Populate with pre-computed Fourier features
            for i in range(1, self.nharmonics + 1):
                data.append(np.cos(2 * np.pi * i * tod))
                data.append(np.sin(2 * np.pi * i * tod))

            fs = np.vstack(data).T

            # Roll features per horizon
            pre_computed = []
            for h in self.horizons:
                pre_computed.append(np.roll(fs, -h, axis = 0))

            # Combine into single DataFrame with multiindex
            variables = [f"fs_day_{i}_{fn}" for i in range(1, self.nharmonics + 1) for fn in ['cos', 'sin']]
            columns = fc_columns_from_product(variables, self.horizons, group_by_horizon=True)
            pre_computed = pd.DataFrame(np.hstack(pre_computed), columns=columns)
            pre_computed.index = times.time
        
        # Compute output per horizon
        data = pre_computed.loc[ts.time]

        data.index = ts

        if isinstance(t, pd.Timestamp):
            data = data.iloc[0]
        elif isinstance(t, pd.DataFrame):
            data.index = t.index

        return data

class Disruption(Transformation):
    # A class for modelling disruptions at a specific day and hour

    def __init__(self, hour, dayofweek = None, duration = None, horizons: int | list = 0, index = DefaultIndex):
        super().__init__(index = index)
        self.hour = hour
        self.dayofweek = dayofweek
        self.duration = duration
        self.end_hour = (hour + duration) % 24 if duration is not None else None
        self.horizons = horizons if isinstance(horizons, list) else [horizons]
    
    def evaluate(self, index):
        result = {}
        for h in self.horizons:
            result[h] = self.evaluate_horizon(index, h) 

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

class SumHorizons(Transformation):
    
    """
    Transformation that sums the values of a variable across specific horizons.
    """

    def __init__(self, variable, start = None, end = None):
        super().__init__(variable = variable, indexer = Memory)
        self.start = start
        self.end = end
        self.name = f'sum_{variable}'

    def evaluate(self, variable, indexer = None):
        start, end = self.start, self.end
        if indexer is None and (start is not None or end is not None):
            if start is None:
                start = 0
            if end is None:
                end = variable.shape[1]
            indexer = subset_columns(variable.columns, horizons = range(start, end), return_index = True)
        if indexer is None:
            result = variable.sum(axis = 1), {"indexer": None}
        else:
            result = variable.iloc[:, indexer].sum(axis = 1), {"indexer": indexer}

        result[0].name = self.name

        return result
        
class BSpline(Transformation):
    # == Description ==============================================================================
    # This class define the B-splines transformation using the Cox-de Boor recursive algorithm.
    # The B-spline basis functions are defined by a knot sequence and a degree.
    # 
    # The default range of the knot sequence is [0, 1]. For other ranges the scaling parameters 
    # need to be changed and implemented in the evaluate method.

    
    def __init__(self, data, degree = 3, knots = [0, 0.33, 0.67, 1], clamping = True, scaling = "None"):
        super().__init__(data = data)
        self.degree = degree
        self.knots = knots
        self.clamping = clamping
        self.scaling = scaling

    def evaluate(self, data):
        
        # == Error checking =======================================================================

        if self.scaling == "None":
            if self.knots[0] != 0 or self.knots[-1] != 1:
                raise ValueError("For scaling='None', the first knot must be 0 and the last knot must be 1.")

            if not ((self.knots[0] <= data) &  (data <= self.knots[-1])).all().all():
                raise ValueError(f"Some data points are out of the range defined by the knots: [{self.knots[0]}, {self.knots[-1]}].")

        # TODO add other cases for the scaling 

        if sorted(self.knots) != self.knots:
            raise ValueError("The knot sequence must be non-decreasing.")
        
        # Add clamping if selected
        if self.clamping:
            # Check if the first and last knots are repeated degree + 1 times if not do it
            if self.knots[0] != self.knots[self.degree] or self.knots[-1] != self.knots[-(self.degree + 1)]:
                # Repeat the first and last knots degree + 1 times
                self.knots = [self.knots[0]] * (self.degree) + self.knots + [self.knots[-1]] * (self.degree)

        if len(self.knots) < self.degree + 2:
            raise ValueError("The knot sequence must have at least degree + 2 elements.")

        # == Main code ============================================================================

        # Initialize the output DataFrame with the same structure as data
        if isinstance(data, pd.Series):
            result_df = pd.DataFrame(index = data.index)
            data = data.to_frame()

        else:
            result_df = data.copy()
            result_df = result_df.iloc[:, :0]  # Remove all columns, keeping only the index

        # Apply the B-spline basis function to each element in the data
        for j in range(len(self.knots) - self.degree - 1):
            column_name = f"BSpline{j + 1}"
            result_df[column_name] = [cox_de_boor_recursive(row.values[0], j, self.degree, self.knots) for _, row in data.iterrows()]

        return result_df

def cox_de_boor_recursive(x, i, k, knots):
    """
    Recursive implementation of the Cox-de Boor formula for B-splines.

    Parameters:
    x : float
        The point at which to evaluate the B-spline basis function.
    i : int
        The index of the basis function.
    k : int
        The degree of the B-spline.
    knots : list or array
        The knot sequence.

    Returns:
    float
        The value of the B-spline basis function at x.
    """
    # Error checking
    if not isinstance(x, (int, float)):
        raise TypeError("x must be a number (int or float).")
    if not isinstance(i, int) or i < 0:
        raise ValueError("i must be a non-negative integer.")
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer.")
    if not isinstance(knots, (list, np.ndarray)):
        raise TypeError("knots must be a list or numpy array.")

    if i >= len(knots) - k - 1:
        raise ValueError("i is out of range for the given knot sequence and degree.")

    if k == 0:
        # Special handling for the right endpoint
        if knots[i] <= x < knots[i + 1] or (x == knots[-1] and x == knots[i + 1]):
            return 1.0
        else:
            return 0.0
    else:
        # Recursive case
        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]

        term1 = ((x - knots[i]) / denom1 * cox_de_boor_recursive(x, i, k - 1, knots)) if denom1 != 0 else 0
        term2 = ((knots[i + k + 1] - x) / denom2 * cox_de_boor_recursive(x, i + 1, k - 1, knots)) if denom2 != 0 else 0

        return term1 + term2

class PrimitiveTransformation(Transformation):

    def __init__(self, a, b):

        if not isinstance(a, Source):
            a = Param(a)

        if not isinstance(b, Source):
            b = Param(b)

        self.a, self.b = a, b

        super().__init__(a = a, b = b)

class SumTransformation(PrimitiveTransformation):

    def evaluate(self, a, b):
        return a+b
    
class MulTransformation(PrimitiveTransformation):

    def evaluate(self, a, b):
        return a*b 

class DivTransformation(PrimitiveTransformation):    

    def evaluate(self, a, b):
        return a/b
        
class SubTransformation(PrimitiveTransformation):        

    def evaluate(self, a, b):
        return a-b  

class PowTransformation(PrimitiveTransformation):

    def evaluate(self, a, b):
        return a**b  

class TimeOfDay(Transformation):

    def __init__(self, t = DefaultIndex):
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

    def __init__(self, t=DefaultIndex):
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

    def __init__(self, t=DefaultIndex):
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

class Identity(Transformation):

    def __init__(self, data):
        super().__init__(data = data)

    def evaluate(self, data):
         return {"id": data}

class Param(Transformation):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self):
        return self.value

class ToArray(Transformation):

    def __init__(self, data):
        super().__init__(data = data)

    def evaluate(self, data):
        return data.to_numpy()

class Map(Transformation):

    def __init__(self, *vars, data = DefaultSource):
        super().__init__(data = data)
        self.vars = list(vars)
    
    def evaluate(self, data):
        # TODO: consider storing index to make this more efficient
        return data[self.vars]

    def __repr__(self):
        return super().__repr__() + f"({self.vars})"

class Select(Transformation):
    
    """
    Select specific keys from a dict or columns from a DataFrame.
    """

    def __init__(self, data, *keys):
        super().__init__(data = data)
        self.keys = keys
    
    def evaluate(self, data):
        if isinstance(data, dict):
            result = {k: data[k] for k in self.keys}
            if len(self.keys) == 1:
                return next(iter(result.values()))
            else:
                return result
        elif isinstance(data, pd.DataFrame):
            return data[self.keys]
        else:
            raise ValueError(f"Cannot select keys from {type(data)}.")

class SelectIndices(Transformation):

    def __init__(self, indices, data = DefaultSource):
        super().__init__(data = data)
        self.indices = list(indices)
    
    def evaluate(self, data):
        if isinstance(data, pd.Series):
            raise ValueError("Cannot select columns from a pd.Series.")
        elif isinstance(data, pd.DataFrame):
            return data.iloc[:, self.indices]
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                raise ValueError("Cannot select columns from a 1D numpy array.")
            else:
                return data[:, self.indices]
        else:
            raise ValueError(f"Cannot select columns from {type(data)}.")

def flatten_data(data, path: list = None):
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if path is None:
                sub_path = key
            elif isinstance(path, tuple):
                sub_path = path + (key,)
            else:
                sub_path = (path, key)

            result.update(flatten_data(value, sub_path))
        return result
    else:
        return {path: data}

def combine_data(data: dict, use_format = None, index = None, columns = None, use_fc_format = True, parse_fc_cols =True):

    # First, flatten any dict inputs
    data = flatten_data(data)

    if use_format is None:
        use_format = any(isinstance(d, (pd.Series, pd.DataFrame)) for d in data.values())

    # Convert to numpy and append dimension if needed. We assume the first dimension is time and each pice of data is 2D.
    converted_data = to_numpy(data, ensure_dim = 2)
    
    # Get list of values
    vals = list(converted_data.values())

    # Stack converted_data
    result = np.hstack(vals)

    if use_format:
        # Parse columns
        if columns is None:

            columns = []
            
            # Get columns
            data_cols = get_vars(data)

            # Make pairs of (key, value) from data_cols
            for path, val in converted_data.items():
                cols = data_cols[path]
                if cols is None:
                    # Create default variable names
                    n_vars = val.shape[1]
                    if n_vars == 1:
                        cols = [path]
                    else:
                        cols = [(path, f"/{i}") for i in range(n_vars)]

                # If use_fc_format, create MultiIndex tuples
                if use_fc_format:
                    if check_fc_format(cols, as_list = parse_fc_cols):
                        cols = cols.tolist()
                    else:
                        if parse_fc_cols and isinstance(path, int):
                            cols  = [(var, path) for var in cols]
                        else:
                            cols = [(var, 0) for var in cols]

                columns.extend(cols)

            # Convert to fc format
            if use_fc_format:
                columns = fc_columns_from_tuples(columns)

        # Get index if missing
        if index is None:
            index = get_index(data)

        # Convert result to pandas
        result = to_pandas(result, index, columns)

    # If first data input value is 1D, squeeze result
    nd = next(iter(data.values())).ndim
    if nd == 1:
        result = result.squeeze(0)

    return result, columns

class Combine(Transformation):
    def __init__(self, *sources, format_result = None, index = DefaultIndex, use_fc_format = True, as_dict = False, names = None):
        super().__init__(*sources, index = index, columns = Memory)
        self.format_result = format_result
        self.use_fc_format = use_fc_format
        self.as_dict = as_dict
        self.sources = sources
        if names is not None:
            if len(names) != len(sources):
                raise ValueError("Length of names must match number of sources.")
        else:
            names = self.sources
        self.names = names

    def evaluate(self, *data, index = None, columns = None):
        data_dict = {name: v for name, v in zip(self.names, data)}
        if self.as_dict:
            return data_dict, columns

        result, columns = combine_data(data_dict, use_format = self.format_result, index = index, columns = columns, use_fc_format = self.use_fc_format)

        return result, columns

class ToPandas(Transformation):

    def __init__(self, data, columns, index = DefaultIndex):
        super().__init__(data = data, index = index)
        self.new_columns = columns

    def evaluate(self, data, index = None):
        return to_pandas(data, index, self.new_columns)

class RenameColumns(Transformation):

    def __init__(self, data, new_columns):
        super().__init__(data = data)
        self.new_columns = new_columns

    def evaluate(self, data):
        data.columns = self.new_columns
        return data

class FillMissing(Transformation):

    def __init__(self, data, fill_value=0):
        super().__init__(data=data)
        self.fill_value = fill_value

    def evaluate(self, data):
        if isinstance(data, np.ndarray):
            return np.nan_to_num(data, nan=self.fill_value)
        else:
            return data.fillna(self.fill_value)

class Subset(Transformation):

    def __init__(self, *variables, data = None, horizons = None):
        self.horizons = horizons
        self.variables = list(variables)
        super().__init__(data = data, indices = Memory)


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

    def __init__(self, data, horizons: int | tuple = 0, drop_horizon = False):
        super().__init__(data = data, indices = Memory)
        self.horizons = horizons if isinstance(horizons, tuple) else (horizons,)
        self.drop_horizon = drop_horizon

    def evaluate(self, data, indices = None):
        if isinstance(data, dict):
            # Assume keys are horizons
            result = {}
            for h in self.horizons:
                val = data.get(h, None)
                if val is not None:
                    result[h] = val
                
            if len(result) == 1:
                return next(iter(result.values())), None

            return result, None
            
        if isinstance(data, pd.DataFrame):
            columns = data.columns
        else:
            return data, None

        indices = subset_columns(columns, horizons = self.horizons, return_index = True)
        result = data.iloc[:, indices]

        # Remove horizon level if requested and possible
        if self.drop_horizon:
            cols = get_vars(result)
            if check_fc_format(cols):
                cols = cols.droplevel(1)

            result.columns = cols

        return result, indices

class Lag(Transformation):

    def __init__(self, data, amount = 1, default_value = None, offsets: int | list = None):
        self.amount = amount
        self.fill_value = float("nan") if default_value is None else default_value
        self.offsets = [offsets] if isinstance(offsets, int) else offsets
        # Check that if offsets is specified, all values are less than amount
        if self.offsets is not None:
            for h in self.offsets:
                if h > amount:
                    raise ValueError(f"Offset {h} must be less than lag amount {amount}.")
        super().__init__(data = data, prev_values = Memory)


    def evaluate(self, data, prev_values = None):
        # Evaluate lag across horizons

        # No specified offset
        if self.offsets is None:
            return self.evaluate_offset(data, prev_values, None)

        # A set of offsets
        else:
            
            # Initialize prev_values per offset
            if prev_values is None:
                prev_values = {h: None for h in self.offsets}
            result = {}

            # Evaluate per offset
            for h in self.offsets:
                result[h], prev_values[h] = self.evaluate_offset(data, prev_values[h], h)

            return result, prev_values

    def evaluate_offset(self, data, prev_values = None, offset = None):
        if isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
            return self.evaluate_array(data, prev_values, offset)
        elif isinstance(data, dict):
            return self.evaluate_dict(data, prev_values, offset)
        else:
            raise ValueError(f"Cannot apply Lag to data of type {type(data)}.")

    def evaluate_dict(self, data, prev_values = None, offset = None):
        if prev_values is None:
            prev_values = {k: None for k in data}

        result = {}
        for k, v in data.items():
            result[k], prev_values[k] = self.evaluate_offset(v, prev_values[k], offset)

        return result, prev_values

    @standardize_wrapper("data", ensure_dim=2, output_as = "data")
    def evaluate_array(self, data, prev_values = None, offset = None):

        shift = self.amount
        if offset is not None:
            shift -= offset
        if shift == 0:
            return data, prev_values

        if prev_values is None:
            # Get shape of data to initialize buffer
            m = data.shape[1]
            prev_values = CircularBuffer(shift, m, default_value = self.fill_value)

        result = prev_values.update(data)

        return result, prev_values

class PredictorConfiguration:

    def __init__(self, predictor_type: type[Predictor], *args, output_as = None, outer_prod = None, predictor_params = {}, **kwargs):
        self.predictor_type = predictor_type
        self.args = args
        self.kwargs = kwargs
        self.predictor_params = predictor_params
        self.output_as = output_as
        self.outer_prod = {k: True for k in outer_prod} if outer_prod else None

    def get_value(self, v, data):
        if isinstance(v, Source):
            if v in data:
                return data[v]
            elif isinstance(v, Transformation):
                return v.apply(data)
        else:
            return v

    def create(self, X, Y) -> Predictor:
        data = {X_init: X, Y_init: Y}
        # Construct init params using data if required
        args = []
        for arg in self.args:
            args.append(self.get_value(arg, data))
        kwargs = {}
        for k, v in self.kwargs.items():
            kwargs[k] = self.get_value(v, data)

        predictor = self.predictor_type(*args, **kwargs)
        predictor.output_as = self.output_as
        predictor.outer_prod = self.outer_prod.copy() if self.outer_prod is not None else None
        return predictor

class Predictor(ABC):

    target = None # Default target variable name. Used to extract from prediction results if needed.
    ensure_dim = None # If set, ensures that input data has this many dimensions.

    def __init__(self):
        self.output_as = None
        self.outer_prod = None

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.set_params()
        
    @classmethod
    def set_params(cls):
        raise NotImplementedError("This method should be overridden by subclasses to set parameters.")
    
    @abstractmethod
    def update(self, X, Y, X_train, Y_hat, horizon, **params) -> np.ndarray | dict:
        pass

    @abstractmethod
    def predict(self, X: dict | np.ndarray, horizon, **params) -> np.ndarray | dict:
        # Should return either Y, or (Y, other_results)
        pass

    @abstractmethod
    def get_model_params(self):
        # Should return fitted model parameters
        pass

    @classmethod
    def configure(cls, *args, output_as = None, outer_prod = None, **kwargs):
        return PredictorConfiguration(cls, *args, output_as = output_as, outer_prod = outer_prod, **kwargs)

class BatchPredictor(Predictor):

    @classmethod
    def set_params(cls):
        # Set update parameters for the predictor
        update_sig = inspect.signature(cls.batch_update)
        cls.params = [k for k, v in list(update_sig.parameters.items())[1:] if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) and k not in ["X", "Y", "X_pred", "Y_hat"]]

        # Set parameters for the predict method
        predict_sig = inspect.signature(cls.predict)
        predict_params = [
            k for k, v in list(predict_sig.parameters.items())[1:]
            if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls.predict_params = predict_params

    def update(self, X, Y, X_train, Y_hat, horizon = None, **params):

        forecast = self.batch_update(X, Y, X_train, Y_hat, horizon, **params)

        return forecast

    @abstractmethod
    def batch_update(self, X, Y, X_train, Y_hat, horizon = None, **params) -> tuple[pd.DataFrame] | pd.DataFrame:
        pass

    @abstractmethod
    def predict(self, X, horizon = None, **params):
        pass

def stack_results(forecasts: list[dict] | list[np.ndarray]) -> dict | np.ndarray:

    # Convert to dict of lists
    if isinstance(forecasts[0], dict):
        forecasts = {key: [r[key] for r in forecasts] for key in forecasts[0]}

    # Stack any numpy arrays
    for key, values in forecasts.items(): 

        # Check for a reference value
        ref_val = next(iter(value for value in values if value is not None))

        # Check if any values are None
        if any(value is None for value in values) and ref_val is not None:

            if isinstance(ref_val, np.ndarray):
                ref_shape = ref_val.shape

                # Convert all None values to appropriate arrays of NaNs
                for j, value in enumerate(values):
                    if value is None:
                        new_val = np.full(ref_shape, np.nan)    
                        values[j] = new_val
            
        # Stack values
        forecasts[key] = np.array(values)
    
    return forecasts

class OnlinePredictor(Predictor):

    @classmethod
    def set_params(cls):
        cls.set_predict_params()
        cls.set_update_model_params()
        cls.params = cls.predict_params + cls.update_model_params

    @classmethod
    def set_predict_params(cls):
        predict_sig = inspect.signature(cls.predict)
        predict_params =  [
            k for k, v in list(predict_sig.parameters.items())[1:]
            if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls.predict_params = predict_params

    @classmethod
    def set_update_model_params(cls):
        update_model_sig = inspect.signature(cls.online_update)
        update_model_params = [
            k for k, v in list(update_model_sig.parameters.items())[1:]
            if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls.update_model_params = update_model_params

    def update(self, X: np.ndarray, Y: np.ndarray, X_train: np.ndarray, Y_hat: np.ndarray, horizon = None, **params):
        # Distribute params
        update_params = {k: v for k, v in params.items() if k in self.update_model_params}
        predict_params = {k: v for k, v in params.items() if k in self.predict_params}

        # Assuming X, Y, X_train and Y_hat are numpy arrays with first dimension as number of samples
        if Y_hat is None:
            Y_hat = np.full_like(Y, np.nan)
        
        new_shape = (horizon, *Y_hat.shape[1:])
        Y_hat_new = np.full(new_shape, np.nan)

        Y_hat_all = np.vstack((Y_hat, Y_hat_new))

        n = X.shape[0]

        forecasts = []

        # Loop over each row of data
        for i in range(n):
            
            x = X[i]
            y = Y[i]
            x_train = X_train[i]
            y_hat = Y_hat_all[i]
            y_ready = not np.isnan(y).any()
            x_train_ready = not np.isnan(x_train).any()
   
            # Only update if data is valid
            if x_train_ready and y_ready:
                self.online_update(x_train, y, y_hat, **update_params)

            forecast = self.online_predict(x, **predict_params)

            target = self.target or next(iter(forecast))

            # Update Y_hat
            Y_hat_all[i+horizon] = forecast[target]
 
            forecasts.append(forecast)

        forecasts = stack_results(forecasts)

        return forecasts

    def predict(self, X: np.ndarray, horizon = None, **params):
        # Check parameters
        for k in params.keys():
            if k not in self.predict_params:
                raise ValueError(f"Parameter '{k}' not recognized for prediction.")

        # Predict multiple rows.
        n = X.shape[0]
        forecasts = []

        for i in range(n):
            x = X[i]
            forecast = self.online_predict(x, **params)
            forecasts.append(forecast)

        forecasts = stack_results(forecasts)

        return forecasts

    @abstractmethod
    def online_update(self, x_i, y_i, y_i_hat, **params):
        """
        Update model with new data rows x_i, y_i, and old predictions y_i_hat.
        """

    @abstractmethod
    def online_predict(self, x_i, **params) -> tuple[pd.Series] | pd.Series:
        # Predict a single row.
        pass

class OLS(BatchPredictor):

    target = "mean"
    ensure_dim = 2

    def __init__(self, n, m):
        self.theta = np.zeros((n, m))
        self._n_updates = 0

    def batch_update(self, X: np.ndarray, Y: np.ndarray, X_train: np.ndarray, Y_hat: np.ndarray, horizon, **params):
        # Fit OLS model
        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y).any(axis=1)
        X_fit = X_train[mask]
        Y_fit = Y[mask]
        self.theta = np.linalg.lstsq(X_fit, Y_fit, rcond=None)[0]
        
        pred = self.predict(X, horizon)

        return pred

    def predict(self, X: np.ndarray, horizon):
        # Predict
        pred = X @ self.theta
        return {"mean": pred}

    def get_model_params(self):
        return self.theta

    @classmethod
    def configure(cls):
        return super().configure(n = DimX, m = DimY, output_as = "Y")

class WLS(BatchPredictor):
    
    target = "mean"
    ensure_dim = 2

    def __init__(self, n, m):
        self.theta = np.zeros((n, m))
        self._n_updates = 0

    def batch_update(self, X: np.ndarray, Y: np.ndarray, X_train: np.ndarray, Y_hat: np.ndarray, horizon, W: np.ndarray = None, **params):
        # Fit WLS model
        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(Y).any(axis=1)
        X_fit = X_train[mask]
        Y_fit = Y[mask]

        if W is None:
            W = np.eye(X_fit.shape[0])
        else:
            W = W[np.ix_(mask, mask)]

        XtW = X_fit.T @ W
        self.theta = np.linalg.solve(XtW @ X_fit, XtW @ Y_fit)

        pred = self.predict(X)

        return pred

    def predict(self, X: np.ndarray):
        # Predict
        pred = X @ self.theta
        return {"mean": pred}

    def get_model_params(self):
        return self.theta

    @classmethod
    def configure(cls):
        return super().configure(n = DimX, m = DimY, output_as = "Y")


class RLS(OnlinePredictor):

    target = "mean"
    ensure_dim = 2

    def __init__(self, n_x, n_y, R_scale = 1/10000, burn_in = 1, estimate_variance = False):
        super().__init__()
        # For variance estimation
        self.memory = None
        self.variance_estimate = None

        # Method to setup class after initialization
        theta, R = np.zeros((n_x, n_y)), R_scale*np.eye(n_x)
        
        self.theta: np.ndarray = theta
        self.R: np.ndarray = R
        self.burn_in = burn_in
        self._n_updates = 0
        self.estimate_variance = estimate_variance

    def online_update(self, x_i, y_i, y_i_hat, rls_lambda = 0.99):
        self.R = rls_lambda * self.R + np.outer(x_i, x_i)
        self.theta = self.theta + np.outer(np.linalg.solve(self.R, x_i), y_i - x_i.T @ self.theta)


        if self.estimate_variance:
            resid = - y_i_hat.sub(y_i.values)
            if self.variance_estimate is None:
                self.variance_estimate = np.outer(resid, resid)
            else:
                outer = np.outer(resid, resid)
                # If outer product contains nan, don't update variance estimate, else
                if not np.isnan(outer).any():
                    # Check if variance estimate is nan, if so, set to outer product of residuals
                    if np.isnan(self.variance_estimate).any():
                        self.variance_estimate = outer
                    else:
                        # Update variance estimate
                        self.variance_estimate = rls_lambda * self.variance_estimate + (1 - rls_lambda) * outer

        self._n_updates += 1        

    def predict(self, x: np.ndarray):
#        pred = np.dot(x.T, self.theta)
        pred = x @ self.theta
        
        if self.estimate_variance:
            var_est = self.variance_estimate.astype(float) # TODO: check, is this the intended return value?
            return {"mean": pred, "var": var_est}

        return {"mean": pred}

    def get_model_params(self):
        return self.theta

class RidgePredictor(BatchPredictor):

    target = "mean"
    ensure_dim = 2

    def __init__(self, m, V = None):
        super().__init__()
        self.V = np.eye(m) if V is None else V
        self._old_U = None

    def batch_update(self, X: np.ndarray, Y: np.ndarray, X_train: np.ndarray, Y_hat: np.ndarray, P: np.ndarray = None,
                Q: np.ndarray = None, theta_tilde: np.ndarray = None, U: np.ndarray = None, V: np.ndarray = None,
                estimate_V: bool = False, return_var: bool = False, return_var_theta: bool = False):

        # Copy X, Y
        X_all = X_train.copy()
        Y_all = Y.copy()

        # Remove missing values
        X_mask = ~np.isnan(X_all).any(axis=1)
        Y_mask = ~np.isnan(Y_all).any(axis=1)
        mask = X_mask & Y_mask
        X_train = X_all[mask]
        Y = Y_all[mask]

        # Prepare arrays
        t, n = X_train.shape
        m = Y.shape[1]

        if theta_tilde is None:
            theta_tilde = np.zeros((n, m))

        if P is None:
            P = np.eye(t)
        
        elif not np.allclose(P, P.T):
            P = P + P.T

        if Q is None:
            Q = np.zeros((n, n))
        elif isinstance(Q, (int, float)):
            Q = np.eye(n) * Q

        elif not np.allclose(Q, Q.T):
            Q = Q + Q.T
       
        self.K = K = X_train.T @ P @ X_train + Q
        R = X_train.T @ P @ Y + Q @ theta_tilde

        # Solve
        self.theta = np.linalg.solve(K, R)

        # Predict
        pred = self.predict(X, return_var = False)

        # Estimate variance
        if estimate_V:
            if V is not None:
                raise ValueError("Cannot estimate variance when V is provided.")
            
            if U is None:
                # Assume U is identity

                # Combine old predictions and new
                Y_hat[self.horizon:] = pred[:-self.horizon]

                Y_hat_mask = ~np.isnan(Y_hat).any(axis=1)
                resid_mask = Y_hat_mask & Y_mask

                resid = Y_all[resid_mask] - Y_hat[resid_mask]

                # Estimate variance
                self.V = np.cov(resid, rowvar=False)

            else:
                raise NotImplementedError("Variance estimation conditional on provided U is not implemented in RidgePredictor.")

        V = self.V if V is None else V

        if U is None:
            U = np.eye(X_all.shape[0])

        # Compute variance component in U
        temp1 = P @ X_train
                
        temp2 = temp1.T @ U[np.ix_(mask, mask)] @ temp1
        
        temp3 = np.linalg.solve(K, temp2)

        self.inner_var_vec_theta = np.linalg.solve(K, temp3.T)

        result = {"mean": pred}

        # If required, compute the decomposed out of sample variance (U, V). Note: this should be partly in sample, but is not feasible without assumptions on U.
        if return_var:
            var_vec_Y_hat_err_U = self.get_out_of_sample_variance(X, U)
            result["var"] = var_vec_Y_hat_err_U
            result["V"] = V

        if return_var_theta:
            var_theta = self.get_var_theta(V)
            result["var_theta"] = var_theta

        return result

    def get_model_params(self):
        return self.theta

    def get_out_of_sample_variance(self, X: np.ndarray = None, U: np.ndarray = None):

        # Initialise U
        if U is None:
            U = np.eye(X.shape[0])
        else:
            if not np.allclose(U, U.T):
                raise ValueError("U must be symmetric.")

        # Compute the decomposed out of sample variance (U, V)
        return U + X @ self.inner_var_vec_theta @ X.T

    def get_var_theta(self, V: np.ndarray = None):
        if V is None:
            V = self.V

        # Compute the variance of theta
        var_theta = np.kron(V, self.inner_var_vec_theta)

        return var_theta

    def predict(self, X: np.ndarray, U: np.ndarray = None, V: np.ndarray = None, return_var: bool = False, return_var_theta = False):

        if V is None:
            V = self.V

        # Predict
        pred = X @ self.theta

        result = [pred]

        if return_var:
            var_vec_Y_hat_err_U = self.get_out_of_sample_variance(X, U, V)
            result.extend([var_vec_Y_hat_err_U, V])

        if return_var_theta:
            result.append(self.get_var_theta(V))

        return result if len(result) > 1 else result[0]


    def configure(cls, predictor_params = {}):
        return super().configure(n = DimX, m = DimY, output_as = {"mean": "Y", "cov": "Y"}, outer_prod = ["cov"], predictor_params = predictor_params)

class RRR(OnlinePredictor):
    # Recursive Ridge Regressor
    # Recursively solves the problem
    # min theta || X_t theta - Y_t ||_{F, D_t} + || theta - tilde_theta_t ||_{F, Q_t}
    # where D = diag(lambda^(t-1), ... , lambda^(t-t)) is a matrix for exponential forgetting,
    # Q is a symmetric weight matrix for the ridge penalty, and tilde_theta_t is a regularization term / prior for theta. 
    # F denotes the Frobenius norm.
    # The solution is used to predict in the model
    # Y_t = X_t theta + E_t, where E_t ~ MN(0, I, V_t).

    target = "mean"
    ensure_dim = 2

    def __init__(self, n, m, burn_in = 1, tilde_k_init_val = 0, track_memory = False, combine_variance = True, full_cov = True):
        self.tilde_K = np.eye(n) * tilde_k_init_val
        self.tilde_R = np.zeros((n,m))
        self.burn_in = burn_in
        self._n_updates = 0
        self.kappa = np.zeros((n, n))
        self.theta = np.full((n, m), np.nan)
        self.inner_var_theta = np.full((n, n), np.nan)
        self.V = np.full((m, m), np.nan)
        self.track_memory = track_memory
        if track_memory:
            self._memory = 0
            self._total_var = np.zeros((m, m))

        self.combine_variance = combine_variance
        self.n, self.m = n, m
        self._forgetting_var_state = None
        self._full_cov = full_cov

    @classmethod
    def configure(cls, burn_in = 1, tilde_k_init_val = 0, track_memory = False, combine_variance = True, full_cov = True, predictor_params = {}):
        params = {"n": DimX, "m": DimY, "burn_in": burn_in, "tilde_k_init_val": tilde_k_init_val, "track_memory": track_memory, "combine_variance": combine_variance, 
                  "full_cov": full_cov, "output_as": {"mean": "Y", "cov": "Y"}, "predictor_params": predictor_params}
        if combine_variance:
            params["outer_prod"] = ["cov"]

        return super().configure(**params)

    def online_update(self, x_i, y_i, y_i_hat, Q: np.ndarray | float = None, theta_tilde: np.ndarray = None, V: np.ndarray = None, estimate_V = True, mem = 0.99):

        if mem < 0 or mem > 1:
            raise ValueError("Memory must be between 0 and 1.")

        n, m = self.n, self.m

        if theta_tilde is None:
            theta_tilde = np.zeros((n, m))
        
        if Q is None:
            Q = np.zeros((n, n))
        elif isinstance(Q, (float, int)):
            Q = np.eye(n) * Q

        elif not np.allclose(Q, Q.T):
            raise ValueError("Q must be symmetric.")

        if theta_tilde is None:
            theta_tilde = np.zeros((n, m))

        if not V is None:
            self.V = V

        x_outer = np.outer(x_i, x_i)

        if self.tilde_K is None:
            self.tilde_K = x_outer
        else:
            self.tilde_K = mem*self.tilde_K + x_outer

        if self.tilde_R is None:
            self.tilde_R = np.outer(x_i, y_i)
        else:
            self.tilde_R = mem*self.tilde_R + np.outer(x_i, y_i)

        K = self.tilde_K + Q
        R = self.tilde_R + Q @ theta_tilde

        self.theta = np.linalg.solve(K, R)

        # Update estimate of variance
        self.kappa = mem**2*self.kappa + x_outer

        temp1 = np.linalg.solve(K, self.kappa)
    
        self.inner_var_theta = np.linalg.solve(K, temp1.T).T # K^-1 kappa K^-1^T

        if estimate_V and self._n_updates >= self.burn_in:
            resid = y_i - y_i_hat
            if self._full_cov:
                sse = np.outer(resid, resid)
                # Pad to make (1, m, m) for use with forgetting_mean
                sse = sse.reshape(1, m, m)
            else:
                sse = resid**2 
                sse = sse.reshape(1, m)
            
            _V, self._forgetting_var_state = forgetting_mean(mem, sse, self._forgetting_var_state, track_memory=self.track_memory)
            if self._full_cov:
                self.V = _V[0]
            else:
                self.V = np.diag(_V[0])

        self._n_updates += 1

    def online_predict(self, x: np.ndarray, V = None, return_var_theta = False):
        result = {}

        if V is None:
            V = self.V

        result['mean'] = x.T @ self.theta

        # Compute covariance of prediction error
        var_pred_err = self.V*(1 + x.T @ self.inner_var_theta @ x)

        if not self._full_cov:
            var_pred_err = np.diag(var_pred_err)

        result['cov'] = var_pred_err

        # Compute variance of theta
        if return_var_theta:
            result["cov_theta"] = np.kron(V, self.inner_var_theta)

        return result

    def get_var_vec_theta(self, V: np.ndarray = None):
        """
        Returns the variance of the model parameters theta.
        If V is not provided, uses the internal V.
        """
        if V is None:
            V = self.V

        # Compute the variance of theta
        var_theta = np.kron(V, self.inner_var_theta)

        return var_theta

    def get_model_params(self):
        return self.theta

class CircularBuffer:

    """
    A class for efficiently storing and retrieving the "size" most recent rows of data in a circular buffer.
    """

    # TODO: consider generalizing to n-dimensional arrays

    def __init__(self, size, m, default_value = np.nan):
        self.offset = 0
        self.size = size
        self._range = np.arange(size)
        self._index = None
        self.m = m
        self.default_value = default_value
        self.data = None

    def set_data(self, dtype = np.float64, default_value = np.nan):
        # Set data type of the buffer
        self.data = np.full((self.size, self.m), default_value, dtype=dtype)

    @property
    def index(self):
        if self._index is None:
            self._index = np.roll(self._range, -self.offset)

        return self._index

    def append(self, value: np.ndarray):
        self._index = None

        if self.data is None:
            self.set_data(dtype=value.dtype, default_value=self.default_value)

        if value.ndim == 1:
            value = value.reshape(-1, self.m)

        # Get only last size rows of value
        if value.shape[0] > self.size:
            value = value[-self.size:]

        n = value.shape[0]
        rem = self.size - self.offset
        i = self.offset + n
        self.data[self.offset:i] = value[:rem]

        self.offset = i % self.size

        if n > rem:
            self.data[:self.offset] = value[rem:n]

    def get_slice(self, start:int = None, end:int = None) -> np.ndarray:
        indices = self.index[start:end]
        if self.data is None:
            n_return = len(indices)
            return np.full((n_return, self.m), self.default_value)
        return self.data[indices]

    def get(self, n: int) -> np.ndarray:
        res = self.get_slice(end = n)
        if n > self.size:
            # Pad with default values
            pad = np.full((n - self.size, self.m), self.default_value)
            res = np.vstack([res, pad])
        elif n == 1:
            res = res.squeeze(0)
        return res

    def update(self, data: np.ndarray):
        n = data.shape[0] if data.ndim > 1 else 1
        res = self.get(n)
        if n > self.size:
            res[self.size:] = data[:n-self.size]
        self.append(data)
        if data.ndim > 1:
            # Ensure output has same shape as input
            res = res.reshape(data.shape)
        return res

    def reset(self):
        self.data.fill(np.nan)
        self._index = None
        self.offset = 0

def get_indexer(subset_index, target_index):
    # Get unique values in subset_index
    subset_index = pd.Index(subset_index).unique()
    return np.array([i for i, col in enumerate(target_index) if col in subset_index])
        
def rmse(x):
    return np.sqrt(np.mean(x**2))

class Prediction(Transformation):
    """ 
    Defines a prediction of the type (Y_{t+h}, Z_{t+h}) = f(X_t) + noise, where
    - Y is the target variable to predict
    - X is a set of predictor variables
    - Z is a set of additional predicted variables
    - h is the prediction horizon
    - f is a predictor function, which can be either an online or batch predictor.
    """
    def __init__(self, X, Y, horizon, predictor_config: PredictorConfiguration, apply_format = True):
        self.X = X
        self.Y = Y
        self.horizon = horizon
        self.update_config(predictor_config, apply_format = apply_format)

        self.X_train = Lag(X, amount=horizon)
        self.Y_hat = Lag(self, amount=horizon)

        super().__init__(X, Y, update_predictor = UpdatePredictor, predictor_params = PredictorParameters, state = Memory)

    def update_config(self, predictor_config: PredictorConfiguration, apply_format = True):
        self.config = predictor_config
        self._apply_format = apply_format

    def apply(self, data, memory=None, recursion_pars=None, return_recursion_pars=False, ref=None, update_predictor=True, eval_mode = False, copy_data = True, **predictor_params):
        if eval_mode:
            # Return self.Y shifted by horizon
            return {self.config.predictor_type.target: shift(self.Y.apply(data), -self.horizon)}, {}

        return super().apply(data, memory, recursion_pars, return_recursion_pars, ref, update_predictor, copy_data=copy_data, **predictor_params)

    def evaluate(self, X, Y, update_predictor, predictor_params, state = None, apply_format = None):
        
        # Determine whether to apply format
        apply_format = self._apply_format if apply_format is None else apply_format

        # Extract relevant predictor parameters
        params = self.config.predictor_params | predictor_params

        if state is None:
            # Create predictor
            predictor = self.config.create(X, Y)
            X_state = None
            _, Y_hat_state = self.Y_hat.evaluate(empty_like(Y), None)
            ref_data = {"X": X, "Y": Y}
            if isinstance(predictor.output_as, str):
                output_format = DataFormat(get_vars(ref_data[predictor.output_as]), outer_prod = predictor.outer_prod)
            else:
                ref_vals = {k: ref_data[v] for k, v in predictor.output_as.items()}
                output_format = DataFormat(get_vars(ref_vals), outer_prod = predictor.outer_prod)

        else:
            predictor, X_state, Y_hat_state, output_format = state

        # Store index if output format requires it
        if predictor.output_as is not None:
            index = get_index(X)

        # Convert inputs to numpy arrays?
        if predictor.ensure_dim is not None:
            X = to_numpy(X, predictor.ensure_dim)

        if update_predictor:
            
            if predictor.ensure_dim is not None:
                Y = to_numpy(Y, predictor.ensure_dim)

            # Get training data
            X_train, X_state = self.X_train.evaluate(X, X_state)

            # Get old predictions
            old_Y_hat = Y_hat_state.get(get_num_obs(Y)).reshape(Y.shape)

            result = predictor.update(X, Y, X_train, old_Y_hat, self.horizon, **params)

            # Update storage
            Y_hat = result[predictor.target or next(iter(result))]
            _, Y_hat_state = self.Y_hat.evaluate(Y_hat, Y_hat_state)

        else:
            result = predictor.predict(X, self.horizon, **params)

        # Format output?
        if predictor.output_as is not None:

            # Apply format
            if apply_format:
                result = output_format.apply(result, index = index)

        # Return result and state
        return result, (predictor, X_state, Y_hat_state, output_format)


    @classmethod
    def construct(cls, X: tuple | Source, Y: tuple | Source, predictor_configuration: PredictorConfiguration, horizon = 1, apply_format = True):
        if isinstance(X, tuple):
            X = Combine(*X)
        if isinstance(Y, tuple):
            Y = Combine(*Y)

        return cls(X, Y, horizon, predictor_configuration, apply_format = apply_format)

    def __repr__(self):
        return f"Prediction(horizon={self.horizon}, predictor={self.config.predictor_type.__name__})"


class PredictionTarget(Transformation):
    
    def __init__(self, prediction: Prediction):
        super().__init__(prediction)
        self.target = prediction.config.predictor_type.target

    def evaluate(self, prediction):
        Y_hat = prediction[self.target or next(iter(prediction))]
        return Y_hat

def make_prediction_ensemble(X: dict | tuple | Transformation, Y : tuple | Transformation, predictor_configuration: PredictorConfiguration, horizons, apply_format = True, input_horizons: dict = None):
    predictions = []

    if input_horizons is None:
        input_horizons = {h: (0, h) for h in horizons}
    
    if isinstance(Y, (tuple, list)):
        Y_sub = [GetHorizons(Y_i, drop_horizon=True) for Y_i in Y] 
        Y = Combine(*Y_sub)  
    else:
        Y = GetHorizons(Y, drop_horizon=True)

    for h, h_in in input_horizons.items():
        if isinstance(X, (tuple, list)):
            X_sub = [GetHorizons(data = X_i, horizons = h_in) for X_i in X]
            X_h = Combine(*X_sub)
        else:
            X_h = GetHorizons(data = X, horizons = h_in)

        prediction = Prediction(X_h, Y, h, predictor_configuration, apply_format = apply_format)
        predictions.append(prediction)

    return predictions

class Model:
    def __init__(self, *output: Source, scorefun = rmse, burn_in = 0, remove_nan = True):
        self.output = Combine(*output, as_dict = True)
        self.state = None
        self.data_format = None

        # Construct transforms for scoring predictions
        targets = []
        for o in output:
            if isinstance(o, Prediction):
                target = PredictionTarget(o)
                targets.append(target)
            else:
                targets.append(o)

        self.targets = Combine(*targets, as_dict = True)
        self.scorefun = scorefun
        self.burn_in = burn_in
        self.remove_nan = remove_nan

        # Fetch all Prediction dependencies in output recursively
        self.predictions = []
        def fetch_predictions(source):
            if isinstance(source, Prediction):
                self.predictions.append(source)

            if isinstance(source, Transformation):
                for dep in source.dependencies:
                    fetch_predictions(dep)

        for o in output:
            fetch_predictions(o)

    @property
    def predictor(self):
        result = {p: self.state[p][0] for p in self.predictions}
        if len(result) == 1:
            return next(iter(result.values()))
        return result

    @classmethod
    def construct(cls, X: dict | tuple | Transformation, Y : tuple | Transformation, predictor_configuration: PredictorConfiguration, horizons = 1, apply_format = True, scorefun = rmse, burn_in = 0, remove_nan = True):

        prediction = Prediction.construct(X, Y, predictor_configuration, horizons, apply_format = apply_format)

        return cls(prediction, scorefun = scorefun, burn_in = burn_in, remove_nan = remove_nan)

    @classmethod
    def construct_ensemble(cls, X: dict | tuple | Transformation, Y : tuple | Transformation, predictor_configuration: PredictorConfiguration, horizons, apply_format = True, input_horizons: dict = None, scorefun = rmse, burn_in = 0, remove_nan = True):

        predictions = make_prediction_ensemble(X, Y, predictor_configuration, horizons, apply_format = apply_format, input_horizons = input_horizons)

        return cls(*predictions, scorefun = scorefun, burn_in = burn_in, remove_nan = remove_nan)

    def reset_state(self):
        self.state = None
        self.data_format = None

    def update(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, check = False, update_predictor = True, **predictor_params):

        if self.data_format is None:
            self.data_format = DataFormat.from_reference(data)

        if check:
            self.data_format.check(data)

        # Transform data
        result, self.state = self.output.apply(data, recursion_pars = self.state, update_predictor=update_predictor, return_recursion_pars=True, **predictor_params)
        
        if len(result) == 1:
            return next(iter(result.values()))

        return result

    def fit(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, **predictor_params):
        self.reset_state()
        return self.update(data, **predictor_params)

    def fit_and_score(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, **predictor_params):
        """
        A method for optimization of model parameters.
        """

        result = self.targets.apply(data, eval_mode = False, **predictor_params)
        ref_result = self.targets.apply(data, eval_mode = True, **predictor_params)

        scores = evaluate_score(result, ref_result, burn_in = self.burn_in, remove_nan = self.remove_nan, scorefun = self.scorefun)

        if len(scores) == 1:
            return next(iter(scores.values()))

        # Rename to outputs
        scores = {o: s for o, s in zip(self.output.names, scores.values())}

        return scores
        
    def configure_prediction(self, predictor_configuration: PredictorConfiguration, apply_format = True, prediction = None):
        if prediction is None:
            prediction = self.predictions
        elif isinstance(prediction, Prediction):
            prediction = [prediction]
        for p in prediction:
            p.update_config(predictor_configuration, apply_format = apply_format)

    def save_model(self, name: str = None):
        if name is None:
            return pickle.dumps(self)
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(self, f)

    def print_model_tree(self):
        self.output.print_dependency_tree()

@data_decorator    
def evaluate_score(Y_hat, Y, burn_in = 0, remove_nan = True, scorefun = rmse):
    resid = Y - Y_hat

    if burn_in > 0:
        resid = resid[burn_in:]
    if remove_nan:
        mask = ~np.isnan(resid).any(axis=1)
        resid = resid[mask]

    return scorefun(resid)

class Ensemble(Model):

    def __init__(self, X: dict | tuple | Transformation, Y : tuple | Transformation, predictor_configuration: PredictorConfiguration, horizons, apply_format = True, input_horizons: dict = None, scorefun = rmse, burn_in = 0, remove_nan = True):
        predictions = make_prediction_ensemble(X, Y, predictor_configuration, horizons, apply_format = apply_format, input_horizons = input_horizons)
        super().__init__(*predictions, scorefun = scorefun, burn_in = burn_in, remove_nan = remove_nan)
        self.ref = predictions[0]
        self._column_names = {}
        self.X = X
        self.Y = Y

    def reset_state(self):
        super().reset_state()
        self._column_names = {}

    def update(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, check = False, update_predictor = True, combine_horizons = True, **predictor_params):
        result = super().update(data, check, update_predictor, **predictor_params)

        if combine_horizons:
            combined_results = {}
            for name in result[self.ref].keys():
                combined_results[name], self._column_names[name] = combine_data({p.horizon: result[p][name] for p in self.output.sources}, columns = self._column_names.get(name, None))

            # Remove individual horizon results
            for p in self.output.sources:
                del result[p]

            result[self] = combined_results

        if len(result) == 1:
            return next(iter(result.values()))
        
        return result

    def fit(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, combine_horizons = True, **predictor_params):
        return self.update(data, combine_horizons=combine_horizons, **predictor_params)

def load_model(file_name):
    if not file_name.endswith(".pkl"):
        file_name = file_name + ".pkl"
    with open(file_name, 'rb') as f:
        return pickle.load(f)
# %%