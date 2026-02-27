#%%
from __future__ import annotations
import numpy as np
import pandas as pd
import re
import inspect
import os
import functools
import pickle
from abc import ABC, abstractmethod
from numpy.lib.stride_tricks import sliding_window_view
from typing import TYPE_CHECKING

# Get the directory of the current module
#module_dir = os.path.dirname(__file__)

# Construct the path to the data file
#data_folder = os.path.join(module_dir, 'data')

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
    cols = pd.MultiIndex.from_product([names, horizons], names = ['Variable', 'Horizon'])
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

# TODO: use lazy loading for sample data
#if os.path.exists(data_folder) and 'simulated_data.csv' in os.listdir(data_folder):
#    sample_data = read_forecast_csv(data_folder + '/simulated_data.csv', horizon_pattern=".k",parse_dates=['t'], index_col = "t")
    
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

    def __getitem__(self, key):
        return GetItemTransformation(self, key)

    def get_attr(self, name):
        return GetAttrTransformation(self, name)

# NOTE: a special class and __reduce__ method is required for keywords for serialization with pickle
class KeywordSource(Source):
    _registry = {}
    def __reduce__(self):
        return make_keyword, (self._name,)

def make_keyword(name: str) -> Source:
    if name not in KeywordSource._registry:
        KeywordSource._registry[name] = KeywordSource(name)
    return KeywordSource._registry[name]
    
# TODO: use CAPITALIZED names for keywords
MEMORY = make_keyword("MEMORY")
DEFAULT_SOURCE = make_keyword("DEFAULT_SOURCE")
DEFAULT_INDEX = make_keyword("DEFAULT_INDEX")
UPDATE_PREDICTOR = make_keyword("UPDATE_PREDICTOR")
X_INIT = make_keyword("X_INIT")
Y_INIT = make_keyword("Y_INIT")
Z_INIT = make_keyword("Z_INIT")

class Transformation(Source):

    """
    Generic class for transformations. Subclasses should implement the evaluate method.

    The class provides functionality for matching placeholder data sources to data, and evaluating
    the transformation based on the provided data. Subclasses should call super().__init__(*args, **kwargs)
    with keyword names matching the evaluate method parameters to specify how data should be passed.   
    Also enables basic operations between transformations, such as +, -, *, etc.
    """

    # TODO: consider making an "online" flag, and an "apply_online" method
    # that retrieves the most recent inpy data only, and iteratively updates
    # outputs. Transforms with online flags should be evaluated using apply_online,
    # whilst other transforms can be evaluated normally. The data
    # dict should be updated incrementally for transforms with the online flag.

    # Superclass for transformations
    def __init__(self, *apply_args, **apply_kwargs):

        """
        apply_args: optional positional arguments to specify input data targets for evaluate when called via. apply.
        apply_kwargs: optional dict to specify input data targets for evaluate when called via. apply.
        """
    
        # Check for unused params and whether there is any dependency on Prediction
        self.apply_kwargs = apply_kwargs
        self.apply_args = apply_args
        for key in apply_kwargs:
            if not (key in self._inputs or self._accepts_kwargs):
                raise KeyError(f"{self} does not use input: {key}.")
            if self.apply_kwargs[key] is None:
                self.apply_kwargs[key] = DEFAULT_SOURCE

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

        # Initialise memory state 
        self.recursion_pars = None

        # Determine free parameters (keyword inputs that are not in apply_kwargs)
        self._free_params = [p for p in self._eval_sig.parameters if p not in self.apply_kwargs and p != "self"]

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

    def apply(self, data = None, memory = None, recursion_pars = None, return_recursion_pars = False, ref = None, update_predictor = True, copy_data = True, track_state = False, **params):

        # NOTE: copy_data is used to avoid modifying input data (unless requested). When applied recursively, copy_data should be False, as we do want to update the data with intermediate results.
        data = parse_data(data, ref = ref, copy = copy_data)

        evaluate_kwargs = {}
        evaluate_args = []

        if recursion_pars is None:
            if track_state:
                recursion_pars = self.recursion_pars or {}
            else:
                recursion_pars = {}

        new_recursion_pars = {}

        targeted_params = params.get(self, {})

        # Load memory from provided recursion pars
        if memory is None:
            memory = recursion_pars.get(self, None)

        # Check inputs
        for name, val in self._apply_pairs:

            # Already in data?
            if val in data:
                t_val = data[val]

            elif val is MEMORY:
                # TODO: consider fetching default from evaluate signature
                t_val = memory

            elif val is UPDATE_PREDICTOR:
                t_val = update_predictor
            
            elif val in targeted_params:
                t_val = targeted_params[val]
            
            elif val in params:
                t_val = params[val]

            # Attempt to fetch transformation dependencies if not provided directly.
            elif isinstance(val, Transformation):
                t_val, t_rec_pars = val.apply(data = data, recursion_pars = recursion_pars, return_recursion_pars = True, copy_data=False, **params)

                new_recursion_pars.update(t_rec_pars)

                data[val] = t_val # Store in data for potential reuse

            else:
                raise ValueError(f"Missing data for input: {val} in {self}.")

            # Store transformed value in evaluate args
            if name is None:
                evaluate_args.append(t_val)
            else:
                evaluate_kwargs[name] = t_val

        # Check for free parameters
        for p in self._free_params:
            if p in targeted_params:
                evaluate_kwargs[p] = targeted_params[p]
            elif p in params:
                evaluate_kwargs[p] = params[p]
        
        # If evaluate accepts var kwargs, pass all params not already used for specific inputs
        if self._accepts_var_kwargs:
            # Update with all kwargs that do not have Source keys
            evaluate_kwargs = evaluate_kwargs | {k: v for k, v in params.items() if not isinstance(v, Source)}

        # Evaluate
        eval_out = self.evaluate(*evaluate_args, **evaluate_kwargs)

        if isinstance(eval_out, tuple):
            result, memory = eval_out
        else:
            result = eval_out
            memory = None

        new_recursion_pars[self] = memory

        # Store recursion pars if tracking enabled
        if track_state:
            self.recursion_pars = new_recursion_pars.copy()

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
            elif dep not in [MEMORY, UPDATE_PREDICTOR]:
                result.add(dep)
        return list(result)

    def evaluate(self) -> tuple[pd.DataFrame, dict] | pd.DataFrame:
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
    
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

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

# TODO: remove DEFAULT_INDEX from data parsing.
def parse_data(data : dict | pd.DataFrame | pd.Series | np.ndarray, ref = None, copy = True):
    if not isinstance(data, dict):
        data = {DEFAULT_SOURCE: data}
    elif copy:
        data = data.copy()
    
    ref_val = data[ref or next(iter(data))]
 
    if not DEFAULT_SOURCE in data:
        data[DEFAULT_SOURCE] = ref_val

    data[DEFAULT_INDEX] = get_index(ref_val)

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

class Dim(Transformation):
    
    def __init__(self, data, axis = 1):
        self.axis = axis
        super().__init__(data = data)

    def evaluate(self, data):
        return get_dim(data, axis = self.axis)
    
DIM_X = Dim(X_INIT)
DIM_Y = Dim(Y_INIT)
DIM_Z = Dim(Z_INIT)
        
class GetItemTransformation(Transformation):

    def __init__(self, data, key):
        super().__init__(data = data)
        self.key = key

    def evaluate(self, data):
        return data[self.key]

class GetAttrTransformation(Transformation):
    
    def __init__(self, data, attr):
        super().__init__(data = data)
        self.attr = attr

    def evaluate(self, data):
        return getattr(data, self.attr)

class ApplyFunctionTransformation(Transformation):

    def __init__(self, func, *args, **kwargs):
        self.func = func
        # Get args and kwargs that have Source values
        self.fixed_args = {i: arg for i, arg in enumerate(args) if not isinstance(arg, Source)}
        self.source_args = {i: arg for i, arg in enumerate(args) if isinstance(arg, Source)}
        self.n_args = len(args)
        self.fixed_kwargs = {k: arg for k, arg in kwargs.items() if not isinstance(arg, Source)}
        self.source_kwargs = {k: arg for k, arg in kwargs.items() if isinstance(arg, Source)}
        super().__init__(*self.source_args.values(), **self.source_kwargs)

    def evaluate(self, *args, **kwargs):
        # Combine fixed and source args and kwargs

        # Build args list by sorting the combined keys of fixed and source args
        all_args = [self.fixed_args[i] if i in self.fixed_args else args[i] for i in range(self.n_args)]

        all_kwargs = kwargs | self.fixed_kwargs

        return self.func(*all_args, **all_kwargs)

def function_wrapper(func):
    """
    Decorator to wrap functions on transformations, to return a new transformation with the function applied to the output of the original transformation.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return ApplyFunctionTransformation(func, self, *args, **kwargs)
    return wrapper

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

class Param(Transformation):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self):
        return self.value

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