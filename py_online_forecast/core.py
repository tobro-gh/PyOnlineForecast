#%%
from __future__ import annotations
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

def create_fc_columns(variables: list | tuple, horizons: list | tuple, group_by_horizon: bool = False) -> pd.MultiIndex:
    if group_by_horizon:
        # Sort according to (var1, h1), (var2, h1), ..., (var1, h2), (var2, h2), ...
        tuples = [(var, h) for h in horizons for var in variables]
        return pd.MultiIndex.from_tuples(tuples, names = ['Variable', 'Horizon'])
    else:
        return pd.MultiIndex.from_product([variables, horizons], names = ['Variable', 'Horizon'])

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

    def __init__(self, name = None, can_be_shared = True):
        self._name = name
        self.can_be_shared = can_be_shared

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

# Placeholder keywords for specifying data sources
Memory = Source("Memory")
DefaultSource = Source("DefaultSource")
DefaultIndex = Source("DefaultIndex")
Prediction = Source("Prediction", can_be_shared=False)
Target = Source("Target", can_be_shared=False)
PredictionHorizon = Source("PredictorHorizon", can_be_shared=False)
DimX = Source("DimX", can_be_shared=False)
DimY = Source("DimY", can_be_shared=False)

def parse_data(data : dict | pd.DataFrame | pd.Series | np.ndarray, ref = None, check = None):

    # If already Data, do not change
    if isinstance(data, Data):
        return data

    return Data(data, ref, check)

def format_array(data: np.ndarray | pd.Series | pd.DataFrame, index, vars):
    if not isinstance(data, np.ndarray):
        data = data.to_numpy()
    if isinstance(index, np.ndarray):
        index = pd.Index(index)

    if isinstance(index, pd.Index):
        if data.ndim == 2:
            return pd.DataFrame(data, index = index, columns = vars)
        else:
            reshaped = data.reshape((data.shape[0], -1))
            if len(vars) != reshaped.shape[1]:
                vars = [f"{var}_{i}" for var in vars for i in range(reshaped.shape[1] // len(vars))]
            return pd.DataFrame(reshaped, index = index, columns = vars)
    else:
        if data.ndim == 1:
            return pd.Series(data, name = index, index = vars)
        else:
            return pd.Series([data], name = index, index = vars)


def get_vars(val):
    if isinstance(val, pd.Series):
        return val.index
    elif isinstance(val, pd.DataFrame):
        return val.columns
    else:
        return None

class Data(dict):

    """
    Class to track data and related metadata. 1D arrays and series are considered singleton multivariate data points, whilst 2D arrays and dataframes are considered multivariate time series.
    """

    def __init__(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, ref = None, check = True):        
        # TODO: use more info from old_data if available
        if isinstance(data, dict):
            if ref is None:
                ref = next(iter(data))
        else:
            ref = ref or DefaultSource
            data = {ref: data}

        if not ref is DefaultSource:
            # Point DefaultSource to same as ref
            data[DefaultSource] = data[ref]

        val = data[ref]
        data[DefaultIndex] = Index._evaluate(val)

        if val.ndim == 1:
            self.n = 1
        else:
            self.n = val.shape[0]

        self._ref = ref
        super().__init__(data)
        if check:
            self.check_data()

    def check_data(self):

        # Check that reference is of correct type
        if not isinstance(self[self._ref], (np.ndarray, pd.Series, pd.DataFrame)):
            raise ValueError(f"Reference data must be either ndarray, Series or DataFrame, got {type(self[self._ref])}")

        # Check that all data sources are valid
        for key in self:
            if not isinstance(key, Source):
                raise ValueError(f"All data keys should be of type Source, got {type(key)}.")

    def vars(self, ref = None):
        val = self[ref or self._ref]
        return get_vars(val)

    def m(self, ref = None):
        val = self[ref or self._ref]
        if val is None:
            raise ValueError(f"Data for {ref} is None.")
        if isinstance(val, np.ndarray):
            return val.shape[0] if val.ndim == 1 else val.shape[1]
        elif isinstance(val, pd.Series):
            return len(val)
        else: # DataFrame
            return val.shape[1]

    def get_element(self, i):
        data_i = {}
        for key, value in self.items():
            if isinstance(key, Index):
                data_i[key] = value[i]
            if isinstance(value, pd.DataFrame):
                data_i[key] = value.iloc[i]
            elif isinstance(value, np.ndarray):
                data_i[key] = np.atleast_1d(value[i])
            else:
                data_i[key] = value

        return data_i

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
            self._apply_kwargs = apply_kwargs
            self._apply_args = apply_args
            for key in apply_kwargs:
                if not (key in self._inputs or self._accepts_kwargs):
                    raise KeyError(f"{self} does not use input: {key}.")
                if self._apply_kwargs[key] is None:
                    self._apply_kwargs[key] = DefaultSource

            # Check that args and kwargs refer to valid inputs
            for val in list(apply_kwargs.values()) + list(apply_args):
                if not isinstance(val, Source):
                    raise ValueError(f"Input {val} must be a Source instance: {self}.")

            # Build pairs of (name, value) for args and kwargs combined
            self._apply_pairs = [(None, val) for val in apply_args] + list(apply_kwargs.items())

            # Determine all sources
            self.sources = list(self._apply_kwargs.values()) + list(self._apply_args)

            self.dependencies = [v for v in self.sources if isinstance(v, Transformation)]

            n = len(self.sources)

            # Determine whether transformation depends on Prediction
            self.depends_on_prediction = Prediction in self.sources
            self.depends_on_horizon = PredictionHorizon in self.sources
            i = 0
            while not self.depends_on_prediction and i < n:
                if isinstance(self.sources[i], Transformation) and self.sources[i].depends_on_prediction:
                    self.depends_on_prediction = True
                i += 1

            # Determine whether transformation depends on Target
            self.depends_on_target = Target in self.sources
            i = 0
            while not self.depends_on_target and i < n:
                if isinstance(self.sources[i], Transformation) and self.sources[i].depends_on_target:
                    self.depends_on_target = True
                i += 1

            self.can_be_shared = not (self.depends_on_prediction or self.depends_on_target or self.depends_on_horizon)

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

    def __repr__(self):
#        return f"{self.__class__.__name__}({self._apply_args}, {self._apply_kwargs})"
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

    def apply(self, data, memory = None, recursion_pars = None, return_recursion_pars = False, check_output = False, ref = None):

        data = parse_data(data, check = False, ref = ref)

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

            # Attempt to fetch transformation dependencies if not provided directly.
            elif isinstance(val, Transformation):
                t_val, t_rec_pars = val.apply(data = data, recursion_pars = recursion_pars, return_recursion_pars = True, check_output = check_output)

                new_recursion_pars.update(t_rec_pars)

                data = data | {val: t_val} # Note, creates new data dict to avoid modifying input data

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
    
        if check_output:
            # Clean up result if needed
            if not isinstance(result, pd.DataFrame):
                try:
                    result = new_fc(result)
                except:
                    raise ValueError(f"Could not convert output to DataFrame: {self}.")
                
            # Check that result adheres to fc format
            if not result.fc.check():
                result = result.fc.convert()

        if return_recursion_pars:
            return result, new_recursion_pars
        else:
            return result

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

class Transformer:

    def __init__(self, *sources, max_dependency_depth = 100):
        # TODO: Check that sources are valid (subclasses of Source?)
        self.sources = sources
        if len(sources) == 0:
            self.sources = (DefaultSource,)
        self.max_dependency_depth = max_dependency_depth
        self.transforms = []
        self.sorted_transforms = None

    def add_transforms(self, *transforms):
        for transform in transforms:

            if transform in self.transforms:
                raise ValueError(f"Transform {transform} already added.")
            else:
                self.transforms.append(transform)

    def set_transforms(self):

        # Get all transforms
        current_transforms = self.transforms.copy()
        check_list = current_transforms
        deps_to_include = set()

        i = 0
        while len(check_list) > 0:
            deps = []
            for t in check_list:
                if not t in self.sources:
                    deps.extend(t.dependencies)
            check_list = [t for t in deps if t not in current_transforms]
            deps_to_include.update(deps)
            i += 1
            if i > self.max_dependency_depth:
                raise ValueError("Dependency depth exceeded. Check for circular dependencies or increase max_dependency_depth.")

        all_transforms = current_transforms.copy()

        # Add missing dependencies to all_transforms
        for t in deps_to_include:
            if t not in all_transforms:
                all_transforms.append(t)

        # Sort transforms
        to_sort = all_transforms.copy()
        self.sorted_transforms = {}
        i = 0
        while len(to_sort) > 0:
            for t in to_sort:
                
                if t in self.sources or all(d in self.sorted_transforms for d in t.dependencies): # Does transforms have all dependencies available, either throug transforms or sources?
                    self.sorted_transforms[t] = None
                    to_sort.remove(t)
            i += 1
            if i > self.max_dependency_depth:
                raise ValueError("Dependency depth exceeded. Check for circular dependencies or increase max_dependency_depth.")

    def transform(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, ref = None, check = True):

        # Initialize if not done
        if self.sorted_transforms is None:
            self.set_transforms()

        data = parse_data(data, ref, check)

        # Check all promised sources are available
        if not all(s in data for s in self.sources):
            missing = [s for s in self.sources if not s in data]
            raise ValueError(f"Missing data for sources: {missing}.")

        for t, memory in self.sorted_transforms.items():

            if not t in data: # Skip if data was provided as input

                # Apply the transform
                new_data, rec_pars = t.apply(data = data, memory = memory, return_recursion_pars=True)

                # Store transformed data and memory (returned in rec_pars dict)
                self.sorted_transforms[t] = rec_pars.get(t, None)

                data[t] = new_data

        return data

    def reset_state(self):
        self.sorted_transforms = None

# TODO: use the format_warpper in place of manual conversion in multiple locations
def format_wrapper(*input_dfs: str, output_as = None, product_vars = False):
    """
    Decorator factory, for making decorators that convert input dataframes to numpy nd.array, and the output to a dataframe.
    If product_vars is True, the output columns are formed as the product of input variable names.
    """
    def decorator(func):
        sig = inspect.signature(func)
        # Check that input_dfs match signature
        for name in input_dfs:
            if not name in sig.parameters:
                raise ValueError(f"Input name {name} not found in signature.")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            bound_args = sig.bind(*args, **kwargs)

            # Get reference for formatting output
            if output_as is not None:
                value = bound_args.arguments[output_as]

                if isinstance(value, pd.DataFrame):
                    ref = [value.index, value.columns]
                elif isinstance(value, pd.Series):
                    ref = [value.name, value.index]
                elif isinstance(value, np.ndarray):
                    ref = None
                else:
                    raise ValueError(f"Reference output should be DataFrame or Series but got {type(value)}")

                ndim = value.ndim

                if product_vars and ref is not None:
                    # Form output variable names as product of input variable names
                    vars = ref[1].unique().tolist()
                    ran = range(len(vars))
                    flat_vars = [(vars[i], vars[j]) for i in ran for j in ran]
                    ref[1] = flat_vars

            # Convert inputs to arrays
            for name in input_dfs:
                if name in bound_args.arguments:
                    value = bound_args.arguments[name]
                    if isinstance(value, (pd.DataFrame, pd.Series)):
                        bound_args.arguments[name] = np.atleast_2d(value.to_numpy())
                    elif isinstance(value, (np.ndarray,)):
                        bound_args.arguments[name] = np.atleast_2d(value)

            result, memory = func(*bound_args.args, **bound_args.kwargs)

            # Format output as reference
            if output_as is not None:

                if ref is None and not isinstance(result, np.ndarray):
                    result = result.to_numpy()

                if ndim == 1:
                    result = result.squeeze(0)

                if ref is not None:
                    result = format_array(result, *ref)

            return result, memory
        
        return wrapper
    
    return decorator


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
        self.m = max(j for i, j in shifts.keys()) + 1 # Number of inputs in data

        if self.n == 0:
            raise ValueError("Shifts cannot be empty")


        self.max_shifts = {i: max(s for (ii, j), s in shifts.items() if ii == i) for i in range(self.n)}
        self.max_shift = max(self.max_shifts.values())
        self.shifts = shifts
        self.skip_duplicates = skip_duplicates
        self.initial_value = initial_value
        super().__init__(data = data, memory = Memory)

    def evaluate(self, data, memory = None):
        ndim = data.ndim
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.to_numpy()

        # Fetch data from memory
        # TODO: use CircularBuffer for efficiency
        if memory is None:
            old_data = np.full((self.max_shift, self.m), self.initial_value)
            if self.skip_duplicates:
                offset = self.max_shift
        else:
            old_data, offset = memory

        data = np.atleast_2d(data)

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
            mask = np.arange(offset, t, self.max_shift + 1)
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

        if ndim == 1:
            X = X.squeeze(0)

        if self.skip_duplicates:
            return X, (all_data[-self.max_shift:], offset)
        else:
            return X, (all_data[-self.max_shift:], None)


class Index(Transformation):

    def __init__(self, data = DefaultSource):
        super().__init__(data = data, i = Memory)
    
    @classmethod
    def _evaluate(cls, data, i = 0):
        if isinstance(data, pd.DataFrame):
            return data.index
        elif isinstance(data, pd.Series):
            return data.name
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return i
            else:
                n = i + data.shape[0]
                return np.arange(i, n)
        else:
            raise ValueError(f"Cannot extract index from {type(data)}.")

    def evaluate(self, data, i = 0):
        return self._evaluate(data, i)

class Horizons(Transformation):
    
    def __init__(self, data = DefaultSource):
        super().__init__(data = data)

    def evaluate(self, data):
        if isinstance(data, pd.DataFrame):
            vars = data.columns
        elif isinstance(data, pd.Series):
            vars = data.index
        else:
            raise ValueError(f"Cannot extract horizons from {type(data)}.")
        if isinstance(vars, pd.MultiIndex):
            return vars.get_level_values(1).unique().tolist()
        else:
            raise ValueError("Data does not have MultiIndex columns.")

class Dim(Transformation):
    
    def __init__(self, data):
        super().__init__(data = data)

    def evaluate(self, data):
        if isinstance(data, dict):
            if "X_pred" in data:
                data = data["X_pred"]
            else:
                raise ValueError("Data dict does not contain 'X_pred' key.")

        if isinstance(data, pd.Series):
            return data.shape[0]
        elif isinstance(data, pd.DataFrame):
            return data.shape[1]
        elif isinstance(data, np.ndarray):
            return data.shape[1] if data.ndim == 2 else data.shape[0]
        else:
            raise ValueError(f"Cannot extract dimension from {type(data)}.")

class Length(Transformation):
    
    def __init__(self, data):
        super().__init__(data = data)

    def evaluate(self, data):
        if isinstance(data, pd.Series):
            return 1
        elif isinstance(data, pd.DataFrame):
            return data.shape[0]
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return 1
            else:
                return data.shape[0]
        else:
            raise ValueError(f"Cannot extract length from {type(data)}.")

class One(Transformation):
    
    def __init__(self, index = DefaultIndex):
        super().__init__(index = index)

    def evaluate(self, index):
        if isinstance(index, (pd.Index, np.ndarray)):
            return np.ones((len(index), 1))
        else:
            return np.ones(1)
        
class Residuals(Transformation):

    def __init__(self, blank_value = 0):
        self.blank_value = blank_value
        super().__init__(targets = Target, predictions = Prediction)

    def evaluate(self, targets, predictions):
        resid = targets - predictions

        # Replace NaNs with blank_value
        resid = resid.astype(float).fillna(self.blank_value)

        return resid

class LowPass(Transformation):
    def __init__(self, var, alpha = 0):
        super().__init__(data=var, prev_value = Memory)
        self.alpha = alpha

    def evaluate(self, data: pd.DataFrame | pd.Series, prev_value=None):
        alpha = self.alpha
        y = data.to_numpy(copy=True)
        y = np.atleast_2d(y)
        n, m = y.shape

        new_vals = np.full((n + 1, m), np.nan)
        if prev_value is not None:
            new_vals[0] = prev_value

        for i in range(len(y)):
            new_vals[i+1] = alpha* new_vals[i] + (1 - alpha) * y[i]

            # If any NaNs, replace them with new values
            new_vals[i+1] = np.where(np.isnan(new_vals[i+1]), y[i], new_vals[i+1])

            # If any NaNs still present, replace with prevous values
            new_vals[i+1] = np.where(np.isnan(new_vals[i+1]), new_vals[i-1], new_vals[i+1])

        if isinstance(data, pd.Series):
            result = pd.Series(new_vals[1:].squeeze(), index=data.index, name=data.name)
        else:
            result = pd.DataFrame(new_vals[1:], index=data.index, columns=data.columns)
        return result, new_vals[-1]


class FourierSeries(Transformation):

    def __init__(self, data, nharmonics = 1):
        self.nharmonics = nharmonics
        super().__init__(data = data)

    def evaluate(self, data):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.to_numpy()
            
        ndim = data.ndim
        data = np.atleast_2d(data)

        results = []
        for i in range(1, self.nharmonics + 1):
            results.append(np.cos(2*np.pi*i*data))
            results.append(np.sin(2*np.pi*i*data))
    
        result = np.hstack(results)

        if ndim == 1:
            result = result.squeeze(0)

        return result

class SlidingSum(Transformation):

    def __init__(self, data = DefaultSource, window_size = 1, *args, **kwargs):
        self.window_size = window_size
        super().__init__(data = data, old_data = Memory)
        self._args = args
        self._kwargs = kwargs

    @format_wrapper("data", output_as = "data")
    def evaluate(self, data, old_data = None):
        if old_data is None:
            n_vars = data.shape[1] if data.ndim == 2 else data.shape[0]
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

    @format_wrapper("data", output_as = "data")
    def evaluate(self, data, old_data = None):
        if old_data is None:
            n_vars = data.shape[1] if data.ndim == 2 else data.shape[0]
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

    result = np.full_like(clean_data, np.nan)

    if track_memory:
        # Old estimate is unnormalized sum
        w_sum = old_est

        for i in range(n):
            w_sum = forgetting * w_sum + data[i]
            memory = memory * forgetting + 1
            result[i] = w_sum / memory

        # Store data for next iteration
        new_est = w_sum

    else:
        # Assume saturated memory and update mean estimate directly (old_est is mean)
        result[0] = forgetting * old_est + (1 - forgetting) * data[0]
        for i in range(1, n):
            result[i] = forgetting * result[i-1] + (1 - forgetting) * data[i]

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

    @format_wrapper("data", output_as = "data")
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

        if covariance:
            self.evaluate = format_wrapper("data","mean", output_as = "data", product_vars = True)(self.evaluate)
        else:
            self.evaluate = format_wrapper("data","mean", output_as = "data")(self.evaluate)

        self.forgetting = forgetting
        self.track_memory = track_memory
        self.covariance = covariance

    def evaluate(self, data, mean = None, state = None):
        
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
            columns = create_fc_columns(variables, self.horizons, group_by_horizon=True)
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

    def __init__(self, hour, dayofweek = None, duration = None):
        super().__init__(index = DefaultIndex, horizon = PredictionHorizon)
        self.hour = hour
        self.dayofweek = dayofweek
        self.duration = duration
    
    def evaluate(self, index, horizon):
        start_time = index + pd.Timedelta(hours=horizon)
        if self.duration is not None:
            end_time = start_time + pd.Timedelta(hours=self.duration)
            cond = (index >= start_time) & (index <= end_time)
        else:
            cond = (start_time.hour == self.hour)
        if self.dayofweek is not None:
            cond &= (start_time.dayofweek == self.dayofweek)

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

    def __init__(self, t = DefaultIndex, as_2d = False):
        super().__init__(t = t)
        self.as_2d = as_2d

    def evaluate(self, t):
        if isinstance(t, pd.DataFrame):
            t = t.iloc[:, 0]
        elif isinstance(t, pd.DatetimeIndex):
            t = pd.Series(t)
        elif isinstance(t, pd.Timestamp):
            t = pd.Series([t])
        else:
            raise ValueError("Input t must be a pd.DatetimeIndex or pd.Timestamp.")
        delta = t - t.dt.floor("D")
        seconds = delta.dt.total_seconds()
        time_of_day_float = seconds / 86400
        time_of_day_float = time_of_day_float.to_numpy()
        if self.as_2d:
            time_of_day_float = time_of_day_float.reshape(-1, 1)
        return time_of_day_float

class TimeOfYear(Transformation):

    def __init__(self, t=DefaultIndex, as_2d = False):
        super().__init__(t=t)
        self.as_2d = as_2d

    def evaluate(self, t):
        if isinstance(t, pd.DataFrame):
            t = t.iloc[:, 0]
        elif isinstance(t, pd.DatetimeIndex):
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

        if self.as_2d:
            result = result.reshape(-1, 1)

        return result

class TimeOfWeek(TimeOfDay):

    def __init__(self, t=DefaultIndex, as_2d = False):
        super().__init__(t=t, as_2d=False)
        self._week_as_2d = as_2d

    def evaluate(self, t):
        tod = super().evaluate(t)
        if isinstance(t, pd.DataFrame):
            t = t.iloc[:, 0]
        elif isinstance(t, pd.DatetimeIndex):
            t = pd.Series(t)
        elif isinstance(t, pd.Timestamp):
            t = pd.Series([t])
        else:
            raise ValueError("Input t must be a pd.DatetimeIndex or pd.Timestamp.")

        tow = (t.dt.dayofweek + tod) / 7
        if self._week_as_2d:
            tow = tow.to_numpy().reshape(-1, 1)
        else:
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

class HorizonTarget(Transformation):
    """
    Targets a specific horizon in the input data and removes the (empty) level if present.
    """
    
    def __init__(self, data):
        super().__init__(data = data, mem = Memory)

    def evaluate(self, data, mem = None):
  
        if isinstance(data, pd.Series):
            old_cols = data.index
        elif isinstance(data, pd.DataFrame):
            old_cols = data.columns
        else:
            return data
  
        if not isinstance(old_cols, pd.MultiIndex):
            return data
  
        if mem is None:
            cols = None
            indices = None
        else:
            cols = mem.get("cols", None)
            indices = mem.get("indices", None)
  
        # Subset columns for horizon.
        if indices is None:
            indices = subset_columns(old_cols, horizons = (0,), return_index = True)

        if cols is None:
            cols = [var[0] for var in old_cols[indices]]

        # Remove second level of MultiIndex if present
        if isinstance(data, pd.DataFrame):
            subset = data.iloc[:, indices]
            subset.columns = cols
        elif isinstance(data, pd.Series):
            subset = data.iloc[indices]
            subset.index = cols

        return subset, {"cols": cols, "indices": indices}

class Map(Transformation):

    def __init__(self, *vars, data = DefaultSource):
        super().__init__(data = data)
        self.vars = list(vars)
    
    def evaluate(self, data):
        return data[self.vars]

    def __repr__(self):
        return super().__repr__() + f"({self.vars})"

class SelectColumns(Transformation):

    def __init__(self, indices, data = DefaultSource):
        super().__init__(data = data)
        self.indices = list(indices)
    
    def evaluate(self, data):
        if isinstance(data, pd.Series):
            return data.iloc[self.indices]
        elif isinstance(data, pd.DataFrame):
            return data.iloc[:, self.indices]
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return data[self.indices]
            else:
                return data[:, self.indices]
        else:
            raise ValueError(f"Cannot select columns from {type(data)}.")


class Combine(Transformation):
    def __init__(self, *sources, format_result = None, index = DefaultIndex):
        super().__init__(*sources, index = index, memory = Memory)
        self.format_result = format_result

    def evaluate(self, *data, index = None, memory = None):
        
        converted_data = {}
        data_vars = {}
        for source, d in zip(self._apply_args, data):
            if isinstance(d, pd.Series):
                d = d.to_numpy()
            elif isinstance(d, pd.DataFrame):
                d = d.to_numpy()

            if d is not None:
                converted_data[source] = d

        ndim = next(iter(converted_data.values())).ndim

        # Stack arrays horizontally
        result = np.hstack([np.atleast_2d(d) for d in converted_data.values()])

        # Format result
        if ndim == 1:
            result = result.squeeze(0)

        # Check memory for formatting options
        if memory is None:
            # Check if formatting should be applied
            apply_format = self.format_result or (self.format_result is None and any(isinstance(d, (pd.Series, pd.DataFrame)) for d in data))
            columns = None
        else:
            apply_format, columns = memory

        if apply_format:
            # Read format
            if columns is None:
                for source, d in zip(self._apply_args, data):
                    if isinstance(d, pd.Series):
                        data_vars[source] = d.index
                    elif isinstance(d, pd.DataFrame):
                        data_vars[source] = d.columns
                    else:
                        m_d = d.shape[1] if d.ndim > 1 else d.shape[0]
                        data_vars[source] = [f"{source.__class__.__name__}_{i}" for i in range(m_d)]

            # Create tuples and MultiIndex
            tuples = []
            for dvars in data_vars.values():
                if isinstance(dvars, pd.MultiIndex):
                    # Check that the multiindex has exactly two levels
                    if dvars.nlevels != 2:
                        raise ValueError("MultiIndex must have exactly two levels.")
                    tuples.extend(dvars.tolist())
                else:
                    tuples.extend([(var, 0) for var in dvars])

                columns = pd.MultiIndex.from_tuples(tuples, names = ["Variable", "Horizon"])

            result = format_array(result, index, columns)

        return result, (apply_format, columns)

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

    def __init__(self, *variables, data = None, horizons = None, indices = None):
        super().__init__(data = data, indices = Memory)
        self.horizons = horizons
        self.variables = list(variables)

    def evaluate(self, data, indices = None):
        # Use fc.subset_columns to get the subset of columns
        if isinstance(data, pd.Series):
            columns = data.index
        elif isinstance(data, pd.DataFrame):
            columns = data.columns
        else:
            return data, None

        if indices is None:
            indices = subset_columns(columns, *self.variables, horizons = self.horizons, return_index = True)

        if isinstance(data, pd.Series):
            return data.iloc[indices], indices
        else:
            return data.iloc[:, indices], indices

    def __repr__(self):
        return super().__repr__() + f"({self.variables}, horizons={self.horizons})"


class Lag(Transformation):

    def __init__(self, data, amount = 1, default_value = None):
        super().__init__(data = data, prev_values = Memory)
        self.amount = amount
        self.fill_value = float("nan") if default_value is None else default_value

    @format_wrapper("data", output_as = "data")
    def evaluate(self, data, prev_values = None):

        if prev_values is None:
            # Get shape of data to initialize buffer
            m = data.shape[1]
            prev_values = CircularBuffer(self.amount, m, default_value = self.fill_value)

        result = prev_values.update(data)

        return result, prev_values

class RelLag(Transformation):

    def __init__(self, data, amount=1):
        self.amount = amount
        super().__init__(data=data, horizon = PredictionHorizon, memory = Memory)

    def evaluate(self, data, horizon, memory):
        shift = self.amount - horizon
        if shift < 0:
            return None

        t = None
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if isinstance(data, pd.Series):
                t = data.name
            else:
                t = data.index
            data = data.to_numpy()

        data = np.atleast_2d(data)
        n = data.shape[0]
        
        if memory is None:
            
            prev_values = np.full((shift, data.shape[1]), np.nan)

            # Construct columns
            if isinstance(data, pd.Series):
                names = data.index.to_list()
            elif isinstance(data, pd.DataFrame):
                names = data.columns.to_list()
            else:
                names = [f"var_{i}" for i in range(data.shape[1])]
            tuples = [(c, shift) for c in names]
            columns = pd.MultiIndex.from_tuples(tuples, names=["Variable", "Horizon"])
        else:
            prev_values, columns = memory

        y = np.vstack((prev_values, data))

        if shift == 0:
            result = y[-n:, :]
        else:
            result = y[-(n + shift):-shift, :]

        # Set column names if provided
        if n == 1:
            result = result.squeeze(0)
        result = format_array(result, t, columns)

        return result, (y[-shift:], columns)


def nested_concat(containers, across: Literal["rows", "columns"] = "rows") -> list:
    """
    Concatenate multiple nested containers elementwise along a specified axis. Elements not recognized as DataFrames, Series will be collected in a list.
    The input should be any number of lists or tuples, containing pandas DataFrames or Series or numpy arrays.
    The function concatenates the nested elements across the containers using the specified axis.
    The function returns a single list of the concatenated elements.
    """

    nested_lens = set([len(p) for p in containers])
    if len(nested_lens) > 1:
        raise ValueError(f"All containers must have the same length, got: {nested_lens}")

    result = []

    for i in range(nested_lens.pop()):    
        data_types = set(type(c[i]) for c in containers)
        if len(data_types) > 1:
            raise TypeError(f"All containers must contain elements of the same type, got: {data_types}")
        data_type = data_types.pop()
        
        if data_type is pd.DataFrame:
            axis = 0 if across == "rows" else 1
            result.append(pd.concat([c[i] for c in containers], axis=axis))
        elif data_type is pd.Series:
            if across == "rows":
                result.append(pd.concat([c[i] for c in containers], axis=1).T)
            else:
                result.append(pd.concat([c[i] for c in containers], axis=0))
#        elif data_type is np.ndarray:
#            axis = 0 if across == "rows" else 1
#            result.append(np.concatenate([c[i] for c in containers], axis=axis))
        else:
            # If the type is not recognized, collect in a list
            result.append([c[i] for c in containers])

    return result


class InputTransform(Transformation):

    def __init__(self, horizon, X, to_numpy = True):
        if isinstance(X, dict):
            self.return_as_dict = True
        else:
            X = {"X": X}
            self.return_as_dict = False

        self.to_numpy = to_numpy
        self.X_train = {k: Lag(X_i, amount=horizon) for k, X_i in X.items()}
        self.X = X
        super().__init__(*self.X.values(), *self.X_train.values())

    def evaluate(self, *args):
        sep = len(args)//2
        X = {k: v for k, v in zip(self.X, args[:sep])}
        X_train = {k: v for k, v in zip(self.X, args[sep:])}

        if self.to_numpy:
            for k in self.X:
                if isinstance(X[k], (pd.Series, pd.DataFrame)):
                    X[k] = X[k].to_numpy()
                if isinstance(X_train[k], (pd.Series, pd.DataFrame)):
                    X_train[k] = X_train[k].to_numpy()

        if not self.return_as_dict:
            X = next(iter(X.values()))
            X_train = next(iter(X_train.values()))

        return {"X_train": X_train, "X_pred": X}


class Predictor(ABC):

    source_init_params = {} # Use {"par": source}, for each parameter in __init__ that requires a source as input.
    target = None

    def __init__(self, format = None):
        if format is None:
            format = {}
        self.format = format

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.set_params()

    @classmethod
    def set_params(cls):
        raise NotImplementedError("This method should be overridden by subclasses to set parameters.")

    @abstractmethod
    def update(self, X: dict | np.ndarray, Y: np.ndarray, X_pred: dict | np.ndarray, Y_hat: np.ndarray, **params) -> np.ndarray | tuple[np.ndarray, dict]:
        pass

    @abstractmethod
    def predict(self, X: dict | np.ndarray, **params) -> np.ndarray | tuple[np.ndarray, dict]:
        # Should return either Y, or (Y, other_results)
        pass

    @abstractmethod
    def get_model_params(self):
        # Should return fitted model parameters
        pass

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


    def update(self, X: dict | np.ndarray, Y: np.ndarray, X_pred: dict | np.ndarray, Y_hat: np.ndarray, **params):


        forecast = self.batch_update(X, Y, X_pred, Y_hat, **params)

        return forecast

    @abstractmethod
    def batch_update(self, X: dict | np.ndarray, Y: np.ndarray, X_pred: dict | np.ndarray, Y_hat: np.ndarray, **params) -> tuple[pd.DataFrame] | pd.DataFrame:
        pass

    @abstractmethod
    def predict(self, X: dict | np.ndarray, **params):
        pass

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
        update_model_sig = inspect.signature(cls.update_model)
        update_model_params = [
            k for k, v in list(update_model_sig.parameters.items())[1:]
            if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls.update_model_params = update_model_params

    def update(self, X: dict | np.ndarray, Y: np.ndarray, X_pred: dict | np.ndarray, Y_hat: np.ndarray, **params):
        # Distribute params
        update_params = {k: v for k, v in params.items() if k in self.update_model_params}
        predict_params = {k: v for k, v in params.items() if k in self.predict_params}

        if isinstance(X, np.ndarray):
            x_ready = not np.isnan(X).any()
        else:
            x_ready = True
        if isinstance(Y, np.ndarray):
            y_ready = not np.isnan(Y).any()
        else:
            y_ready = True

        # Only update if data is valid
        if x_ready and y_ready:
            self.update_model(X, Y, Y_hat, **update_params)

        forecast = self.predict(X_pred, **predict_params)
 
        return forecast

    @abstractmethod
    def update_model(self, x_i, y_i, y_i_hat, **params):
        """
        Update model with new data rows x_i, y_i, and old predictions y_i_hat.
        """

    @abstractmethod
    def predict(self, x_i, **params) -> tuple[pd.Series] | pd.Series:
        # Predict a single row.
        pass

class OLS(BatchPredictor):

    source_init_params = {"n": DimX, "m": DimY}
    target = "mean"

    def __init__(self, n, m):
        super().__init__(format={"mean": Target})
        self.theta = np.zeros((n, m))
        self._n_updates = 0

    def batch_update(self, X: np.ndarray, Y: np.ndarray, X_pred: np.ndarray, Y_hat: np.ndarray, **params):
        # Fit OLS model
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
        X_fit = X[mask]
        Y_fit = Y[mask]
        self.theta = np.linalg.lstsq(X_fit, Y_fit, rcond=None)[0]
        
        pred = self.predict(X_pred)

        return pred

    def predict(self, X: np.ndarray):
        # Predict
        pred = X @ self.theta
        return {"mean": pred}

    def get_model_params(self):
        return self.theta

class WLS(BatchPredictor):

    source_init_params = {"n": DimX, "m": DimY}
    target = "mean"

    def __init__(self, n, m):
        super().__init__(format={"mean": Target})
        self.theta = np.zeros((n, m))
        self._n_updates = 0

    def batch_update(self, X: np.ndarray, Y: np.ndarray, X_pred: np.ndarray, Y_hat: np.ndarray, W: np.ndarray = None, **params):
        # Fit WLS model
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
        X_fit = X[mask]
        Y_fit = Y[mask]

        if W is None:
            W = np.eye(X_fit.shape[0])
        else:
            W = W[np.ix_(mask, mask)]

        XtW = X_fit.T @ W
        self.theta = np.linalg.solve(XtW @ X_fit, XtW @ Y_fit)

        pred = self.predict(X_pred)

        return pred

    def predict(self, X: np.ndarray):
        # Predict
        pred = X @ self.theta
        return {"mean": pred}

    def get_model_params(self):
        return self.theta


class RLS(OnlinePredictor):

    source_init_params = {"n_x": DimX, "n_y": DimY}
    target = "mean"

    def __init__(self, n_x, n_y, R_scale = 1/10000, burn_in = 1, estimate_variance = False):
        super().__init__(format={"mean": Target})
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

    def update_model(self, x_i, y_i, y_i_hat, rls_lambda = 0.99):
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
        pred = np.dot(x.T, self.theta)
        
        if self.estimate_variance:
            var_est = self.variance_estimate.astype(float) # TODO: check, is this the intended return value?
            return {"mean": pred, "var": var_est}

        return {"mean": pred}

    def get_model_params(self):
        return self.theta

class RidgePredictor(BatchPredictor):

    source_init_params = {"m": DimY}
    target = "mean"

    def __init__(self, m, V = None):
        super().__init__(format={"mean": Target})
        self.V = np.eye(m) if V is None else V
        self._old_U = None

    def batch_update(self, X: np.ndarray, Y: np.ndarray, X_pred: np.ndarray, Y_hat: np.ndarray, P: np.ndarray = None,
                Q: np.ndarray = None, theta_tilde: np.ndarray = None, U: np.ndarray = None, V: np.ndarray = None,
                estimate_V: bool = False, return_var: bool = False, return_var_theta: bool = False):

        # Copy X, Y
        X_all = X.copy()
        Y_all = Y.copy()

        # Remove missing values
        X_mask = ~np.isnan(X_all).any(axis=1)
        Y_mask = ~np.isnan(Y_all).any(axis=1)
        mask = X_mask & Y_mask
        X = X_all[mask]
        Y = Y_all[mask]

        # Prepare arrays
        t, n = X.shape
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
       
        self.K = K = X.T @ P @ X + Q
        R = X.T @ P @ Y + Q @ theta_tilde

        # Solve
        self.theta = np.linalg.solve(K, R)

        # Predict
        pred = self.predict(X_pred, return_var = False)

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
        temp1 = P @ X
                
        temp2 = temp1.T @ U[np.ix_(mask, mask)] @ temp1
        
        temp3 = np.linalg.solve(K, temp2)

        self.inner_var_vec_theta = np.linalg.solve(K, temp3.T)

        result = {"mean": pred}

        # If required, compute the decomposed out of sample variance (U, V). Note: this should be partly in sample, but is not feasible without assumptions on U.
        if return_var:
            var_vec_Y_hat_err_U = self.get_out_of_sample_variance(X_pred, U)
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


class RRR(OnlinePredictor):
    # Recursive Ridge Regressor
    # Recursively solves the problem
    # min theta || X_t theta - Y_t ||_{F, D_t} + || theta - tilde_theta_t ||_{F, Q_t}
    # where D = diag(lambda^(t-1), ... , lambda^(t-t)) is a matrix for exponential forgetting,
    # Q is a symmetric weight matrix for the ridge penalty, and tilde_theta_t is a regularization term / prior for theta. 
    # F denotes the Frobenius norm.
    # The solution is used to predict in the model
    # Y_t = X_t theta + E_t, where E_t ~ MN(0, I, V_t).

    source_init_params = {"n": DimX, "m": DimY}
    target = "mean"

    def __init__(self, n, m, burn_in = 1, tilde_k_init_val = 0, track_memory = False, combine_variance = True, full_cov = True):
        super().__init__(format={"mean": Target, "cov": None})  
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
        if combine_variance:
            self.format["cov"] = None

        self.n, self.m = n, m

        self._forgetting_var_state = None
        self._full_cov = full_cov


    def update_model(self, x_i, y_i, y_i_hat, Q: np.ndarray | float = None, theta_tilde: np.ndarray = None, V: np.ndarray = None, estimate_V = True, mem = 0.99):

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

    def predict(self, x: np.ndarray, V = None, return_var_theta = False):
        result = {}

        if V is None:
            V = self.V

        result['mean'] = x.T @ self.theta

        # Compute covariance (u x V)
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

class ExogenousTransform(Transformation):
    """
    Transformation to prepare exogenous inputs for ARMAX type models.
    """

    def __init__(self, data, horizon):
        self.horizons = list(range(1, horizon + 1))
        super().__init__(data = data, sorted_cols = Memory)

    def evaluate(self, data, sorted_cols = None):
        ndim = data.ndim
        if isinstance(data, pd.Series):
            data = pd.DataFrame([data])

        # Subset data using fc
        data = data.fc.subset(horizons=self.horizons)

        # Ensure ordering is correct, i.e. (var1, h1), (var2, h1), ..., (var1, h2), (var2, h2), ...
        if sorted_cols is None:
            sorted_cols = sorted(data.columns, key=lambda x: (x[1], x[0]))

             # Check that horizons are as expected for all variables
            for var in set([col[0] for col in data.columns]):
                var_horizons = [col[1] for col in data.columns if col[0] == var]
                if var_horizons != self.horizons:
                    raise ValueError(f"Variable {var} has horizons {var_horizons}, but expected horizons are {self.horizons}.")

        data = data[sorted_cols]
        if ndim == 1:
            data = data.iloc[0]

        return data, sorted_cols


class CircularBuffer:

    """
    A class for efficiently storing and retrieving the "size" most recent rows of data in a circular buffer.
    """

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
            res = np.vstack([pad, res])
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

class Model(ABC):
    """
    Defines a model of the type Y_{t+h} = f(X_t) + noise, where
    - Y is the target variable to predict
    - X is a set of predictor variables
    - h is the prediction horizon
    - f is a predictor function, which can be either an online or batch predictor.
    The predictor return additional variables, such as multi step predictions (multiples of h),
    prediction intervals, variances etc. The predictor may also use past predictions as input.
    """
    def __init__(self, X: dict | tuple | Transformation, Y : tuple | Transformation, predictor_type, horizon = 1, max_dependency_depth = 100, predictor_init_params = None,  predictor_params = None, format = None, ignore_format = None, sources = None):
        super().__init__()

        if isinstance(X, tuple):
            X = Combine(*X)
        if isinstance(Y, tuple):
            Y = Combine(*Y)
            # TODO: check if Y depends on old Y_hat, if so raise error

        self.dimX = None if isinstance(X, dict) else Dim(X)
        self.dimY = Dim(Y)
        X = InputTransform(horizon, X)

        self.X = X
        self.Y = Y

        self.horizon = horizon

        # Check if X depends on old Y_hat
        self._depends_on_prediction = self.X.depends_on_prediction

        # Create transformers
        if sources is None:
            sources = (DefaultSource,)
        input_source = DefaultSource if not self._depends_on_prediction else Prediction
        self.input_transformer = Transformer(input_source, max_dependency_depth = max_dependency_depth)
        self.target_transformer = Transformer(*sources, max_dependency_depth = max_dependency_depth)
        self.input_transformer.add_transforms(self.X)
        self.target_transformer.add_transforms(self.Y)

        self._old_Y_hat = None
        self._old_X = None
        
        self.predictor = None
        self._format = format or {}
        self._ignore_format = ignore_format or []

        self.configure_predictor(predictor_type, predictor_init_params, predictor_params)

    def configure_predictor(self, predictor_type, predictor_init_params = None, predictor_params = None):
        self.predictor_type = predictor_type
        self.predictor_init_params = predictor_type.source_init_params.copy() | (predictor_init_params or {})
        self.predictor_params = predictor_params or {}
        self.online = issubclass(predictor_type, OnlinePredictor)

    @property
    def format(self):
        predictor_format = {k: v for k, v in self.predictor.format.items() if k not in self._ignore_format}
        return predictor_format | self._format

    def reset_state(self):
        self.input_transformer.reset_state()
        self.target_transformer.reset_state()
        self._old_Y_hat = None
        self._old_X = None
        self.predictor = None


    def update(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, ref = None, check = False, apply_format = True, return_y = False, update_predictor = True, return_data = False, **params):

        data = parse_data(data, ref, check)
        n = data.n            

        # Check for unused parameters
        for k in params.keys():
            if k not in self.predictor_type.params:
                raise ValueError(f"Parameter '{k}' not used by predictor of type {self.predictor_type.__name__}.")

        if self.online and n > 1:
            result = []
            for i in range(n):

                # Parse data at time i.
                data_i = parse_data(data.get_element(i), ref, check)

                # Update model, and store newly transformed data
                result_i = self._update(data_i, 1, return_y, update_predictor, **params)

                result.append(result_i)

            # Convert result to dict of lists
            if isinstance(result[0], dict):
                result = {key: [r[key] for r in result] for key in result[0]}
            
            # Stack any arrays
            for key, values in result.items(): 

                # Check for a reference value
                ref_val = next(iter(value for value in values if value is not None))
                value_type = type(ref_val)

                # Check if any values are None
                if any(value is None for value in values) and ref_val is not None:

                    # Convert to array if pandas object
                    if isinstance(ref_val, (pd.DataFrame, pd.Series)):
                        ref_val = ref_val.to_numpy()

                    # If successful, convert all None values to arrays of NaNs of appropriate shape
                    if isinstance(ref_val, np.ndarray):
                        ref_shape = ref_val.shape

                        # Convert all None values to appropriate type with NaNs
                        for j, value in enumerate(values):
                            if value is None:
                                new_val = np.full(ref_shape, np.nan)    
                                if value_type is pd.DataFrame:
                                    new_val = pd.DataFrame(new_val, columns=ref_val.columns, index=ref_val.index)
                                elif value_type is pd.Series:
                                    new_val = pd.Series(new_val, index=ref_val.index)
                                values[j] = new_val
                    
                # Stack values
                if value_type is np.ndarray:
                    result[key] = np.array(values)
                elif value_type is pd.DataFrame:
                    result[key] = pd.concat(values, axis=0)
                    result[key].columns = ref_val.columns
                elif value_type is pd.Series:
                    result[key] = pd.DataFrame(values, columns = ref_val.index)

        else:
            result = self._update(data, n, return_y, update_predictor, **params)

        # Format output
        if apply_format:
            result = self.format_result(result, data)

        if return_data:
            return result, data

        return result

    def format_result(self, result, ref_data):
        for output, format in self.format.items():
            val = result[output]
            if format is None:
                m = val.shape[1] if ref_data.n > 1 else val.shape[0]
                vars = [f"{output}_{i}" for i in range(m)]
            else:
                if format is Target:
                    vars = get_vars(self._Y)
                else:
                    vars = ref_data.vars(format) # Get variable names, depending on reference data
            result[output] = format_array(val, ref_data[DefaultIndex], vars)

        return result


    def initialize_predictor(self, **init_params):
        init_params = self.predictor_init_params | init_params
        if any(isinstance(v, Source) for v in init_params.values()):
            raise ValueError("Cannot initialize predictor without satisfying Sources in init_params.")
        self.predictor = self.predictor_type(**init_params)

    def _update(self, data: Data, n, return_y = False, update_predictor = True, **params):

        # Include prediction horizon in transform data
        data[PredictionHorizon] = self.horizon

        data = self.target_transformer.transform(data)

        self._Y = data[self.Y]
        data[Target] = self._Y
        n_y = data.m(self.Y)

        # Convert to numpy array if necessary
        if isinstance(self._Y, (pd.DataFrame, pd.Series)):
            Y = self._Y.to_numpy()
        else:
            Y = self._Y

        # Ensure shape of Y
        Y = np.atleast_1d(Y)

        # Initialize storage for old predictions
        if self._old_Y_hat is None:
            self._old_Y_hat = CircularBuffer(size=self.horizon, m=n_y, default_value=np.nan)

        # Get old outputs
        Y_hat_old = self._old_Y_hat.get(n)
        
        # Add old outputs to data
        data[Prediction] = Y_hat_old

        # Transform
        data = self.input_transformer.transform(data)

        # Get X
        X_pred = data[self.X]["X_pred"]
        X_train = data[self.X]["X_train"]

        # NOTE: X is already ensured to be dict or numpy array due to the InputTransform

        # Initialize predictor if necessary
        if self.predictor is None:
            # Construct init params using data if required
            init_params = {}
            for k, v in self.predictor_init_params.items():
                if isinstance(v, Source):
                    if v in data:
                        init_params[k] = data[v]
                    elif v is DimX:
                        init_params[k] = self.dimX.apply(data)
                    elif v is DimY:
                        init_params[k] = self.dimY.apply(data)
                    elif v is Target:
                        init_params[k] = self._Y
                    elif isinstance(v, Transformation):
                        init_params[k] = v.apply(data)
                    else:
                        raise ValueError(f"Cannot fetch data for source {v}.")
                else:
                    init_params[k] = v

            self.initialize_predictor(**init_params)

        # Update predictor parameters
        params = self.predictor_params | params

        # Update predictor
        if update_predictor:
            result = self.predictor.update(X_train, Y, X_pred, Y_hat_old, **params)
        else:
            params = {k: v for k, v in params.items() if k in self.predictor.predict_params}
            result = self.predictor.predict(X_pred, **params)

        # Get new Y_hat
        if isinstance(result, dict):
            Y_hat_name = self.predictor.target or next(iter(result))
            Y_hat = result[Y_hat_name]
        else:
            Y_hat = result

        # Store Y_hat for next update
        self._old_Y_hat.append(Y_hat)

        # Attach Y to result if required
        if return_y:
            result[Target] = self._Y

        # Clean up data dict, making sure no data that depends on predictions or targets are being passed on
        to_remove = {source for source in data if not source.can_be_shared}
        for source in to_remove:
            del data[source]

        return result

    def fit(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, return_y = False, **predictor_params):
        self.reset_state()
        return self.update(data, return_y=return_y, **predictor_params)

    def fit_and_score(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, scorefun = None, burn_in = 0, **predictor_params):
        """
        A method for optimization of model parameters.
        """
        if scorefun is None:
            scorefun = rmse

        result = self.fit(data, return_y = True, **predictor_params)
        Y_hat = result[self.predictor.target or next(iter(result))]
        Y = result[Target]
        resid = Y - Y_hat.shift(self.horizon)
        
        if burn_in > 0:
            resid = resid.iloc[burn_in:]
        
        mask = ~resid.isna().any(axis=1)
        score = scorefun(resid[mask].to_numpy())
        return score
        
    def save_model(self, name: str = None):
        if name is None:
            return pickle.dumps(self)
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(self, f)

    def print_model_tree(self):
        print("X")
        self.X.print_dependency_tree()
        print("\nY")
        self.Y.print_dependency_tree()

class Ensemble:

    def __init__(self, *models: Model):
        self.models = models
        
    @property
    def online(self):
        return any(model.online for model in self.models)

    def _update(self, data, return_y = False, update_predictor = True, model_params_dict = None, **params):
        results = {}
        for model in self.models:
            model_params = {k: v for k, v in params.items() if k in model.predictor_type.params}
            if model_params_dict is not None and model in model_params_dict:
                model_params = model_params | model_params_dict[model]
            result = model.update(data, return_y=return_y, update_predictor=update_predictor, apply_format = False, **model_params)
            results[model] = result

        return results

    def update(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, ref = None, check = True, return_y = False, update_predictor = True, apply_format = True, model_params_dict = None, **params):

        data = parse_data(data, ref, check)
        n = data.n

        if self.online and n > 1:
            result = []
            for i in range(n):

                # Parse data at time i.
                data_i = parse_data(data.get_element(i), ref, check)

                # Update model, and store newly transformed data
                result_i = self._update(data_i, return_y, update_predictor, model_params_dict, **params)

                result.append(result_i)

            # Convert dict of {model: [{output: value}]} to dict of {model: {output: [value]}}
            reshaped_result = {}
            for model in self.models:
                # Reshape each model's result
                reshaped_result[model] = {key: [r[model][key] for r in result] for key in result[0][model]}
            
                # Stack any arrays
                for key, values in reshaped_result[model].items():
                    if isinstance(values[0], np.ndarray):
                        reshaped_result[model][key] = np.array(values)
                    elif isinstance(values[0], pd.DataFrame):
                        reshaped_result[model][key] = pd.concat(values, axis=0)
                    elif isinstance(values[0], pd.Series):
                        reshaped_result[model][key] = pd.DataFrame(values)

            result = reshaped_result
        else:
            result = self._update(data, return_y, update_predictor, **params)

        # Format output
        if apply_format:
            for model in self.models:
                result[model] = model.format_result(result[model], data)

        return result

    def fit(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, update_predictor = True, **params):
        self.reset_state()
        return self.update(data, update_predictor=update_predictor, **params)
    
    def _fit(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, return_y = False, **params):
        self.reset_state()
        return Ensemble.update(self, data, return_y=return_y, update_predictor=True, apply_format = False, **params)

    def fit_and_score(self, data: dict | pd.DataFrame | pd.Series | np.ndarray, scorefun = None, burn_in = 0, **params):
        if scorefun is None:
            scorefun = rmse

        # TODO: fix, maybe horizons align incorrectly?
        result = self._fit(data, return_y = True, **params)
        scores = {}
        for model in self.models:
            Y_hat = result[model][model.predictor.target or next(iter(result[model]))]
            Y = result[model][Target]
            if not isinstance(Y, np.ndarray):
                Y = Y.to_numpy()
            resid = Y[model.horizon:] - Y_hat[:-model.horizon]
            
            if burn_in > 0:
                resid = resid[burn_in:]
            
            mask = ~np.isnan(resid).any(axis=1)
            scores[model] = scorefun(resid[mask])
        return scores        

    def reset_state(self):
        for model in self.models:
            model.reset_state()

    def get_parameters(self):
        return {model: model.predictor.get_model_params() for model in self.models}


    def update_configuration(self, model, predictor_type = None, predictor_init_params = None, predictor_params = None):
        if predictor_type is None:
            predictor_type = model.predictor_type
        predictor_init_params = model.predictor_init_params | (predictor_init_params or {})
        predictor_params = model.predictor_params | (predictor_params or {})
        model.configure_predictor(predictor_type, predictor_init_params, predictor_params)
        model.reset_state()


    def save_model(self, name: str = None):
        if name is None:
            return pickle.dumps(self)
        if not name.endswith(".pkl"):
            name = name + ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(self, f)

class HorizonEnsemble(Ensemble):

    def __init__(self, X, Y, predictor_type, horizons: tuple = (1,), max_dependency_depth = 100, input_horizons: dict = None, predictor_init_params = None,  predictor_params = None):
        self.horizons = horizons
        if input_horizons is None:
            input_horizons = {h: (0, h) for h in horizons}
        models = []

        if isinstance(Y, (tuple, list)):
            Y_sub = [HorizonTarget(Y_i) for Y_i in Y] # Cheating a bit here to ensure formatting is correct.
            Y = Combine(*Y_sub)  
        else:
            Y = HorizonTarget(Y)  # Cheating a bit here to ensure formatting is correct.

        for h, h_in in input_horizons.items():
            if isinstance(X, (tuple, list)):
                X_sub = [Subset(data = X_i, horizons = h_in) for X_i in X]
                X_h = Combine(*X_sub)
            else:
                X_h = Subset(data = X, horizons = h_in)
                
            model = Model(X_h, Y, predictor_type, horizon = h, max_dependency_depth = max_dependency_depth, predictor_init_params = predictor_init_params, predictor_params = predictor_params)
            models.append(model)
        
        super().__init__(*models)

    def update(self, data, ref=None, check=True, return_y = False, update_predictor=True, apply_format = True, model_params_dict = None, **params):
        res = super().update(data, ref, check, return_y=return_y, update_predictor=update_predictor, apply_format=apply_format, model_params_dict=model_params_dict, **params)

        # Initialize variable for target
        y = None

        # Apply multiindex format and store in nested list
        all_values = []
        for h, res_h in zip(self.horizons, res.values()):
            all_values_h = []
            if isinstance(res_h, dict):
                if Target in res_h:
                    if y is None:
                        y = res_h[Target]
                    del res_h[Target]

                for val in res_h.values():
                    if isinstance(val, (pd.DataFrame, pd.Series, np.ndarray)):
                        if isinstance(val, pd.DataFrame):
                            val.columns = pd.MultiIndex.from_product([val.columns, [h]], names=['Variable', 'Horizon'])
                        elif isinstance(val, pd.Series):
                            val.index = pd.MultiIndex.from_product([val.index, [h]], names=['Variable', 'Horizon'])
                    all_values_h.append(val)
            else:
                all_values_h.append(res_h)

            all_values.append(all_values_h)
        
        combined_values = nested_concat(all_values, across="columns")

        result = {output: val for output, val in zip(res_h.keys(), combined_values)}

        if return_y and y is not None:
            result[Target] = y

        return result

    def get_model_parameters(self):
        return {model.horizon: model.predictor.get_model_params() for model in self.models}

    def update_configuration(self, horizon, predictor_type=None, predictor_init_params=None, predictor_params=None):
        model = self.models[self.horizons.index(horizon)]
        return super().update_configuration(model, predictor_type, predictor_init_params, predictor_params)

    def fit_and_score(self, data, scorefun=None, burn_in=0, **params):
        res = super().fit_and_score(data, scorefun, burn_in, **params)
        return {h: res_h for h, res_h in zip(self.horizons, res.values())}


class ARXPredictor(RRR):

    """
    Predictor for the ARX model based on the RRR predictor.
    """

    target = 1

    def __init__(self, horizon, n, p, trend, *args, **kwargs):
        self.horizon = horizon
        n_rrr = n // horizon + p + (1 if trend else 0)
        super().__init__(n_rrr, *args, **kwargs)
        self.format = {} # Overwrite format, as it no longer applies as specified in RRR
    
    def predict(self, x_i, *args, **kwargs):
        # Fetch y_hist and exogenous from x_i
        y_hist = x_i["y_hist"]
        exogenous = x_i["exogenous"]
        trend = x_i["trend"] if "trend" in x_i else None

        # Get the 1-step exogenous variables, i.e. the n first entries
        n = exogenous.shape[0] // self.horizon

        # Apply predict recursively for each horizon
        result = []
        for i in range(self.horizon):
            exog_i = exogenous[i*n:(i+1)*n]

            # Combine into single input
            if trend is not None:
                x_ii = np.hstack([y_hist, exog_i, trend])
            else:
                x_ii = np.hstack([y_hist, exog_i])

            # Predict
            result.append(super().predict(x_ii, *args, **kwargs)["mean"])

            # Update y_hist
            y_hist = np.append(result[-1], y_hist[:-1])

        # TODO: include prediction error variance estimates

        return {i+1: result[i] for i in range(self.horizon)}

    def update_model(self, x_i, y_i, y_i_hat, Q = None, theta_tilde = None, u = 1, V = None, estimate_V=True, mem=0.99):
        y_hist = x_i["y_hist"]
        exogenous = x_i["exogenous"]
        n = exogenous.shape[0] // self.horizon
        exog = exogenous[0:n] # Use only 1-step exogenous for update
        if "trend" in x_i:
            x_i = np.hstack([y_hist, exog, x_i["trend"]])
        else:
            x_i = np.hstack([y_hist, exog])
        if not np.isnan(x_i).any():
            return super().update_model(x_i, y_i, y_i_hat, Q, theta_tilde, u, V, estimate_V, mem)


class ARX(Model):

    def __init__(self, X, Y, p, horizon, trend = False, predictor_init_params = {}):

        if isinstance(X, (tuple, list)):
            X = Combine(*X)

        X = ExogenousTransform(X, horizon)

        # Construct moving average and autoregressive lags
        y_hist = BackShift([i for i in range(p)], data = Target, skip_duplicates=False, initial_value=0)

        # Note, predictions are already lagged by 1 step, so we start from i=1

        # TODO: consider multivariate case -backshift should be created to maintain ordering of variables and predictor updated accordingly

        X_dict = {"y_hist": y_hist, "exogenous": X}

        if trend:
            X_dict["trend"] = One()

        predictor_init_params.update({"horizon": horizon, "n": Dim(X), "p": p, "trend": trend})

        self._pred_horizon = horizon

        super().__init__(X_dict, Y, ARXPredictor, 1, predictor_init_params=predictor_init_params, format = {i+1: Target for i in range(horizon)})

#        self._format["cov"] = None

    def update(self, data, ref=None, check=False, apply_format=True, return_y=False, update_predictor=True, return_data=False, **params):
        result = super().update(data, ref, check, apply_format, return_y, update_predictor, return_data, **params)

        # Format "mean_i" into single "mean" with horizon
        means = [result.pop(i+1) for i in range(self._pred_horizon)]
        result["mean"] = pd.concat(means, axis=1)
        # Set second level names to horizons
        result["mean"].columns = pd.MultiIndex.from_tuples(
            [(col[0], i+1) for col, i in zip(result["mean"].columns, range(self._pred_horizon))]
        )

        return result

def load_model(file_name):
    if not file_name.endswith(".pkl"):
        file_name = file_name + ".pkl"
    with open(file_name, 'rb') as f:
        return pickle.load(f)
# %%