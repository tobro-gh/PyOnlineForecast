#%%
from __future__ import annotations
import numpy as np
import inspect
import os
import functools
import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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
UPDATE_PREDICTOR = make_keyword("UPDATE_PREDICTOR")
X_INIT = make_keyword("X_INIT")
Y_INIT = make_keyword("Y_INIT")
Z_INIT = make_keyword("Z_INIT")
STATE = make_keyword("STATE")

class Transformation(Source):

    """
    Generic class for transformations. Subclasses should implement the evaluate method.

    The class provides functionality for matching placeholder data sources to data, and evaluating
    the transformation based on the provided data. Subclasses should call super().__init__(*args, **kwargs)
    with keyword names matching the evaluate method parameters to specify how data should be passed.   
    Also enables basic operations between transformations, such as +, -, *, etc.
    """

    # TODO: consider making an "online" flag, and an "apply_online" method
    # that retrieves the most recent input data only, and iteratively updates
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
            if not (key in self._inputs or self._accepts_var_kwargs):
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

        super().__init__()

    def __init_subclass__(cls):

        if hasattr(cls, "evaluate"):
            sig = inspect.signature(cls.evaluate)
            cls.evaluate_kwargs = list(sig.parameters.keys())[1:]

#            var_positional_arg = next((param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.VAR_POSITIONAL), None)
 #           var_keyword_arg = next((param.name for param in sig.parameters.values() if param.kind == inspect.Parameter.VAR_KEYWORD), None)

  #          cls._accepts_args = var_keyword_arg is not None
  #          cls._accepts_kwargs = var_positional_arg is not None

            cls._accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())

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

    def apply(self, data = None, memory = None, recursion_pars = None, return_recursion_pars = False, ref = None, update_predictor = True, copy_data = True, track_state = False, **params):

        # NOTE: copy_data is used to avoid modifying input data (unless requested). When applied recursively, copy_data should be False, as we do want to update the data with intermediate results.
        data = parse_data(data, ref = ref, copy = copy_data)

        evaluate_kwargs = {}
        evaluate_args = []

        if recursion_pars is None:
            if track_state:
                recursion_pars = self.recursion_pars or {} # Only loads stored state if not provided. Nested calls of apply should never load own state.
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

            elif val is STATE:
                t_val = data.copy() # Pass all data computed so far
            
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

        # TODO: consider tracking used params and warning if any are unused

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

    def evaluate(self) -> tuple[Any, Any] | Any:
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
    
    def reset_state(self):
        self.recursion_pars = None

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

def parse_data(data : dict, ref = None, copy = True):
    if not isinstance(data, dict):
        data = {DEFAULT_SOURCE: data}
    elif copy:
        data = data.copy()
    
    ref_val = data[ref or next(iter(data))]
 
    if not DEFAULT_SOURCE in data:
        data[DEFAULT_SOURCE] = ref_val

    return data
    
class Dim(Transformation):
    
    def __init__(self, data, axis = 1):
        self.axis = axis
        super().__init__(data = data)

    def evaluate(self, data):
        return data.shape[self.axis]
    

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

class Apply(Transformation):

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

def transform_wrapper(func):
    """
    Decorator to wrap functions on transformations, to return a new transformation with the function applied to the output of the original transformation.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return Apply(func, self, *args, **kwargs)
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

class ToArray(Transformation):

    def __init__(self, data):
        super().__init__(data = data)

    def evaluate(self, data):
        if isinstance(data, np.ndarray):
            return data
        return np.asarray(data)    

class Combine(Transformation):
    def __init__(self, *sources, names = None):
        super().__init__(*sources)
        self.sources = sources
        if names is not None:
            if len(names) != len(sources):
                raise ValueError("Length of names must match number of sources.")
        else:
            names = self.sources
        self.names = names

    def evaluate(self, *data):
        return {name: v for name, v in zip(self.names, data)}

class DesignMatrix(Transformation):

    def __init__(self, *data):
        data = [ToArray(d) for d in data]
        super().__init__(*data)
    
    def evaluate(self, *data):
        # Reshape arrays from (t, d1, d2, ...) to (t, d1*d2*...) and stack horizontally
        data = [d.reshape(d.shape[0], -1) for d in data]
        return np.hstack(data)

    
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
        data = ToArray(data)
        super().__init__(data = data, prev_values = MEMORY)


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
        if isinstance(data, np.ndarray):
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


class Prediction(Transformation):

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.set_params()

    @classmethod
    def set_params(cls):
        # Set update parameters for the predictor
        update_sig = inspect.signature(cls.update)
        cls.params = [k for k, v in list(update_sig.parameters.items())[1:] if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) and k not in ["state", "X", "Y", "X_train"]]

        # Set parameters for the predict method
        predict_sig = inspect.signature(cls.predict)
        predict_params = [
            k for k, v in list(predict_sig.parameters.items())[1:]
            if v.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        cls.predict_params = predict_params

    def __init__(self, X, Y, horizon, *args, Z = None, score_mode = False, default_params = None, **kwargs):
        """
        Y_t = f(X_{t-horizon}) => state_t
        hat{Y}_{t+h} = g(state_t, Z_t)

        Generic transformation providing a base class for predictors.
        Parameters:
        - X: input data (Source with output that can be used with Lag)
        - Y: target (Source, not lagged)
        - Z: additional features (Source, not lagged) that may also be used for prediction.
        - horizon: forecast horizon (int)
        - default_params: dict of default predictor parameters to be used in the update and predict methods. These can be overridden by providing parameters in the evaluate method.
        - args, kwargs: arguments to be used by the create method to initialise the predictor.
        """
        self.X = X
        self.Y = Y
        self.X_train = Lag(X, amount = horizon)
        self.args = args
        self.kwargs = kwargs
        self._score_mode = score_mode
        self._default_params = default_params or {}

        # Check that default_params are valid
        for param in self._default_params.keys():
            if param not in self.params:
                raise ValueError(f"Parameter '{param}' not recognized for predictor.")

        if Z is None:
            Z_kwarg = {}
            self._use_Z = False
        else:
            Z_kwarg = {"Z": Z}
            self._use_Z = True

        super().__init__(X, Y, update_predictor = UPDATE_PREDICTOR, state = MEMORY, **Z_kwarg)
    
    @property
    def score_mode(self):
        return self._score_mode

    @score_mode.setter
    def score_mode(self, value: bool):
        self._score_mode = value

    def set_score_mode(self):
        self._score_mode = True
    
    def unset_score_mode(self):
        self._score_mode = False
        
    def _get_value(self, v, data):
        if isinstance(v, Source):
            if v in data:
                return data[v]
            elif isinstance(v, Transformation):
                return v.apply(data)
        else:
            return v

    def _create(self, X, Y, Z):
        data = {X_INIT: X, Y_INIT: Y, Z_INIT: Z}
        # Construct init params using data if required
        args = []
        for arg in self.args:
            args.append(self._get_value(arg, data))
        kwargs = {}
        for k, v in self.kwargs.items():
            kwargs[k] = self._get_value(v, data)

        # Call create method with constructed params
        return self.create(*args, **kwargs)

    @abstractmethod
    def create(self, *args, **kwargs):
        """
        Method to create the predictor state. Should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def update(self, state, X, Y, X_train, Z = None, **params) -> tuple:
        """
        Method to update the predictor with new data. Should be implemented by subclasses. Return value should be the prediction for the current time step and the updated state of the predictor.
        Fitting should be done as Y~X_train.
        """
        pass

    @abstractmethod
    def predict(self, state, X, Z = None, **params):
        """
        Method to make predictions. Should be implemented by subclasses.
        """
        pass

    def score(self, state, X, Y, prediction, Z = None, **params):
        raise NotImplementedError("Score method not implemented for this predictor.")
    

    def evaluate(self, X, Y, update_predictor, state = None, Z = None, **params):

        # Combine provided and default predictor parameters
        params = self._default_params | params

        if self._use_Z:
            params["Z"] = Z

        if state is None:
            # Create predictor state
            predictor_state = self._create(X, Y, Z)

            X_state = None
            
        else:
            predictor_state, X_state = state

        if update_predictor:

            # Get training data
            X_train, X_state = self.X_train.evaluate(X, X_state)
            result, predictor_state = self.update(predictor_state, X, Y, X_train, **params)

        else:
            result = self.predict(predictor_state, X, **params)

        if self.score_mode:
            result = self.score(predictor_state, X, Y, result, **params)

        # Return result and state
        return result, (predictor_state, X_state)

    @property
    def predictor(self):
        if self.recursion_pars is None:
            return None
        else:
            return self.recursion_pars[self][0]


class OnlinePrediction(Prediction):

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.set_params()

    @classmethod
    def set_params(cls):
        cls.set_predict_params()
        cls.set_update_model_params()
        cls.params = cls.predict_params + cls.update_model_params

    @classmethod
    def set_predict_params(cls):
        predict_sig = inspect.signature(cls.online_predict)
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

    @classmethod
    def convert_arrays(cls, X, Y = None, X_train = None, Z = None):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(Y, np.ndarray) and Y is not None:
            Y = np.asarray(Y)
        if not isinstance(X_train, np.ndarray) and X_train is not None:
            X_train = np.asarray(X_train)
        if not isinstance(Z, np.ndarray) and Z is not None:
            Z = np.asarray(Z)
        
        # Ensure 2D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y is not None and Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if X_train is not None and X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if Z is not None and Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        return X, Y, X_train, Z

    def update(self, state, X: np.ndarray, Y: np.ndarray, X_train: np.ndarray, Z: np.ndarray = None, **params):
        X, Y, X_train, Z = self.convert_arrays(X, Y, X_train, Z)
            
        # Distribute params
        update_params = {k: v for k, v in params.items() if k in self.update_model_params}
        predict_params = {k: v for k, v in params.items() if k in self.predict_params}

        n = X.shape[0]
        forecasts = []

        # Loop over each row of data
        for i in range(n):
            
            x = X[i]
            y = Y[i]
            x_train = X_train[i]

            if self._use_Z:
                update_params["z_i"] = Z[i]
                predict_params["z_i"] = Z[i]

            y_ready = not np.isnan(y).any()
            x_train_ready = not np.isnan(x_train).any()
   
            # Only update if data is valid
            if x_train_ready and y_ready:
                forecast, state = self.online_update(state, x, y, x_train, **update_params)
            else:
                forecast = self.online_predict(state, x, **predict_params)
 
            forecasts.append(forecast)

        forecasts = stack_results(forecasts)

        return forecasts, state

    def predict(self, state, X: np.ndarray, Z = None, **params):
        # Check parameters
        for k in params.keys():
            if k not in self.predict_params:
                raise ValueError(f"Parameter '{k}' not recognized for prediction.")

        X, _, _, Z = self.convert_arrays(X, Z=Z)

        # Predict multiple rows.
        n = X.shape[0]
        forecasts = []

        for i in range(n):
            x = X[i]
            if self._use_Z:
                params["z"] = Z[i]
            forecast = self.online_predict(state, x, **params)
            forecasts.append(forecast)

        forecasts = stack_results(forecasts)

        return forecasts

    @abstractmethod
    def online_update(self, state, x_i, y_i, x_train_i, z_i = None, **params) -> tuple:
        """
        Update model with new data rows x_train_i, y_i, and make prediction for x_i. Should return the prediction for x_i and the updated state of the model.
        """

    @abstractmethod
    def online_predict(self, state, x_i, z_i = None, **params):
        # Predict a single row.
        pass

_format_like_registry = {}

def register_format_like(source: type, target: type):
    def decorator(format_func):
        sig = inspect.signature(format_func)
        # Check that format_func has a single argument
        if len(sig.parameters) != 2:
            raise ValueError("Format function must have exactly two arguments: value and reference value.")

        @functools.wraps(format_func)
        def wrapped_format_func(val, ref_val):
            return format_func(val, ref_val)

        _format_like_registry[(source, target)] = wrapped_format_func
        return wrapped_format_func
    return decorator

def format_like(val, ref_val):
    if val is ref_val:
        return val

    for base in val.__class__.__mro__:
        if (base, ref_val.__class__) in _format_like_registry:
            format_func = _format_like_registry[(base, ref_val.__class__)]
            return format_func(val, ref_val)
    return val

@register_format_like(object, np.ndarray)
def _(val, ref_val):
    return np.asarray(val)

@register_format_like(np.ndarray, np.ndarray)
def _(val, ref_val):
    return val.reshape(ref_val.shape)

class Format(Transformation):
    """
    Generic transformation to format the output according to rules provided for specific subclasses and source types.
    """

    def __init_subclass__(cls):
        # Make empty subclass registry
        cls.registry = {}
        cls.resolver_registry = {}
        super().__init_subclass__()

    def __init__(self, source: Source):
        self.source = source
        self.formatter = self.get_formatter(source)
        super().__init__(source, STATE, memory = MEMORY)
        # data needs to be a dependence since STATE needs to contain all dependencies

    @classmethod
    def find_formatter(cls, source):
        # Use multiple dispatch to find a formatter for the source type, checking for registered formatters for the source type and its base classes, and then for registered resolvers that can generate a formatter based on the source type and its base classes.
        for base in type(source).__mro__:
            if base in cls.registry:
                return cls.registry[base]

    @classmethod
    def find_resolver(cls, source):
        for base in type(source).__mro__:
            if base in cls.resolver_registry:
                return cls.resolver_registry[base]

    @classmethod
    def get_formatter(cls, source):
        # Check if formatter (source, state) -> formatted value is registered for the source type
        formatter = cls.find_formatter(source)

        # If not, find a resolver that can generate a formatter based on the source
        if formatter is None:
            resolver = cls.find_resolver(source)
            if resolver is not None:
                formatter = resolver(source)

        # If not, format like source if possible
        if formatter is None:
            # Return formatter that formats like source
            def formatter(transform, state, memory):
                return format_like(state[transform], state[source])
            
        return formatter

    @classmethod
    def get_resolver(cls, source):
        return cls.resolver_registry.get(type(source), None)

    def evaluate(self, source, state, memory = None):
        # Note, source should stay in the evaluate method to ensure transformation treats it as a dependency
        return self.formatter(self.source, state, memory)
    
    @classmethod
    def check_formatter(cls, formatter):
        sig = inspect.signature(formatter)
        if len(sig.parameters) != 3:
            raise ValueError("Formatter function must have exactly three arguments: source, state, and memory.")

    @classmethod
    def register(cls, source_type: type):
        def decorator(format_func):
            cls.check_formatter(format_func)
            cls.registry[source_type] = format_func
            return format_func
        return decorator

    @classmethod
    def register_resolver(cls, source_type: type):
        def decorator(resolver_func):
            # Check that resolver_func has the correct signature
            sig = inspect.signature(resolver_func)
            if len(sig.parameters) != 1:
                raise ValueError("Resolver function must have exactly one argument: the source.")
            cls.resolver_registry[source_type] = resolver_func
            return resolver_func
        return decorator