"""Core functionality for sources and transformations.

The module provides the basic objects to define data pipelines using sources and
transformations. The main components of the module are,

- Source: a placeholder for data sources
- Transformation: a base class for data transformations.

Sources and transformations can be composed to build a computational graph which is
lazily evaluating using the ``apply`` or ``__call__`` methods.

"""

from __future__ import annotations

import functools
import inspect
from typing import Any

import numpy as np


class Source:
    """Base class for composable placeholders for data sources.

    Supports operator overloading (+, -, *, /, **) and attribute/item access to enable
    data transformations.

    Parameters
    ----------
    name : str, optional
        Name identifier for the source.

    Methods
    -------
    get_attr
        Access an attribute of the source.
    """

    _name = None

    def __init__(self, name=None):
        self._name = name

    def __repr__(self):
        """Return a string representation of the source, using the name if provided."""
        return self._name

    def __add__(self, other):
        """Return self + other as a _SumTransformation."""
        return _SumTransformation(self, other)

    def __radd__(self, other):
        """Return other + self as a _SumTransformation."""
        return _SumTransformation(other, self)

    def __sub__(self, other):
        """Return self - other as a _SubTransformation."""
        return _SubTransformation(self, other)

    def __rsub__(self, other):
        """Return other - self as a _SubTransformation."""
        return _SubTransformation(other, self)

    def __mul__(self, other):
        """Return self * other as a _MulTransformation."""
        return _MulTransformation(self, other)

    def __rmul__(self, other):
        """Return other * self as a _MulTransformation."""
        return _MulTransformation(self, other)

    def __truediv__(self, other):
        """Return self / other as a _DivTransformation."""
        return _DivTransformation(self, other)

    def __rtruediv__(self, other):
        """Return other / self as a _DivTransformation."""
        return _DivTransformation(other, self)

    def __pow__(self, other):
        """Return self ** other as a _PowTransformation."""
        return _PowTransformation(self, other)

    def __rpow__(self, other):
        """Return other ** self as a _PowTransformation."""
        return _PowTransformation(other, self)

    def __getitem__(self, key):
        """Return self[key] as a GetItem."""
        return GetItem(self, key)

    def __matmul__(self, other):
        """Return self @ other as a _MatmulTransformation."""
        return _MatmulTransformation(self, other)
    
    def __rmatmul__(self, other):
        """Return other @ self as a _MatmulTransformation."""
        return _MatmulTransformation(other, self)

    def get_attr(self, name):
        """Access an attribute of the source as a GetAttr.
        
        Parameters
        ----------
        name : str
            Attribute name to access.
        
        Returns
        -------
        GetAttr
            Transformation accessing self.name.
        """
        return GetAttr(self, name)

class KeywordSource(Source):
    """Singleton keyword placeholder for special data sources.
    
    KeywordSource instances are cached in a registry and support pickle serialization.
    Use module-level keywords like MEMORY, DEFAULT_SOURCE, and STATE instead of
    creating new instances.
    
    Methods
    -------
    list_keywords
        Return all registered keyword names.
    
    See Also
    --------
    make_keyword : Factory function to access or create keywords.
    """
    
    _registry = {}

    def __reduce__(self):
        """Return a tuple for pickling the KeyWordSource ensuring singleton behavior."""
        return make_keyword, (self._name,)

    @classmethod
    def list_keywords(cls):
        """Return a list of all registered keyword names."""
        return list(cls._registry.keys())


def make_keyword(name: str) -> Source:
    """Create or retrieve a singleton KeywordSource with the given name."""
    if name not in KeywordSource._registry:
        KeywordSource._registry[name] = KeywordSource(name)
    return KeywordSource._registry[name]

# Create module-level keywords
MEMORY = make_keyword("MEMORY")
"""Special source for memory parameters of transformations."""
DEFAULT_SOURCE = make_keyword("DEFAULT_SOURCE")
"""Default source for data.
``parse_data`` will map unnamed data to this source for use in transformations."""
STATE = make_keyword("STATE")
"""Source for intermediate data when applying transformations."""

class Transformation(Source):
    """Base class for data transformations.
    
    Transforms combine sources and data via an ``evaluate`` method. Subclasses specify 
    input sources as arguments to ``__init__``, which are matched to ``evaluate`` method
    parameters when called via ``apply``. Matching is done by name for keyword sources and
    by position for positional sources by inspecting the ``evaluate`` method signature.
    
    Subclasses should override the ``evaluate`` method to implement the transformation
    logic and extend the '__init__' method to initialise fixed parameters and specify
    external Source inputs for use with the ``apply`` method. When the ``apply`` method is
    called, sources specified in ``__init__`` will be matched to values, either as direct
    inputs if specified in the input data of ``apply``, or by recursively calling the
    ``apply`` method on the sources that are transforms. Once dependencies are resolved,
    the ``evaluate`` method is called with the resolved arguments to compute the output.
    
    The special keyword source MEMORY can be used to access outputs
    from previous evaluations of the transformation, see also the ``evaluate`` method.
    The STATE source can be used to access all intermediate results computed by the
    ``apply`` method.
        
    Parameters
    ----------
    *apply_args : Source
        Positional sources passed to ``evaluate`` in order.
    **apply_kwargs : Source
        Keyword sources mapped to ``evaluate`` method parameters.
        Keys must match parameter names in the ``evaluate`` signature.
    
    Raises
    ------
    KeyError
        If a keyword argument name does not match an ``evaluate`` parameter.
    ValueError
        If any input is not a Source instance.

    Attributes
    ----------
    sources : list[Source]
        List of input sources for the transformation.
    formatter : Format or None
        Optional formatter to apply to the output of the transformation.
    recursion_pars : dict or None
        Cached state of transformation and nested transformations.

    Methods
    -------
    evaluate
        Compute the transformation output given input data.
    apply
        Match sources to data and evaluate the transformation.
    """

    # TODO: consider making an "online" flag, and an "apply_online" method
    # that retrieves the most recent input data only, and iteratively updates
    # outputs. Transforms with online flags should be evaluated using apply_online,
    # whilst other transforms can be evaluated normally. The data
    # dict should be updated incrementally for transforms with the online flag.

    # Superclass for transformations
    def __init__(self, *apply_args, **apply_kwargs):
        # Use inspect to bind *apply_args and **apply_kwargs
        sig = inspect.signature(self.evaluate)
        bound_args = sig.bind(*apply_args, **apply_kwargs)
        self.apply_kwargs = bound_args.kwargs
        self.apply_args = bound_args.args

        # Check that args and kwargs refer to valid inputs
        for val in list(apply_kwargs.values()) + list(apply_args):
            if not isinstance(val, Source):
                raise ValueError(f"Input {val} must be a Source instance: {self}.")

        # Build pairs of (name, value) for args and kwargs combined
        self._apply_pairs = [(None, val) for val in apply_args] + list(
            apply_kwargs.items()
        )

        # Determine all sources
        self.sources = list(self.apply_kwargs.values()) + list(self.apply_args)
        self.dependencies = [v for v in self.sources if isinstance(v, Transformation)]

        # Check if MEMORY is used
        self._use_memory = MEMORY in self.sources

        # Fetch evaluate signature
        self._eval_sig = inspect.signature(self.evaluate)

        # Initialise memory state
        self.recursion_pars = None

        # Set formatter
        self.formatter = None

        # Determine free parameters (keyword inputs that are not in apply_kwargs)
        self._free_params = [
            p
            for p in self._eval_sig.parameters
            if p not in self.apply_kwargs and p != "self"
        ]

        super().__init__()

    def __init_subclass__(cls):
        """
        Process ``evaluate`` signature and modify ``__init__`` to capture sources.

        Process the evaluate method signature and modify the __init__ method to capture
        parameters for tracking the computational graph.
        """
        if hasattr(cls, "evaluate"):
            sig = inspect.signature(cls.evaluate)
            cls.evaluate_kwargs = list(sig.parameters.keys())[1:]

            cls._accepts_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in sig.parameters.values()
            )

            cls._inputs = [
                n
                for n, p in inspect.signature(cls.evaluate).parameters.items()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            ][1:]

        else:
            raise ValueError("No evaluate method found.")

        # Overwrite init to capture parameters
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._init_params = (
                inspect.signature(original_init).bind(self, *args, **kwargs).arguments
            )
            del self._init_params["self"]

        cls.__init__ = new_init

    def __repr__(self):
        """Return a string representation of the transformation."""
        return f"{self.__class__.__name__}"

    def set_formatter(self, formatter: Format):
        """Set the formatter for the transformation.

        Set a formatter transformation which will be applied to the output of the
        transformation.

        Parameters
        ----------        
            formatter (Format): The formatter to use.
        """
        self.formatter = formatter

    def set_format(self, data_format: type[Format]):
        """Set a format for the transformation.

        This is the same as calling ``set_formatter`` with formatter instantiated on self.
        """
        self.formatter = data_format(self)

    def clear_formatter(self):
        """Clear the formatter."""
        self.formatter = None

    # TODO: add check for circular dependencies

    def apply(
        self,
        data=None,
        memory=None,
        recursion_pars=None,
        return_recursion_pars=False,
        ref=None,
        copy_data=True,
        track_state=False,
        formatter: Format = None,
        keywords=None,
        **params,
    ):
        """Evaluate the transformation with given data.
        
        Matches input sources to data, evaluates dependencies recursively, then calls
        ``evaluate`` with resolved arguments.
        
        Parameters
        ----------
        data : dict, optional
            Data mapping sources to values.
        memory : Any, optional
            Cached state of the current transformation for use in evaluation.
        recursion_pars : dict, optional
            Memory states of nested transformations for recursive evaluation, stored in
            a flat dict mapping transformations to their memory states
        return_recursion_pars : bool, default False
            If True, return (result, recursion_pars) tuple.
        ref : object, optional
            Reference object for data parsing.
        copy_data : bool, default True
            If True, copy input data to avoid modification.
        track_state : bool, default False
            If True, store recursion state for future calls.
        formatter : Format, optional
            Formatter to apply to results.
        keywords : dict, optional
            Mapping of keyword names to values.
        **params : Any
            Additional parameters for free parameters or nested transforms.
        
        Returns
        -------
        result : Any
            The transformation output.
        recursion_pars : dict
            (Only if ``return_recursion_pars=True``)
        """
        # NOTE: copy_data is used to avoid modifying input data (unless requested). When applied recursively, copy_data should be False, as we do want to update the data with intermediate results.
        data = parse_data(data, ref=ref, copy=copy_data)

        evaluate_kwargs = {}
        evaluate_args = []

        if recursion_pars is None:
            if track_state:
                recursion_pars = (
                    self.recursion_pars or {}
                )  # Only loads stored state if not provided. Nested calls of apply should never load own state.
            else:
                recursion_pars = {}

        new_recursion_pars = {}

        targeted_params = params.get(self, {})

        # Load memory from provided recursion pars
        if memory is None:
            memory = recursion_pars.get(self, None)

        if keywords is None:
            keywords = {}

        # Check inputs
        for name, val in self._apply_pairs:

            # Already in data?
            if val in data:
                t_val = data[val]

            elif val is MEMORY:
                # TODO: consider fetching default from evaluate signature
                t_val = memory

            elif val is STATE:
                t_val = data.copy()  # Pass all data computed so far

            elif val in targeted_params:
                t_val = targeted_params[val]

            elif val in params:
                t_val = params[val]

            # Attempt to fetch transformation dependencies if not provided directly.
            elif isinstance(val, Transformation):
                t_val, t_rec_pars = val.apply(
                    data=data,
                    recursion_pars=recursion_pars,
                    return_recursion_pars=True,
                    copy_data=False,
                    **params,
                )

                new_recursion_pars.update(t_rec_pars)

                data[val] = t_val  # Store in data for potential reuse

            elif val in keywords:
                t_val = keywords[val]

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
            evaluate_kwargs = evaluate_kwargs | {
                k: v for k, v in params.items() if not isinstance(v, Source)
            }

        # TODO: consider tracking used params and warning if any are unused

        # Evaluate
        eval_out = self.evaluate(*evaluate_args, **evaluate_kwargs)

        if self._use_memory and isinstance(eval_out, tuple):
            result, memory = eval_out
        else:
            result = eval_out
            memory = None

        new_recursion_pars[self] = memory

        # Store recursion pars if tracking enabled
        if track_state:
            self.recursion_pars = new_recursion_pars.copy()

        # If format is specified, apply it
        formatter = formatter or self.formatter
        if formatter is not None:
            result = formatter(data | {self: result})

        if return_recursion_pars:
            return result, new_recursion_pars
        else:
            return result

    def ancestors(self) -> list[Source]:
        """Get ancestor nodes."""
        # Recursively get the top most dependencies (sources)
        result = set()
        for dep in self.sources:
            if isinstance(dep, Transformation):
                for anc in dep.ancestors():
                    result.add(anc)
            elif dep not in KeywordSource.list_keywords():
                result.add(dep)
        return list(result)

    def evaluate(self) -> tuple[Any, Any] | Any:
        """Compute the transformation output.
    
        Subclasses should override this method with a custom signature
        matching the input sources specified in ``__init__``.
        
        Returns
        -------
        result : Any
            Transformation output or a tuple of (result, memory).
        memory : Any, optional
            Cached state for subsequent calls (if returning tuple).
        
        Raises
        ------
        NotImplementedError
            This method must be overridden by subclasses.
        
        Notes
        -----
        This method signature can be freely defined by subclasses.
        Parameters will be matched to input sources by name.
        """
        # In: signature can be freely defined by subclasses.
        # Out: result or tuple of (result, memory), where memory
        # should be passed to subsequent calls.

        raise NotImplementedError("This method should be overridden by subclasses")

    def print_dependency_tree(self, level=0):
        """Print the transformation dependency tree to stdout."""
        indent = "  " * level
        print(f"{indent}- {self}")
        for dep in self.dependencies:
            if isinstance(dep, Transformation):
                dep.print_dependency_tree(level + 1)
            else:
                print(f"{indent}  - {dep}")

    def get_all_dependencies(self) -> list[Transformation]:
        """Return all recursive transformation dependencies as a deduplicated list."""
        result = set()
        for dep in self.dependencies:
            result.add(dep)
            if isinstance(dep, Transformation):
                for sub_dep in dep.get_all_dependencies():
                    result.add(sub_dep)
        return list(result)


    def get_graph(self) -> dict:
        """Return a dict with nodes and edges rooted at this transformation."""
        result = {}

        result[self] = set(self.sources)

        for dep in self.dependencies:
            if isinstance(dep, Transformation):
                result = result | dep.get_graph()
        
        return result

    def get_summary(self, names = None):
        """Return a summary of the transformation graph and parameters."""
        graph = self.get_graph()

        # Make unique names for each node
        names = names or {}
        counts = {}
        missing = set()
        sources = set(graph.keys()) | set().union(*graph.values())
        for node in sources:
            if node not in names:
                if node._name is not None:
                    names[node] = node._name
                else:
                    counts[node.__class__] = counts.get(node.__class__, 0) + 1
                    missing.add(node)

        used_names = {}        
        for node in missing:
            if counts[node.__class__] > 1:
                used_names[node.__class__] = used_names.get(node.__class__, 0) + 1
                names[node] = f"{node.__class__.__name__}_{used_names[node.__class__]}"
            else:
                names[node] = node.__class__.__name__

        summary = {}
        for node, deps in graph.items():
            summary[names[node]] = {
                "class": node.__class__.__name__,
                "params": node._init_params,
                "sources": [names[dep] for dep in deps],
            }

        return summary

    def reset_state(self):
        """Clear cached recursion state stored on this transformation."""
        self.recursion_pars = None

    def __call__(self, *args, **kwargs):
        """Alias for :meth:`apply."""
        return self.apply(*args, **kwargs)


def parse_data(data: dict, ref=None, copy=True):
    """Normalize input into a source-value dict and ensure DEFAULT_SOURCE is present."""
    if not isinstance(data, dict):
        data = {DEFAULT_SOURCE: data}
    elif copy:
        data = data.copy()

    ref_val = data[ref or next(iter(data))]

    if DEFAULT_SOURCE not in data:
        data[DEFAULT_SOURCE] = ref_val

    return data


class Dim(Transformation):
    """Transformation that returns the size of a selected axis of the input."""

    def __init__(self, data, axis=1):
        """Initialize Dim transformation with input data and axis.
        
        Parameters        ----------
        data : Source
            Source that provides an array-like input.
        axis : int, default=1
            The axis along which to get the size.
        """
        self.axis = axis
        super().__init__(data=data)

    def evaluate(self, data):
        """Return the value of the data shape at the specified axis."""
        return data.shape[self.axis]

class GetItem(Transformation):
    """Transformation that returns ``data[key]`` from an input source."""
    
    def __init__(self, data: Source, key):
        """Initialize GetItem with input data and key."""
        super().__init__(data=data)
        self.key = key

    def evaluate(self, data):
        """Return the item from data corresponding to the key."""
        return data[self.key]


class GetAttr(Transformation):
    """Transformation that returns ``data.attr`` from an input source."""

    def __init__(self, data: Source, attr):
        """Initialize GetAttr with input data and attribute."""
        super().__init__(data=data)
        self.attr = attr

    def evaluate(self, data):
        """Return the attribute from data corresponding to the attribute name."""
        return getattr(data, self.attr)

class Apply(Transformation):
    """Transformation that applies a callable to source inputs."""

    def __init__(self, func, *args, **kwargs):
        """Initialize Apply transformation with a callable and its arguments.
        
        Parameters
        ----------
        func : callable
            The function to apply to the input sources.
        *args : Source or otherwise
            Positional arguments for the function, which can be Sources or fixed values.
        **kwargs : Source or otherwise
            Keyword arguments for the function, which can be Sources or fixed values.
        """
        self.func = func
        # Get args and kwargs that have Source values
        self.fixed_args = {
            i: arg for i, arg in enumerate(args) if not isinstance(arg, Source)
        }
        self.source_args = {
            i: arg for i, arg in enumerate(args) if isinstance(arg, Source)
        }
        self.n_args = len(args)
        self.fixed_kwargs = {
            k: arg for k, arg in kwargs.items() if not isinstance(arg, Source)
        }
        self.source_kwargs = {
            k: arg for k, arg in kwargs.items() if isinstance(arg, Source)
        }
        super().__init__(*self.source_args.values(), **self.source_kwargs)

    def evaluate(self, *args, **kwargs):
        """Apply the callable to fixed inputs and Source values."""
        # Combine fixed and source args and kwargs

        # Build args list by sorting the combined keys of fixed and source args
        all_args = [
            self.fixed_args[i] if i in self.fixed_args else args[i]
            for i in range(self.n_args)
        ]

        all_kwargs = kwargs | self.fixed_kwargs

        return self.func(*all_args, **all_kwargs)

def transform_wrapper(func):
    """Return a wrapper that makes Apply transformations."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        return Apply(func, self, *args, **kwargs)

    return wrapper

class _PrimitiveTransformation(Transformation):
    """Base class for primitive transformations supporting operator overloading."""

    def __init__(self, a, b):

        if not isinstance(a, Source):
            a = Param(a)

        if not isinstance(b, Source):
            b = Param(b)

        self.a, self.b = a, b

        super().__init__(a=a, b=b)

class _SumTransformation(_PrimitiveTransformation):
    

    def evaluate(self, a, b):
        return a + b


class _MulTransformation(_PrimitiveTransformation):

    def evaluate(self, a, b):
        return a * b


class _DivTransformation(_PrimitiveTransformation):

    def evaluate(self, a, b):
        return a / b


class _SubTransformation(_PrimitiveTransformation):

    def evaluate(self, a, b):
        return a - b


class _PowTransformation(_PrimitiveTransformation):

    def evaluate(self, a, b):
        return a**b

class _MatmulTransformation(_PrimitiveTransformation):

    def evaluate(self, a, b):
        return a @ b


class Param(Transformation):
    """Transformation that wraps a fixed parameter value as a Source."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self):
        """Return the parameter value."""
        return self.value

class CircularBuffer:
    """Circular buffer for storage and retrieval of fixed-length 2D arrays.

    This class implements a circular buffer data structure that stores and retrieves the
    most recent rows of data without requiring expensive array shifting operations. It
    maintains a fixed size and uses modular arithmetic to wrap around when the buffer is
    full.

    Parameters
    ----------
    size : int
        The maximum number of rows the buffer can store.
    m : int
        The number of columns in each row (width of the 2D array).
    default_value : float, optional
        The default value used to fill the buffer and for padding operations. Default is
        np.nan.

    Attributes
    ----------
    offset : int
        Current write position in the circular buffer.
    size : int
        Maximum number of rows the buffer can store.
    m : int
        Number of columns in each row.
    default_value : float
        Default fill value for the buffer.
    data : ndarray or None
        The underlying 2D array storing the buffer data (shape: (size, m)).
    index : ndarray
        Lazily computed index array for accessing the buffer in the correct order.

    Notes
    -----
    - The buffer uses a fixed-size 2D numpy array with shape (size, m).
    - Access patterns use an index array that is lazily computed and cached.
    - TODO: Consider generalizing to n-dimensional arrays.
    
    """

    def __init__(self, size, m, default_value=np.nan):
        self.offset = 0
        self.size = size
        self._range = np.arange(size)
        self._index = None
        self.m = m
        self.default_value = default_value
        self.data = None

    def set_data(self, dtype=np.float64, default_value=np.nan):
        """Reset the data using provided dtype and default value."""
        # Set data type of the buffer
        self.data = np.full((self.size, self.m), default_value, dtype=dtype)

    @property
    def index(self):
        """Lazily compute array for accessing the buffer in the correct order."""
        if self._index is None:
            self._index = np.roll(self._range, -self.offset)

        return self._index

    def append(self, value: np.ndarray):
        """Append new rows of data to the buffer, pushing out old data as needed."""
        self._index = None

        if self.data is None:
            self.set_data(dtype=value.dtype, default_value=self.default_value)

        if value.ndim == 1:
            value = value.reshape(-1, self.m)

        # Get only last size rows of value
        if value.shape[0] > self.size:
            value = value[-self.size :]

        n = value.shape[0]
        rem = self.size - self.offset
        i = self.offset + n
        self.data[self.offset : i] = value[:rem]

        self.offset = i % self.size

        if n > rem:
            self.data[: self.offset] = value[rem:n]

    def get_slice(self, start: int = None, end: int = None) -> np.ndarray:
        """Return a slice of the buffer from start to end, where end is exclusive."""
        indices = self.index[start:end]
        if self.data is None:
            n_return = len(indices)
            return np.full((n_return, self.m), self.default_value)
        return self.data[indices]

    def get(self, n: int) -> np.ndarray:
        """Return the n oldest rows of the buffer."""
        res = self.get_slice(end=n)
        if n > self.size:
            # Pad with default values
            pad = np.full((n - self.size, self.m), self.default_value)
            res = np.vstack([res, pad])
        elif n == 1:
            res = res.squeeze(0)
        return res

    def update(self, data: np.ndarray):
        """Append new data to the buffer and return the pushed-out data."""
        n = data.shape[0] if data.ndim > 1 else 1
        res = self.get(n)
        if n > self.size:
            res[self.size :] = data[: n - self.size]
        self.append(data)
        if data.ndim > 1:
            # Ensure output has same shape as input
            res = res.reshape(data.shape)
        return res

    def reset(self):
        """Reset the buffer data and index."""
        self.data.fill(np.nan)
        self._index = None
        self.offset = 0


_format_like_registry = {}

def register_format_like(source: type, target: type):
    """Return decorator to register a formatting function for use with 'format_like'.

    Parameters
    ----------
    source : type
        The source type to format.
    target : type
        The target type to format like.
    
    Returns
    -------
    decorator : function
        A decorator that registers a formatting function for use with 'format_like'. The
        registered function will be used to convert values of the source type to the
        target type when 'format_like' is called with value as the source type and
        reference value as the target type. The formatting function must have the 
        signature 'format_func(value, reference_value)'.
    """
    def decorator(format_func):
        sig = inspect.signature(format_func)
        # Check that format_func has a single argument
        if len(sig.parameters) != 2:
            raise ValueError(
                "Format function must have exactly two arguments: value and reference value."
            )

        @functools.wraps(format_func)
        def wrapped_format_func(val, ref_val):
            return format_func(val, ref_val)

        _format_like_registry[(source, target)] = wrapped_format_func
        return wrapped_format_func

    return decorator

def format_like(val, ref_val):
    """Format val like ref_val using registered formatting functions.
    
    The function uses multiple dispatch to find a registered formatting function to
    convert val to the type of ref_val.

    Parameters
    ----------
    val : Any
        The value to format.
    ref_val : Any
        The reference value to format like.
    
    Returns
    -------
    formatted_val : Any
        The formatted value.
    """
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
    """Base class for formatting transformation outputs.
    
    Formats are transformations that modify values according to a set of rules specified
    using formatting functions and resolvers. Each subclass of Format maintains a
    registry of such rules and uses multiple dispatch to find the appropriate formatting
    function.

    Subclasses can register formatting functions or resolvers. Formatting functions
    directly specify how the output of a given source type should be formatted for that
    given Format subclass. Resolvers enable more control, by instating an intermediate
    function that generates a formatting function based on the specific source instance,
    allowing instance specific formatting rules.

    Parameters
    ----------
    source : Source
        The input source (instance) to format.
    
    Methods
    -------
    get_formatter
        Get a formatter for given source instance using either registered formatting
        functions or resolvers.
    register
        Decorator to register a formatting function for a given source type.
    register_resolver
        Decorator to register a resolver function for a given source type.
    """

    def __init_subclass__(cls):
        """Initialise registries for the subclass."""
        # Make empty subclass registry
        cls.registry = {}
        cls.resolver_registry = {}
        super().__init_subclass__()

    def __init__(self, source: Source):
        self.source = source
        self._formatter = self.get_formatter(source)
        super().__init__(source, STATE, memory=MEMORY)
        # data needs to be a dependence since STATE needs to contain all dependencies

    @classmethod
    def find_formatter(cls, source):
        """Use multiple dispatch to find a registered formatter."""
        for base in type(source).__mro__:
            if base in cls.registry:
                return cls.registry[base]

    @classmethod
    def find_resolver(cls, source):
        """Use multiple dispatch to find a registered resolver."""
        for base in type(source).__mro__:
            if base in cls.resolver_registry:
                return cls.resolver_registry[base]

    @classmethod
    def get_formatter(cls, source):
        """Find or resolve a formatter function for given source."""
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

    def evaluate(self, source, state, memory=None):
        """Apply resolved formatting function to source."""
        # Note, source should stay in the evaluate method to ensure transformation treats it as a dependency
        return self._formatter(self.source, state, memory)

    @classmethod
    def check_formatter(cls, formatter):
        """Check that formatting function has correct signature."""
        sig = inspect.signature(formatter)
        if len(sig.parameters) != 3:
            raise ValueError(
                "Formatter function must have exactly three arguments: value_source, state, and memory."
            )

    @classmethod
    def register(cls, source_type: type):
        """Return decorator for registering function as formatter for source_type.

        The decorated function should take exactly three arguments, 
        (value_source, state, memory), where value_source is the source instance of the
        value to convert, state is a dict of known values of transforms and memory is 
        the format instance memory parameter, see Transformation for details on the two 
        latter arguments.

        Parameters
        ----------
        source_type : type
            The source type for which the formatting function should be registered.
        
        Returns
        -------
        decorator : function
            A decorator that registers a formatting function for the given source type.
        """
        def decorator(format_func):
            cls.check_formatter(format_func)
            cls.registry[source_type] = format_func
            return format_func

        return decorator

    @classmethod
    def register_resolver(cls, source_type: type):
        """Return decorator for registering function as resolver for source_type.

        The decorated function should take one argument, the source instance to resolve,
        and return a formatter function.

        Parameters
        ----------
        source_type : type
            The source type for which the resolver should be registered.

        Returns
        -------
        decorator : function
            A decorator that registers a resolver for the given source type.
        """
        def decorator(resolver_func):
            # Check that resolver_func has the correct signature
            sig = inspect.signature(resolver_func)
            if len(sig.parameters) != 1:
                raise ValueError(
                    "Resolver function must have exactly one argument: the source."
                )
            cls.resolver_registry[source_type] = resolver_func
            return resolver_func

        return decorator
