"""Core functionality for sources and transformations.

The module provides the basic objects to define data pipelines using sources and
transformations. The main components of the module are,

- Source: a placeholder for data sources
- Transformation: a base class for data transformations.

Sources and transformations can be composed to build a computational graph which is
lazily evaluating using the ``apply`` or ``__call__`` methods.

"""
#%%
from __future__ import annotations

import functools
import inspect
from typing import Any
from collections import defaultdict

import numpy as np




def _iterate(*iterative_args, state_arg = None):
    """Decorator to wrap a function to be applied iteratively over input data."""

    def decorator(function):

        # Insepct the function signature
        sig = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # Bind the arguments to the function signature
            bound_args = sig.bind(*args, **kwargs)

            # Extract the iterated arguments from the bound arguments
            iterated_args = [bound_args.arguments[arg] for arg in iterative_args]

            # Check that all iterated arguments have the same length
            lengths = [len(arg) for arg in iterated_args]
            if len(set(lengths)) > 1:
                raise ValueError("All iterated arguments must have the same length.")

            # Apply the function iteratively over the input data
            results = []
            state = None
            for i in range(lengths[0]):
                # Update the bound arguments with the current iteration's values
                for j, arg in enumerate(iterative_args):
                    bound_args.arguments[arg] = iterated_args[j][i]

                # Update state if the function has a state argument
                if state_arg is not None and state is not None:
                    bound_args.arguments[state_arg] = state

                # Call the function with the current iteration's arguments
                if state_arg is not None:
                    result, state = function(*bound_args.args, **bound_args.kwargs)
                else:
                    result = function(*bound_args.args, **bound_args.kwargs)

                results.append(result)

            if state_arg is not None:
                return results, state
            else:
                return results
         
        return wrapper

    return decorator

class Transformation:
    
    def __init__(self, function, *args, STATE = None, ITERATE = None, **kwargs):

        # Inspect the function signature and bind the provided arguments
        self._sig = inspect.signature(function)

        self._bound = self._sig.bind(*args, **kwargs)

        dependencies = []

        # Add args
        for arg in self._bound.args:
            if isinstance(arg, Transformation) and arg not in dependencies:
                dependencies.append(arg)

        # Add kwargs
        for arg in self._bound.kwargs.values():
            if isinstance(arg, Transformation) and arg not in dependencies:
                dependencies.append(arg)

        self.dependencies = list(dependencies)

        # Wrap the function if some variables are to be iterated over.
        if ITERATE is not None:
            if not isinstance(ITERATE, (list, tuple)):
                ITERATE = (ITERATE,)
            for arg in ITERATE:
                if arg not in self._sig.parameters:
                    raise ValueError(f"ITERATE argument '{arg}' not found in function signature.")

        self._function = _iterate(*ITERATE, state_arg=STATE)(function) if ITERATE is not None else function
        self._STATE = STATE

        # Check that STATE is in signature arguments
        if STATE is not None and STATE not in self._sig.parameters:
            raise ValueError(f"STATE argument '{STATE}' not found in function signature.")

        descendants = set()

        stack = list(self.dependencies)

        while stack:
            node = stack.pop()

            if node not in descendants:
                descendants.add(node)
                stack.extend(node.dependencies)

        self._descendants = descendants

        self._parents = defaultdict(set)
        for parent in self._descendants | {self}:
            for child in parent.dependencies:
                self._parents[child].add(parent)

        self.states = None

    
    def _apply(self, data, parents = None):

        # Evaluate all dependencies first
        for dep in self.dependencies:
            if dep not in data:
                dep._apply(data, parents)

        # Bind args
        args = [data[arg] if isinstance(arg, Transformation) else arg for arg in self._bound.args]

        # Bind kwargs
        kwargs = {k: (data[v] if isinstance(v, Transformation) else v) for k, v in self._bound.kwargs.items()}

        # Replace STATE argument with current state if it is in the signature
        if self._STATE is not None:
            kwargs[self._STATE] = data[STATE][self]

        # Then evaluate this transformation and update data and state
        result = self._function(*args, **kwargs)
        if self._STATE is not None:
                data[self], data[STATE][self] = result
        else:
            data[self] = result

        # Remove data for dependencies if parents is not None
        if parents is not None:
            for dep in self.dependencies:
                # If all parents of dep have been evaluated, remove dep from data
                if all(parent in data for parent in parents[dep]):
                    del data[dep]

    def __call__(self, data, track_states = False, states = None, return_states = False):
        """Apply the transformation to the input data."""

        # Parse data
        if not isinstance(data, dict):
            data = {DEFAULT_SOURCE: data}

        else:
            data = data.copy()

        if DEFAULT_SOURCE not in data:
            data[DEFAULT_SOURCE] = data

        if self in data:
            return data[self]

        data[STATE] = defaultdict(lambda: None)

        # Update state with existing state if tracking
        if states is not None:
            data[STATE].update(states)
        elif track_states:
            data[STATE].update(self.states or {})

        self._apply(data, self._parents)

        # Update own state if tracking
        if track_states:
            self.states = data[STATE]

        if return_states:
            return data[self], data[STATE]

        return data[self]

    def reset_state(self):
        """Reset the state of the transformation."""
        self.states = None

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
    
    def print_dependency_tree(self, level=0):
        """Print the transformation dependency tree to stdout."""
        indent = "  " * level
        print(f"{indent}- {self}")
        for dep in self.dependencies:
            dep.print_dependency_tree(level + 1)


class Data(Transformation):

    def __init__(self):
        super().__init__(lambda: None)

class Param(Transformation):
    
    def __init__(self, value):
        self.value = value
        super().__init__(lambda: self.value)


def _get_default_source():
    return DEFAULT_SOURCE


def _get_state():
    return STATE


class _DefaultSource(Data):
    def __reduce__(self):
        return _get_default_source, ()


class _State(Data):
    def __reduce__(self):
        return _get_state, ()


DEFAULT_SOURCE = _DefaultSource()
STATE = _State()

class _SumTransformation(Transformation):
    
    def __init__(self, a, b):
        super().__init__(lambda x, y: x + y, a, b)

class _SubTransformation(Transformation):
    
    def __init__(self, a, b):
        super().__init__(lambda x, y: x - y, a, b)

class _MulTransformation(Transformation):
    
    def __init__(self, a, b):
        super().__init__(lambda x, y: x * y, a, b)

class _DivTransformation(Transformation):
    
    def __init__(self, a, b):
        super().__init__(lambda x, y: x / y, a, b)

class _PowTransformation(Transformation):
    
    def __init__(self, a, b):
        super().__init__(lambda x, y: x ** y, a, b)

class _MatmulTransformation(Transformation):

    def __init__(self, a, b):
        super().__init__(lambda x, y: x @ y, a, b)

class GetItem(Transformation):
    
    def __init__(self, a, key):
        super().__init__(lambda x: x[key], a)
        self.key = key

class GetAttr(Transformation):
    
    def __init__(self, a, name):
        super().__init__(lambda x: getattr(x, name), a)
        self.name = name

#%%

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
        super().__init__(lambda x: x.shape[axis], data)

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


