"""Selection of feature transformations.

This module provides a selection of feature transformations based on the transformation
class :class:`py_online_forecast.core.Transformation`. The transformations are focused
on use with online regression models.

Most transformations process observations along the first axis and preserve
state between calls when they are evaluated with ``track_states=True``.

"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .core import DEFAULT_SOURCE, CircularBuffer, Dim, Transformation

### Transformations
class ToArray(Transformation):
    """Transformation that converts input data to a NumPy array.

    Parameters
    ----------
    data : Transformation
        Input transformation providing array-like data.

    Notes
    -----
    Calling an instance returns the input converted with
    ``numpy.asarray``.
    """

    def __init__(self, data):
        super().__init__(lambda x: np.asarray(x), data)

class Combine(Transformation):
    """Combine multiple inputs into a dictionary.

    Parameters
    ----------
    *sources : Transformation
        Input transformations whose results should be combined.
    names : sequence, optional
        Keys for the resulting dictionary. If omitted, the input
        transformations themselves are used as keys.

    Raises
    ------
    ValueError
        If ``names`` does not contain one name for each input source.
    """

    def __init__(self, *sources, names = None):
        if names is not None:
            if len(names) != len(sources):
                raise ValueError("Length of names must match number of sources.")
        else:
            names = sources

        self.names = names
        super().__init__(self._evaluate, *sources)
    
    def _evaluate(self, *data):
        """Combine input data into a dictionary."""
        return {name: v for name, v in zip(self.names, data)}

class DesignMatrix(Transformation):
    """Transformation that constructs a two-dimensional design matrix.

    Each input is converted to an array. Its first axis represents
    observations, and all remaining axes are flattened before the inputs
    are concatenated horizontally.

    Parameters
    ----------
    *data : Transformation
        Input transformations providing array-like data.

    Notes
    -----
    Calling an instance returns an array with shape
    ``(n_samples, n_features)``.
    """

    def __init__(self, *data):
        data = [ToArray(d) for d in data]
        super().__init__(self._stack, *data)
    
    def _stack(self, *data):
        """Reshape input data and stack horizontally.
        
        Parameters
        ----------
        data : list of ndarray
            The input data to combine into a design matrix. Each element should have
            shape (n_samples, d_i) where n_samples is the number of observations and d_i
            is the number of features for the i-th data input. Higher dimensional arrays
            will be reshaped to 2D arrays, keeping the first dimension.

        Returns
        -------
        result : ndarray
            The combined design matrix with shape (n_samples, sum(D_1, ...)) where 
            D_i = d_i_1 * d_i_2 * ... is the number of features in the i-th input array.
        """
        data = [d.reshape(d.shape[0], -1) for d in data]
        return np.hstack(data)

class One(Transformation):
    """Transformation that creates an array of ones.

    Parameters
    ----------
    ref : Transformation, default=DEFAULT_SOURCE
        Transformation whose first-axis length determines the output length.
    shape : tuple, optional
        Additional output dimensions. If omitted, the output has shape
        ``(n_samples,)``. Otherwise, it has shape
        ``(n_samples,) + shape``.

    Notes
    -----
    Calling an instance returns an array filled with ones.
    """

    def __init__(self, ref = DEFAULT_SOURCE, shape = None):
        t = Dim(ref, axis = 0)
        self._shape = shape
        super().__init__(self._evaluate, t)

    def _evaluate(self, t):
        """Return a column of ones of length t."""
        if self._shape is None:
            return np.ones(t)
        else:
            return np.ones((t,) + self._shape)

class SlidingSum(Transformation):
    """Transformation that computes sliding-window sums.

    Parameters
    ----------
    data : Transformation, default=DEFAULT_SOURCE
        Input transformation providing a two-dimensional array. The first
        axis represents observations.
    window_size : int, default=1
        Number of observations included in each sliding window.
    *args
        Additional positional arguments passed to
        ``sliding_window_view``.
    **kwargs
        Additional keyword arguments passed to
        ``sliding_window_view``.

    Notes
    -----
    The transformation uses a state to include observations from previous
    input batches when computing the sliding sums.
    """

    def __init__(self, data = DEFAULT_SOURCE, window_size = 1, *args, **kwargs):
        self.window_size = window_size
        data = ToArray(data)
        super().__init__(self._evaluate, data = data, STATE = "old_data")
        self._args = args
        self._kwargs = kwargs

    def _evaluate(self, data, old_data = None):
        """Evaluate the sliding window sum.
        
        Parameters
        ----------
        data : ndarray of shape (n_samples, ...)
            The input data to compute the sliding sum over.
        old_data : ndarray
            Previous values of the input data to complete the sliding view.

        Returns
        -------
        result : ndarray
            The sliding window sum of the input data.
        """
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
    """Transformation that computes sliding-window means.

    Parameters
    ----------
    data : Transformation
        Input transformation providing a two-dimensional array. The first
        axis represents observations.
    window_size : int
        Number of observations included in each sliding window.
    *args
        Additional positional arguments passed to
        ``sliding_window_view``.
    **kwargs
        Additional keyword arguments passed to
        ``sliding_window_view``.

    Notes
    -----
    The transformation uses a state to include observations from previous
    input batches when computing the sliding means.
    """

    def __init__(self, data = None, window_size = None, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.window_size = window_size
        data = ToArray(data)
        super().__init__(self._evaluate, data = data, STATE = "old_data")

    def _evaluate(self, data, old_data = None):
        """Evaluate the sliding window mean.
        
        Parameters
        ----------
        data : ndarray of shape (n_samples, ...)
            The input data to compute the sliding mean over.
        old_data : ndarray
            Previous values of the input data to complete the sliding view.

        Returns
        -------
        result : ndarray
            The sliding window mean of the input data.

        """
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
    """Compute the running mean with exponential forgetting.
    
    Parameters
    ----------
    forgetting : float
        The forgetting factor for the exponential forgetting.
    data : ndarray of shape (n_samples, ...)
        The input data to compute the running mean over.
    state : tuple
        The previous state of the computation, i.e. a tuple of the most recent mean
        estimate and the effective memory.
    track_memory : bool
        Whether to track the effective or assume saturated memory. If  True, the memory
        is updated as memory = forgetting * memory + 1 at each step. Otherwise, the
        memory is assumed to be saturated, i.e. 1/(1 - forgetting).

    Returns
    -------
    result : ndarray of shape (n_samples, ...)
        The running mean of the input data.
    new_state : tuple
        The new state of the running mean and memory.
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    # Remove observations with nans
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
    """Transformation that computes a running mean.

    The running mean is computed with exponential forgetting and ignores observations
    containing NaN values.

    Parameters
    ----------
    forgetting : float
        Forgetting factor applied between observations. Larger values give
        greater weight to previous observations.
    data : Transformation, default=DEFAULT_SOURCE
        Input transformation providing observations along the first axis.
    track_memory : bool, default=True
        If True, track the effective memory explicitly. If False, assume
        that the effective memory is saturated.

    Notes
    -----
    The transformation uses a state to continue the running mean across
    separate input batches.
    """

    def __init__(self, forgetting, data = DEFAULT_SOURCE, track_memory = True):
        super().__init__(self._evaluate, data = data, forgetting = forgetting, STATE = "state")
        self.track_memory = track_memory
        self.data = data

    def _evaluate(self, data, forgetting, state = None):
        """Evaluate the running mean using ``forgetting_mean``.
        
        Parameters
        ----------
        data : ndarray of shape (n_samples, ...)
            The input data to compute the running mean over.

        """
        return forgetting_mean(forgetting, data, state, self.track_memory)

class ForgettingVariance(Transformation):
    """Transformation that computes running variance estimates.

    The calculation uses exponential forgetting and can optionally center
    observations using a running mean.

    Parameters
    ----------
    forgetting : float
        Forgetting factor applied between observations.
    data : Transformation, default=DEFAULT_SOURCE
        Input transformation providing observations with shape
        ``(n_samples, n_features)``.
    track_memory : bool, default=True
        If True, track the effective memory explicitly. If False, assume that the
        effective memory is saturated.
    center : bool or Transformation, default=True
        Whether to center the observations before computing their variance. A 
        transformation may be supplied to provide a custom center.
    covariance : bool, default=False
        If True, compute a full covariance matrix. Otherwise, compute one variance value
        for each feature.

    Notes
    -----
    Calling an instance returns running variance estimates with shape
    ``(n_samples, n_features)`` when ``covariance=False``. When ``covariance=True``, it
    returns estimates with shape ``(n_samples, n_features, n_features)``.

    The transformation uses a state to continue the variance or covariance
    calculation across separate input batches.
    """

    def __init__(self, forgetting, data = DEFAULT_SOURCE, track_memory = True, center = True, covariance = False):
        if center:
            if isinstance(center, Transformation):
                mean = center
            else:
                mean = ForgettingMean(forgetting, data = data, track_memory = track_memory)
            super().__init__(self._evaluate, data, forgetting, mean = mean, STATE = "state")
        else:
            super().__init__(self._evaluate, data, forgetting, STATE = "state")

        self.track_memory = track_memory
        self.covariance = covariance

    def _evaluate(self, data, forgetting, mean = None, state = None):
        """Evaluate the running variance or covariance.
        
        Parameters
        ----------
        data : ndarray of shape (n_samples, d)
            The input data to compute the running variance over.
        mean : ndarray of shape (n_samples, d)
            The running mean estimate to center the data. If None, the data is not
            centered.
        state : tuple
            The previous state of the computation, i.e. a tuple of the most recent
            variance estimate and the effective memory.
        forgetting : float
            The forgetting factor for the exponential forgetting. If None, method uses
            the forgetting factor specified at initialization.
        
        Returns
        -------
        result : ndarray of shape (n_samples, d) or (n_samples, d, d)
            The running variance or covariance of the input data.
        new_state : tuple
            The new state of the running variance and memory.
        """
        if self.covariance:
            return self._eval_covariance(data, mean, state, forgetting)
        else:
            return self._eval(data, mean, state, forgetting)

    def _eval_covariance(self, data, mean = None, state = None, forgetting = None):
        return self._evaluate(data, forgetting, mean, state)
    
    def _eval(self, data, mean = None, state = None, forgetting = None):
        return self._evaluate(data, forgetting, mean, state)

    def _evaluate(self, data, forgetting, mean = None, state = None):

        # Center estimate if mean is provided
        if mean is not None:
            data = data - mean

        # Compute unentered variance
        if self.covariance:
            data = np.einsum('ij,ik->ijk', data, data)
        else:
            data = data**2

        var, state = forgetting_mean(forgetting, data, state, self.track_memory)

        return var, state
    
class LowPass(Transformation):
    """Transformation that applies an exponential low-pass filter.

    NaN values are replaced using the current filtered value or, when
    necessary, the previous filtered value.

    Parameters
    ----------
    data : Transformation
        Input transformation providing a two-dimensional array. The first axis 
        represents observations.
    alpha : float, default=0
        Weight assigned to the previous filtered value. Larger values produce stronger
        smoothing.

    Notes
    -----
    The transformation uses a state to continue filtering across separate
    input batches.
    """

    def __init__(self, data, alpha = 0):
        data = ToArray(data)
        super().__init__(self._evaluate, data=data, STATE = "prev_value")
        self.alpha = alpha

    def _evaluate(self, data: np.ndarray, prev_value=None):
        """Evaluate the low-pass filter on the input data.
        
        Parameters
        ----------
        data : ndarray of shape (n_samples, ...)
            The input data to apply the low-pass filter to.
        prev_value : ndarray of shape (...,)
            The previous value of the low-pass filter to resume the computation. If
            None, it is initialized as NaN.

        Returns
        -------
        result : ndarray
            The low-pass filtered data.
        """
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
    """Transformation that generates Fourier-series features.

    The input is interpreted as observations along the first axis. All remaining input
    dimensions are flattened before sine and cosine features are generated.

    Parameters
    ----------
    data : Transformation
        Input transformation providing the reference values.
    nharmonics : int, default=1
        Number of harmonics to include.

    Notes
    -----
    Calling an instance returns an array with shape
    ``(n_samples, 2 * nharmonics * n_features)``. For each harmonic, cosine features are
    followed by sine features.
    """

    def __init__(self, data, nharmonics = 1):
        self.nharmonics = nharmonics
        data = ToArray(data)
        super().__init__(self._evaluate, data = data)

    def _evaluate(self, data: np.ndarray):
        """Evaluate the Fourier series features.

        Parameters
        ----------
        data : ndarray of shape (n_samples, ...)
            The input data to compute the Fourier features from. The data is reshaped to
            (n_samples, d), by collapsing all dimensions except the first one.

        Returns
        -------
        result : ndarray of shape (n_samples, 2*nharmonics*d)
            The Fourier series features of the input data.
        """
        results = []
        if data.ndim != 2:
            data = data.reshape(data.shape[0], -1)
        for i in range(1, self.nharmonics + 1):
            results.append(np.cos(2*np.pi*i*data))
            results.append(np.sin(2*np.pi*i*data))
    
        result = np.hstack(results)
        return result

class Lag(Transformation):
    """Transformation that lags observations along the first axis.

    Parameters
    ----------
    data : Transformation
        Input transformation providing an array. The first axis represents observations.
    amount : int, default=1
        Number of observations by which to lag the input.
    default_value : scalar, optional
        Value used where insufficient history is available. Defaults to NaN.
    offsets : int or sequence of int, optional
        Additional offsets to evaluate. For an offset ``h``, the effective lag is 
        ``amount - h``.

    Notes
    -----
    Calling an instance returns an array containing the lagged observations.
    If ``offsets`` is provided, it returns a dictionary mapping each offset
    to its corresponding lagged array.

    The transformation uses a state to provide observations from previous
    input batches when constructing lagged values.
    """

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
        super().__init__(self._evaluate, data = data, STATE = "prev_values")


    def _evaluate(self, data, prev_values = None):
        """Lag the data along the first dimension.

        Parameters
        ----------
        data : ndarray of shape (n_samples, ...) or dict of such arrays
            The input data to lag.
        prev_values : ndarray of shape (n_samples, ...) or dict of such arrays, optional
            The previous values to use for the lag.

        Returns
        -------
        result : ndarray of shape (n_samples, ...) or dict
            The lagged data. If the input was a dict, the lagged value is also a dict
            where each value has been lagged. If offsets were specified at 
            initialization, the result is a dict of lagged values per offset. If both
            the input was a dict, and offsets were specified, the output is a dict of
            dicts, where the first level corresponds to offsets, and the second level
            corresponds to the keys of the input dict.
        prev_values : ndarray of shape (n_samples, ...)
            The updated previous values.

        """
        # Evaluate lag across horizons

        # No specified offset
        if self.offsets is None:
            return self._evaluate_offset(data, prev_values, None)

        # A set of offsets
        else:
            
            # Initialize prev_values per offset
            if prev_values is None:
                prev_values = {h: None for h in self.offsets}
            result = {}

            # Evaluate per offset
            for h in self.offsets:
                result[h], prev_values[h] = self._evaluate_offset(data, prev_values[h], h)

            return result, prev_values

    def _evaluate_offset(self, data, prev_values = None, offset = None):
        if isinstance(data, np.ndarray):
            return self._evaluate_array(data, prev_values, offset)
        elif isinstance(data, dict):
            return self._evaluate_dict(data, prev_values, offset)
        else:
            raise ValueError(f"Cannot apply Lag to data of type {type(data)}.")

    def _evaluate_dict(self, data, prev_values = None, offset = None):
        if prev_values is None:
            prev_values = {k: None for k in data}

        result = {}
        for k, v in data.items():
            result[k], prev_values[k] = self._evaluate_offset(v, prev_values[k], offset)

        return result, prev_values

    def _evaluate_array(self, data, prev_values = None, offset = None):

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
    
class Periodic(Transformation):
    """Transformation that maps indices to periodically repeating values.

    Parameters
    ----------
    values : ndarray
        Precomputed values. The first axis represents the periodic cycle.
    index : Transformation
        Transformation providing integer reference indices.
    horizons : sequence of int, optional
        Integer offsets to apply to each reference index.

    Notes
    -----
    Calling an instance maps each index using modulo arithmetic, so indices
    outside the range of ``values`` wrap around the periodic cycle.

    Without ``horizons``, the output has the shape of ``values[index]``.
    With ``horizons``, the output includes a horizon axis after the index
    axis.
    """

    def __init__(self, values: np.ndarray, index: Transformation, horizons: list = None):
        self.values = values

        self.horizons = horizons
        self._n = len(values)
        
        super().__init__(self._evaluate, index)

    def _evaluate(self, index: np.ndarray) -> np.ndarray:
        """Return values mapped from reference_time.
        
        Parameters
        ----------
        reference_time: np.ndarray
            Array of shape (t,) containing the times for which to map values.
        
        Returns
        -------
        result: np.ndarray
            Array of shape (t, len(horizons), ...) containing the mapped values for each
            time and horizon.
        """
        # Map reference_time to values using times as reference, applying offsets
        index = index % self._n
        
        if self.horizons is None:
            return self.values[index]

        result = []
        for h in self.horizons:
            shifted_index = (index + h) % self._n
            result.append(self.values[shifted_index])

        return np.stack(result, axis=1)
    
class Difference(Transformation):
    """Transformation that computes first differences.

    Parameters
    ----------
    data : Transformation
        Input transformation providing an array. The first axis represents
        observations.

    Notes
    -----
    The transformation uses a state to compute differences continuously
    across separate input batches.
    """

    def __init__(self, data):
        data = ToArray(data)
        super().__init__(self._evaluate, data = data, STATE = "prev_value")

    def _evaluate(self, data, prev_value = None):
        """Return the difference along the first dimension of the input data."""
        if prev_value is None:
            prev_value = np.full(data.shape[1:], np.nan)
        data = np.vstack((prev_value, data))
        diff = np.diff(data, axis = 0)
        new_prev_value = data[-1]
        return diff, new_prev_value
    
class Map(Transformation):
    """Transformation that applies another transformation to multiple inputs.

    Parameters
    ----------
    transformation : Transformation
        Transformation to apply independently for each input.
    *mapping : Transformation or dict
        An iterable of input mappings. Each mapping should be either a dict or a 
        Transformation. If a dict, each key should be a dependency of the 
        transformation, and the corresponding value should be the source to use as
        input for that dependency. If a Transformation, it is assumed to be the only
        dependency (DEFAULT_SOURCE) for the transformation.
    names : sequence of str, optional
        Names for the outputs. If provided, outputs are returned as a dict mapping each
        name to the corresponding output. If omitted, outputs are returned as a list.

    Raises
    ------
    ValueError
        If a mapping is neither a transformation nor a dictionary, or if
        ``names`` does not contain one name per mapping.

    Notes
    -----
    Calling an instance evaluates ``transformation`` separately for each
    mapping. Each mapping is evaluated using its own state.
    """
    
    def __init__(self, transformation, *mapping, names = None):

        if names is not None and len(names) != len(mapping):
            raise ValueError("Length of names must match number of mappings.")
        self.names = names

        standard_mapping = []
        for m in mapping:
            if isinstance(m, Transformation):
                standard_mapping.append({DEFAULT_SOURCE: m})
            elif isinstance(m, dict):
                standard_mapping.append(m)
            else:
                raise ValueError(
                    "Mapping should be either a dict or a Transformation, "
                    f"got {type(m)}."
                )

        sources = set()
        for m in standard_mapping:
            sources.update(m.values())
            
        self.transform_sources = list(sources)
        self.mapping = standard_mapping
        super().__init__(self._evaluate, *self.transform_sources, STATE = "states")
        self.transformation = transformation
    
    def _evaluate(self, *data, states = None):
        """Apply the transformation to the input data."""
        results = []
        if states is None:
            states = [None] * len(self.mapping)

        for i, m in enumerate(self.mapping):
            mapped_data = {dep: data[self.transform_sources.index(src)] for dep, src in m.items()}
            result, state = self.transformation(mapped_data, states = states[i], return_states=True)
            results.append(result)
            states[i] = state
        
        if self.names is not None:
            results = {name: res for name, res in zip(self.names, results)}

        return results, states



