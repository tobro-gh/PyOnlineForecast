"""Selection of feature transformations.

This module provides a selection of feature transformations based on the core module
Transformation class. The transformations are focused on use with online regression
models.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .core import DEFAULT_SOURCE, MEMORY, CircularBuffer, Dim, Transformation


### Transformations
class ToArray(Transformation):
    """Transformation that converts input data to a numpy array."""

    def __init__(self, data):
        super().__init__(data = data)

    def evaluate(self, data):
        """Convert input data to a numpy array."""
        if isinstance(data, np.ndarray):
            return data
        return np.asarray(data)

class Combine(Transformation):
    """Transformation that combines multiple input sources into a dictionary.
    
    Parameters
    ----------
    sources : list of Source
        The input sources to combine into a dictionary.
    names : list, optional
        The names to use for the combined features in the output dictionary. If not 
        provided, the names of the input sources will be used.
    """

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
        """Combine input data into a dictionary."""
        return {name: v for name, v in zip(self.names, data)}

class DesignMatrix(Transformation):
    """Transformation that combines multiple input sources into a 2D design matrix.

    Parameters
    ----------
    sources : list of Source
        Input sources with array like values to combine into a design matrix.
    """

    def __init__(self, *data):
        data = [ToArray(d) for d in data]
        super().__init__(*data)
    
    def evaluate(self, *data):
        """Reshape input data and stack horizontally.
        
        Parameters
        ----------
        data : list of ndarray
            The input data to combine into a design matrix. Each element should have
            shape (n_obs, d_i) where n_obs is the number of observations and d_i is the
            number of features for the i-th data input. Higher dimensional arrays will
            be reshaped to 2D arrays, keeping the first dimension.

        Returns
        -------
        result : ndarray
            The combined design matrix with shape (n_obs, sum(D_1, ...)) where 
            D_i = d_i_1 * d_i_2 * ... is the number of features in the i-th input array.
        """
        data = [d.reshape(d.shape[0], -1) for d in data]
        return np.hstack(data)

class One(Transformation):
    """Return a column of ones.

    Parameters
    ----------
    ref : Source, optional
        The source to use as reference for the length of the output column.
    """

    def __init__(self, ref = DEFAULT_SOURCE):
        t = Dim(ref, axis = 0)
        super().__init__(t)

    def evaluate(self, t):
        """Return a column of ones of length t."""
        return np.ones(t)

class SlidingSum(Transformation):
    """Compute sliding window sum using numpy's sliding_window_view function.

    Parameters
    ----------
    data : Source
        The input data to compute the sliding sum over.
    window_size : int
        The size of the sliding window to compute the sum over.
    *args, **kwargs
        Additional arguments to pass to numpy's sliding_window_view function
    """

    def __init__(self, data = DEFAULT_SOURCE, window_size = 1, *args, **kwargs):
        self.window_size = window_size
        data = ToArray(data)
        super().__init__(data = data, old_data = MEMORY)
        self._args = args
        self._kwargs = kwargs

    def evaluate(self, data, old_data = None):
        """Evaluate the sliding window sum.
        
        Parameters
        ----------
        data : ndarray of shape (n_obs, ...)
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
    """Compute sliding window mean using numpy's sliding_window_view function.

    Parameters
    ----------
    data : Source
        The input data to compute the sliding mean over.
    window_size : int
        The size of the sliding window to compute the mean over.
    *args, **kwargs
        Additional arguments to pass to numpy's sliding_window_view function
    """

    def __init__(self, data = None, window_size = None, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.window_size = window_size
        data = ToArray(data)
        super().__init__(data = data, old_data = MEMORY)

    def evaluate(self, data, old_data = None):
        """Evaluate the sliding window mean.
        
        Parameters
        ----------
        data : ndarray of shape (n_obs, ...)
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
    data : ndarray of shape (n_obs, ...)
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
    result : ndarray of shape (n_obs, ...)
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
    """Compute the running mean with exponential forgetting.
    
    This is simply a wrapper around the `forgetting_mean` function, which manages the
    state and forgetting parameter. See the `forgetting_mean` function for details.

    Parameters
    ----------
    forgetting : float
        The forgetting factor for the exponential forgetting.
    data : Source
        The input data to compute the running mean over.
    track_memory : bool
        Whether to track the effective or assume saturated memory.
    """

    def __init__(self, forgetting, data = DEFAULT_SOURCE, track_memory = True):
        super().__init__(data, state = MEMORY)
        self.forgetting = forgetting
        self.track_memory = track_memory
        self.data = data

    def evaluate(self, data, state = None):
        """Evaluate the running mean using `forgetting_mean`.
        
        Parameters
        ----------
        data : ndarray of shape (n_obs, ...)
            The input data to compute the running mean over.

        """
        return forgetting_mean(self.forgetting, data, state, self.track_memory)

class ForgettingVariance(Transformation):
    """Compute the running variance with exponential forgetting.
    
    This transformation computes a running variance estimate with exponential forgetting
    by applying the `forgetting_mean` function to the optionally centered squared data.
    The transformation can compute either the marginal variance or the full covariance
    matrix. 

    Parameters
    ----------
    forgetting : float
        The forgetting factor for the exponential forgetting.
    data : Source
        The input data to compute the running variance over. The value of the source
        should be array-like with shape (t, d) where t is the time dimension.
    track_memory : bool
        Whether to track the effective or assume saturated memory.
    center : bool
        Whether to center the data before computing the variance.
    covariance : bool
        Whether to compute the covariance matrix instead of the variance.
    """

    def __init__(self, forgetting, data = DEFAULT_SOURCE, track_memory = True, center = True, covariance = False):
        if center:
            if isinstance(center, Transformation):
                mean = center
            else:
                mean = ForgettingMean(forgetting, data = data, track_memory = track_memory)
            super().__init__(mean = mean, data = data, state = MEMORY)
        else:
            super().__init__(data = data, state = MEMORY)

        self.forgetting = forgetting
        self.track_memory = track_memory
        self.covariance = covariance

    def evaluate(self, data, mean = None, state = None, forgetting = None):
        """Evaluate the running variance or covariance.
        
        Parameters
        data : ndarray of shape (n_obs, d)
            The input data to compute the running variance over.
        mean : ndarray of shape (n_obs, d)
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
        result : ndarray of shape (n_obs, d) or (n_obs, d, d)
            The running variance or covariance of the input data.
        new_state : tuple
            The new state of the running variance and memory.
        """
        if self.covariance:
            return self._eval_covariance(data, mean, state, forgetting)
        else:
            return self._eval(data, mean, state, forgetting)

    def _eval_covariance(self, data, mean = None, state = None, forgetting = None):
        return self._evaluate(data, mean, state, forgetting)
    
    def _eval(self, data, mean = None, state = None, forgetting = None):
        return self._evaluate(data, mean, state, forgetting)

    def _evaluate(self, data, mean = None, state = None, forgetting = None):

        forgetting = forgetting or self.forgetting

        # Compute unentered variance
        if self.covariance:
            data = np.einsum('ij,ik->ijk', data, data)
        else:
            data = data**2

        var, state = forgetting_mean(forgetting, data, state, self.track_memory)

        # Center estimate if mean is provided
        if mean is not None:
            if self.covariance:
                mean_outer = np.einsum('ij,ik->ijk', mean, mean)
                var = var - mean_outer
            else:
                mean_sq = mean**2
                var = var - mean_sq

        return var, state
    
class LowPass(Transformation):
    """Low-pass filter using exponential forgetting.
    
    This transformation applied a low-pass filter with exponential forgetting and NaN
    handling to the input data.

    Parameters
    ----------
    data : Source
        The input data to apply the low-pass filter to.
    alpha : float
        The forgetting factor for the exponential forgetting. Higher values of alpha
        correspond to more smoothing.        
    """

    def __init__(self, data, alpha = 0):
        data = ToArray(data)
        super().__init__(data=data, prev_value = MEMORY)
        self.alpha = alpha

    def evaluate(self, data: np.ndarray, prev_value=None):
        """Evaluate the low-pass filter on the input data.
        
        Parameters
        ----------
        data : ndarray of shape (n_obs, ...)
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
    """Fourier series features.
    
    The transformation computes the basis functions of a Fourier series expansion up to
    a specified number of harmonics.

    Parameters
    ----------
    data : Source
        The input data to compute the Fourier features from.
    nharmonics : int
        The number of terms in the Fourier series expansion.
    """

    def __init__(self, data, nharmonics = 1):
        self.nharmonics = nharmonics
        data = ToArray(data)
        super().__init__(data = data)

    def evaluate(self, data: np.ndarray):
        """Evaluate the Fourier series features.

        Parameters
        ----------
        data : ndarray of shape (n_obs, ...)
            The input data to compute the Fourier features from. The data is reshaped to
            (n_obs, d), by collapsing all dimensions except the first one.

        Returns
        -------
        result : ndarray of shape (n_obs, 2*nharmonics*d)
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
    """Lag features along the observations dimension.
    
    This transformation lags arrays and dictionaries of arrays along the first
    dimension.

    Parameters
    ----------
    data : Source
        The input data to compute the lag features from.
    amount : int
        The number of steps to lag the data.
    default_value : scalar, optional
        The value to use for the lagged values when there is not enough history. If
        None, the default value is NaN.
    offsets : int or list of int, optional
        A set of offsets to lag from. If specified, evaluation of the transformation
        will return a dictionary of lagged values per offset. The lag amount for the
        i-th offset is given by amount - offsets[i]. If None, the transformation will
        simply lag by the specified amount and return a single array of lagged values.
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
        super().__init__(data = data, prev_values = MEMORY)


    def evaluate(self, data, prev_values = None):
        """Lag the data along the first dimension.

        Parameters
        ----------
        data : ndarray of shape (n_obs, ...) or dict of such arrays
            The input data to lag.
        prev_values : ndarray of shape (n_obs, ...) or dict of such arrays, optional
            The previous values to use for the lag.

        Returns
        -------
        result : ndarray of shape (n_obs, ...) or dict
            The lagged data. If the input was a dict, the lagged value is also a dict
            where each value has been lagged. If offsets were specified at 
            initialization, the result is a dict of lagged values per offset. If both
            the input was a dict, and offsets were specified, the output is a dict of
            dicts, where the first level corresponds to offsets, and the second level
            corresponds to the keys of the input dict.
        prev_values : ndarray of shape (n_obs, ...)
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