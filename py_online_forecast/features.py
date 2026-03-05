from .core import *
from numpy.lib.stride_tricks import sliding_window_view

### Transformations
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

class One(Transformation):
    
    def __init__(self, ref = DEFAULT_SOURCE):
        t = Dim(ref, axis = 0)
        super().__init__(t)

    def evaluate(self, t):
        return np.ones(t)

class SlidingSum(Transformation):

    def __init__(self, data = DEFAULT_SOURCE, window_size = 1, *args, **kwargs):
        self.window_size = window_size
        data = ToArray(data)
        super().__init__(data = data, old_data = MEMORY)
        self._args = args
        self._kwargs = kwargs

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
        data = ToArray(data)
        super().__init__(data = data, old_data = MEMORY)

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
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

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

    def __init__(self, forgetting, data = DEFAULT_SOURCE, track_memory = True):
        super().__init__(data, state = MEMORY)
        self.forgetting = forgetting
        self.track_memory = track_memory
        self.data = data

    def evaluate(self, data, state = None):
        return forgetting_mean(self.forgetting, data, state, self.track_memory)

class ForgettingVariance(Transformation):

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
    def __init__(self, data, alpha = 0):
        data = ToArray(data)
        super().__init__(data=data, prev_value = MEMORY)
        self.alpha = alpha

    def evaluate(self, data: np.ndarray, prev_value=None):
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
        data = ToArray(data)
        super().__init__(data = data)

    def evaluate(self, data: np.ndarray):            
        results = []
        if data.ndim != 2:
            data = data.reshape(data.shape[0], -1)
        for i in range(1, self.nharmonics + 1):
            results.append(np.cos(2*np.pi*i*data))
            results.append(np.sin(2*np.pi*i*data))
    
        result = np.hstack(results)
        return result

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