from .core import *
### Transformations

class BackShift(Transformation):

    def __init__(self, shifts: list | dict, data = DEFAULT_SOURCE, skip_duplicates = False, initial_value = np.nan):
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
        super().__init__(data = data, memory = MEMORY)

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

    def __init__(self, data = DEFAULT_SOURCE):
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
    
    def __init__(self, data = DEFAULT_SOURCE):
        super().__init__(data = data)

    def evaluate(self, data):
        if isinstance(data, pd.DataFrame):
            vars = data.columns
        elif isinstance(data, pd.Series):
            vars =  data.name
        return get_horizons(vars)
    

class Length(Transformation):
    
    def __init__(self, data):
        super().__init__(data = data)

    def evaluate(self, data):
        return get_num_obs(data)

class One(Transformation):
    
    def __init__(self, index = DEFAULT_INDEX):
        super().__init__(index = index)

    def evaluate(self, index):
        if isinstance(index, (pd.Index, np.ndarray)):
            return np.ones((len(index), 1))
        else:
            return np.ones(1)
        
class LowPass(Transformation):
    def __init__(self, var, alpha = 0):
        super().__init__(data=var, prev_value = MEMORY)
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

def get_indexer(subset_index, target_index):
    # Get unique values in subset_index
    subset_index = pd.Index(subset_index).unique()
    return np.array([i for i, col in enumerate(target_index) if col in subset_index])

class Combine(Transformation):
    def __init__(self, *sources, format_result = None, index = DEFAULT_INDEX, use_fc_format = True, as_dict = False, names = None):
        super().__init__(*sources, index = index, columns = MEMORY)
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

    def __init__(self, data, columns, index = DEFAULT_INDEX):
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
        super().__init__(data = data, indices = MEMORY)


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
        super().__init__(data = data, indices = MEMORY)
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
    
class TimeOfDay(Transformation):

    def __init__(self, t = DEFAULT_INDEX):
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

    def __init__(self, t=DEFAULT_INDEX):
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

    def __init__(self, t=DEFAULT_INDEX):
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

class ToArray(Transformation):

    def __init__(self, data):
        super().__init__(data = data)

    def evaluate(self, data):
        if isinstance(data, np.ndarray):
            return data
        return np.asarray(data)

class Map(Transformation):

    def __init__(self, *vars, data = DEFAULT_SOURCE):
        super().__init__(data = data)
        self.vars = list(vars)
    
    def evaluate(self, data):
        # TODO: consider storing index to make this more efficient
        return data[self.vars]

    def __repr__(self):
        return super().__repr__() + f"({self.vars})"

class Select(Transformation):
    # TODO: deprecate, instead use transform[key]
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

    def __init__(self, indices, data = DEFAULT_SOURCE):
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
    
class FSDay(Transformation):
    
    def __init__(self, freq, t = DEFAULT_INDEX, nharmonics = 1, horizons = None):
        self.nharmonics = nharmonics
        self.freq = freq
        self.horizons = [0] if horizons is None else horizons
        super().__init__(t = t, pre_computed = MEMORY)

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

    def __init__(self, hour, dayofweek = None, duration = None, horizons: int | list = 0, index = DEFAULT_INDEX):
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
        super().__init__(variable = variable, indexer = MEMORY)
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
    
class SlidingSum(Transformation):

    def __init__(self, data = DEFAULT_SOURCE, window_size = 1, *args, **kwargs):
        self.window_size = window_size
        super().__init__(data = data, old_data = MEMORY)
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
        super().__init__(data = data, old_data = MEMORY)

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

#    @standardize_wrapper("data", ensure_dim=2, output_as = "data")
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

#    @standardize_wrapper("data","mean", output_as = "data", outer_prod = True, ensure_dim = 2)
    def _eval_covariance(self, data, mean = None, state = None, forgetting = None):
        return self._evaluate(data, mean, state, forgetting)
    
#    @standardize_wrapper("data","mean", output_as = "data", ensure_dim = 2)
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