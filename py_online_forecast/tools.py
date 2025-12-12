from __future__ import annotations
from .core import *
from typing import List
import pandas as pd
from datetime import datetime
import time
import sched


class DataHandler:

    def __init__(self, **data_sets):
        self.end_date = None
        self.data: dict[str, DataFrame] = {}
        self.add_data(**data_sets)

    def add_data(self, time_var = None, **data_sets):
        
        for name, data_set in data_sets.items():
            data_set = data_set.copy()
            if time_var:
                
                # Set time_var as index
                data_set.set_index((time_var, "NA"), inplace=True)

            else:
                # Check if index is datetime
                if not pd.api.types.is_datetime64_any_dtype(data_set.index):
                    raise ValueError(f"Index of {name} is not in datetime format.")

            # Ensure datetime index is in datetime format
            data_set.index = pd.to_datetime(data_set.index)

            if name in self.data:
                self.data[name].fc.append(data_set)
            else:
                self.data[name] = data_set

    def extract_time_series(self, *sets, start_time = None, end_time = None, method = "ffill", frequency = None, match_data = None):
        for name in sets:
            if not name in self.data:
                raise ValueError(f"Data set {name} not found in data handler.")

        if start_time is None:
            # Get first time of data set
            start_time = min([data_set.index.min() for data_set in self.data.values()])
        if end_time is None:
            # Get last time of data set
            end_time = max([data_set.index.max() for data_set in self.data.values()])

        if match_data:
            if frequency:
                raise ValueError("Only one of match_data or frequency can be provided.")
            expected_times = self.data[match_data].index
        elif frequency:
            expected_times = pd.date_range(start_time, end_time, freq=frequency)
        else:
            raise ValueError("Either match_data or frequency must be provided.")
        
        dataframes = []
        for name, data_set in self.data.items():
            if len(sets) == 0 or name in sets:
                data_set = data_set.copy().reindex(expected_times, method=method)
                dataframes.append(data_set)
        
        # Merge all dataframes on time
        result = pd.concat(dataframes, axis=1, join="outer", ignore_index=False, sort=True)

        return result

    def forget(self, time, var = "t"):
        for name, data_set in self.data.items():
            # Remove data with "t" older than time
            self.data[name] = data_set[data_set.fc[var] > time]



class DataTracker:

    """
    A class for tracking and managing time-series with memory constraints.
    """

    def __init__(self, memory: int = None, freq = None):
        self.data: pd.DataFrame = None
        self.memory = memory
        self.freq = freq
        self.preprocess_storage = {}

        # Make room for any online updaters that need to track updates
        self._online_updaters = set()

    def _subscribe(self, updater: OnlineUpdater):
        self._online_updaters.add(updater)

    def _unsubscribe(self, updater: OnlineUpdater):
        self._online_updaters.remove(updater)

    def _notify_online_updaters(self):
        for updater in self._online_updaters:
            updater._notify(self)

    def add_data(self, data: pd.Series | pd.DataFrame, forget = True):

        if self.data is None:
            self.data = data
            self._notify_online_updaters()
            return
        
        self.data = self.data.fc.append(data, sort = True)

        # Truncate to memory
        if forget:
            self.forget()

        self._notify_online_updaters()

    def forget(self):
        if self.memory is not None and len(self.data) > self.memory:
            self.data = self.data.iloc[-self.memory:]

    @property
    def has_data(self):
        return self.data is not None

    def get_data(self, start_time=None, end_time=None, expected_times=None, method=None):
        if not self.has_data:
            raise ValueError("No data available in tracker.")
        subset = self.data.loc[start_time:end_time]
        if expected_times is not None:
            return subset.reindex(expected_times, method=method)
        return subset


def extract_data(*data_trackers: DataTracker, start_time = None, end_time = None, method = None, frequency = None, match_tracker: DataTracker = None, inclusive = "both") -> pd.DataFrame:

    for tracker in data_trackers:
        if not tracker.has_data:
            raise ValueError("One or more data trackers have no data.")

    if start_time is None:
        # Get first time of data set
        start_time = min([tracker.data.index.min() for tracker in data_trackers])
    if end_time is None:
        # Get last time of data set
        end_time = max([tracker.data.index.max() for tracker in data_trackers])

    if match_tracker:
        if frequency:
            raise ValueError("Only one of match_tracker or frequency can be provided.")
        right_cond = (start_time <= match_tracker.data.index) if inclusive in ["both", "left"] else (start_time < match_tracker.data.index)
        left_cond = (end_time >= match_tracker.data.index) if inclusive in ["both", "right"] else (end_time > match_tracker.data.index)
        expected_times = match_tracker.data.index[(right_cond) & (left_cond)]
    elif frequency:
        expected_times = pd.date_range(start=start_time, end=end_time, freq=frequency, inclusive=inclusive)
    else:
        raise ValueError("Either match_tracker or frequency must be provided.")
    
    dataframes = []
    for tracker in data_trackers:
        data_set = tracker.get_data(start_time=start_time, end_time=end_time, expected_times=expected_times, method=method)
        dataframes.append(data_set)
    
    # Merge all dataframes on time
    result = pd.concat(dataframes, axis=1, join="outer", ignore_index=False, sort=True)

    return result



def get_next_time(delta, start = None, round:str = None):
    """
    Get the next update time assuming a fixed frequency based on delta. Corrects automatically to nearest scheduled time if misaligned.
    """
    start = start or pd.Timestamp.now().round(round) if round else pd.Timestamp.now()

    next_time = start + delta

    delay = next_time - pd.Timestamp.now()

    if delay < pd.Timedelta(0):
        # Move forward to first future occurrence
        intervals = 1 - (delay // delta)
        next_time = next_time + intervals * delta
        delay = next_time - pd.Timestamp.now()
    elif delay > delta:
        # Move backward to first future occurrence
        intervals = (delay // delta)
        next_time = next_time - intervals * delta
        delay = next_time - pd.Timestamp.now()

    return next_time, delay


class OnlineUpdater:

    def __init__(self, job, data_trackers: List[DataTracker], start_time: pd.Timestamp = None, start_time_round: str = None, frequency = None,
                  match_tracker: DataTracker = None, update_start_time: datetime = None, backup_interval: int = None, storage_dir: str = None,
                    data_job = None, data_job_frequency = None):

        self.data_trackers = data_trackers

        self.job = job
        self.data_job = data_job

        self.storage_dir = storage_dir

        self.update_time = None
        self.data_update_time = None

        self.backup_interval = backup_interval
        self.updates_since_backup = 0

        if bool(frequency and start_time) == bool(match_tracker):
            raise ValueError("Provide either (frequency and start_time) or match_tracker.")


        if self.backup_interval is not None and self.storage_dir is None:
            raise ValueError("Storage directory must be provided if backup interval is set.")

        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.update_start_time = update_start_time

        self.frequency = frequency
        self.match_tracker = match_tracker
        if match_tracker:
            if match_tracker not in data_trackers:
                raise ValueError("Match tracker must be one of the data trackers.")
            match_tracker._subscribe(self)
        elif frequency:
            self._delta = pd.Timedelta(frequency)
            self._schedule(round = start_time_round)

        self.data_job_frequency = data_job_frequency
        self._delta_data = pd.Timedelta(data_job_frequency)

        if data_job:
            self._schedule_data_job(round=start_time_round)

        self.event = None
        self.data_event = None
        self._last_update_time = None

    def _notify(self, tracker: DataTracker):
        if tracker == self.match_tracker:
            self._job(tracker.data.index[-1])
        else:
            tracker._unsubscribe(self)


    def _schedule(self, round: str = None):
        self.update_time, delay = get_next_time(self._delta, self.update_time, round=round)
        self.event = self.scheduler.enter(delay.total_seconds(), 1, self._job, argument=(self.update_time,))

    def _schedule_data_job(self, round: str = None):
        self.data_update_time, delay = get_next_time(self._delta_data, self.data_update_time, round=round)
        self.data_event = self.scheduler.enter(delay.total_seconds(), 1, self._data_job, argument=(self.data_update_time,))

    def _job(self, scheduled_time: pd.Timestamp):
        
        # Schedule next job
        if self.frequency:
            self._schedule()

        # Get data and update models
        start_time = self._last_update_time
        data_ready = any([tracker.has_data for tracker in self.data_trackers])
        if data_ready:
            data = extract_data(*self.data_trackers, start_time = start_time, end_time = scheduled_time, frequency=self.frequency, match_tracker=self.match_tracker, inclusive = "right")
            self._last_update_time = data.index[-1]
        else:
            data = None

        self.job(scheduled_time, data)

        if self.backup_interval is not None and self.updates_since_backup >= self.backup_interval:
            self._backup()
            self.updates_since_backup = 0

        self.updates_since_backup += 1

    def _data_job(self, scheduled_time: pd.Timestamp):
        self._schedule_data_job()
        self.data_job(scheduled_time)

    def _backup(self):
        for i, model in enumerate(self.models):
            path = self.storage_dir / f"backup_model_{i}"
            model.save_model(path)

    def start(self):
        if self.scheduler.empty():
            self._schedule()
        self.scheduler.run(blocking = True)

    def stop(self):
        self.scheduler.clear()
        self._last_update_time = None

class UpdateModels:

    def __init__(self, storage_dir = None, update_kwargs = None, **models):
        self.storage_dir = storage_dir
        self.models = models
        self._update_kwargs = update_kwargs or {}

    def __call__(self, scheduled_time, data):
        for name, model in self.models.items():
            res = model.update(data, **self._update_kwargs)
            if self.storage_dir:
                if isinstance(res, pd.DataFrame):
                    res = [res]
                for j, df in enumerate(res):
                    path = os.path.join(self.storage_dir, f"{name}_output_{j}.csv")
                    if os.path.exists(path):
                        df.to_csv(path, mode='a', header=False, index_label="index")
                    else:
                        df.to_csv(path, mode='w', header=True, index_label="index")

class DataCleaner(Transformation):

    def __init__(self, forgetting, data = DefaultSource, z_thresh = 3, forward_fill = True, track_memory = True, freq: str = None):
        mean = ForgettingMean(forgetting, track_memory = track_memory, data = data)
        variance = ForgettingVariance(forgetting, track_memory = track_memory, center = mean, covariance = False, data = data)
        super().__init__(data, variance = variance, mean = mean)
        self.z_thresh = z_thresh
        self.forward_fill = forward_fill
        self.freq = freq

    def evaluate(self, data, variance, mean):
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise ValueError("DataCleaner can only be applied to pandas DataFrame or Series.")

        # Check frequency if freq is set
        if self.freq is not None:
            
            # Check if freq is already correct
            if data.index.freq != self.freq:
                # Try to reindex to expected frequency
                expected_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=self.freq)
                data = data.reindex(expected_index)

        std = np.sqrt(variance)
        z_scores = np.abs((data - mean) / std)

        # Identify outliers
        outliers = z_scores > self.z_thresh

        # Replace outliers with NaN
        data_cleaned = data.mask(outliers)

        if self.forward_fill:
            data_cleaned = data_cleaned.ffill()

        return data_cleaned

class AlignIndex(Transformation):
    # Align dataframes/series to expected index

    def __init__(self, *data, expected_index = DefaultIndex, method: str = None):
        super().__init__(*data, expected_index = expected_index)
        self.method = method
    
    def evaluate(self, *data, expected_index):
        aligned_data = []
        for df in data:
            if not isinstance(df, (pd.DataFrame, pd.Series)):
                raise ValueError("AlignIndex can only be applied to pandas DataFrame or Series.")
            aligned_df = df.reindex(expected_index, method=self.method)
            aligned_data.append(aligned_df)
        if len(aligned_data) == 1:
            result = aligned_data[0]
        result = aligned_data

        # Output as dict using self.apply_args
        return {k: r for k, r in zip(self.apply_args, result)}

class Scaler(Transformation):

    def __init__(self, data = DefaultSource, var_scales: dict[str, float] = None):
        super().__init__(data, state = Memory)
        self.var_scales = var_scales

    def evaluate(self, data, state = None):
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise ValueError("DataScaler can only be applied to pandas DataFrame or Series.")

        if state is None:
            data_cols = data.columns if isinstance(data, pd.DataFrame) else data.index
            indexer = subset_columns(data_cols, *list(self.var_scales.keys()), return_index = True)

            # Translate bool indexer to location index
            indexer = np.where(indexer)[0]

            cols = data_cols[indexer]
            if isinstance(cols, pd.MultiIndex):
                scales = np.array([self.var_scales[col[0]] for col in cols])
            else:
                scales = np.array([self.var_scales[col] for col in cols])
            state = (indexer, scales)

        indexer, scales = state

        scaled_data = data.copy()

        if isinstance(data, pd.Series):
            scaled_data.iloc[indexer] = data.iloc[indexer] * scales            
            
        else:

            scaled_data.iloc[:, indexer] = data.iloc[:, indexer] * scales

        return scaled_data

class Aggregator(Transformation):

    def __init__(self, level, data = DefaultSource, agg_type: Literal["sum", "mean"] = None):
        if agg_type == "sum":
            agg_data = SlidingSum(window_size=level, data=data)
        elif agg_type == "mean":
            agg_data = SlidingMean(window_size=level, data=data)
        else:
            agg_data = data

        self.level = level
        
        super().__init__(agg_data, state = Memory)

    def evaluate(self, agg_data, state = None):
        if not isinstance(agg_data, pd.DataFrame):
            raise ValueError("Aggregator can only be applied to pandas dataframe.")
        if agg_data.shape[0] < self.level:
            raise ValueError("Not enough data to aggregate.")

        if state is None:
            state = 0

        # Get start index
        start = self.level - state if state > 0 else 0

        # Get every self.level-th value
        result = agg_data.iloc[start::self.level]

        # Update state
        state = (state + len(agg_data)) % self.level

        return result, state



class PreProcessor:

    def __init__(self, *outputs, clean = True, combine = False, ref = None, z_thresh = 3, forward_fill = True, forgetting = 0.995, track_memory = True, agg_level = None, agg_type = None, **kwargs):

        if not outputs:
            self._outputs = [DefaultSource]
        else:
            self._outputs = list(outputs)

        if clean:
            self._outputs = [DataCleaner(forgetting, z_thresh = z_thresh, forward_fill = forward_fill, track_memory = track_memory, data = o) for o in self._outputs]

        if agg_level is not None:
            self._outputs = [Aggregator(agg_level, data = o, agg_type = agg_type) for o in self._outputs]

        self._ref = ref or self._outputs[0]

        if combine:
            self._outputs = [Combine(*self._outputs)]

        self._transformer = Transformer(**kwargs)

        self._transformer.add_transforms(*self._outputs)

        self._transformer.set_transforms()

        self.data = {key: None for key in self._transformer.sorted_transforms} | {DefaultSource: None}

    def add_data(self, data: dict | pd.DataFrame | pd.Series | np.ndarray):
        if not isinstance(data, dict):
            data = {DefaultSource: data}

        for key, value in data.items():
            if key is not DefaultSource and key not in self.data:
                raise ValueError(f"Unknown data source: {key}")
            if self.data[key] is None:
                self.data[key] = value
            else:
                self.data[key] = self.data[key].fc.append(value, sort = True)

    def forget(self):

        for key in self.data:
            self.data[key] = None

    def update(self, data = None, index = None, forget = True):

        if data is not None:
            self.add_data(data)


        if index is None:
            if self.data[self._ref] is not None:
                index = self.data[self._ref].index
            elif DefaultSource in self.data and self.data[DefaultSource] is not None:
                index = self.data[DefaultSource].index
            else:
                raise ValueError("No reference data available for determining update index.")

        # Get available data in time range
        update_data = {}
        for key, value in self.data.items():
            if value is not None:
                update_data[key] = value.loc[index]
                
        # Update transformer
        result = self._transformer.transform(update_data)

        if forget:
            self.forget()
    
        outputs = {key: result[key] for key in self._outputs}

        if len(outputs) == 1:
            return list(outputs.values())[0]

        return outputs 