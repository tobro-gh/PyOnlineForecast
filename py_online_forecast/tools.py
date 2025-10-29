from __future__ import annotations
from dataclasses import replace
from .core import *
from typing import List
import pandas as pd
from datetime import datetime
import time
from typing import List, Tuple
import sched
import inspect

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
    A class for tracking and managing time-series data with optional preprocessing and memory constraints.
    """

    def __init__(self, memory: int = None, preprocess_fn = None):
        self.data: pd.DataFrame = None
        self.memory = memory
        self.preprocess_fn = preprocess_fn
        self.preprocess_storage = {}
        # Get number of arguments for preprocess function
        if preprocess_fn:
            num_args = len(inspect.signature(preprocess_fn).parameters)
            if num_args == 2:
                self.preprocess_macro = lambda x: self.preprocess_fn(x, self.preprocess_storage)
            elif num_args == 1:
                self.preprocess_macro = lambda x: self.preprocess_fn(x)
            else:
                raise ValueError("preprocess_fn must accept either 1 or 2 arguments.")

        # Make room for any online updaters that need to track updates
        self._online_updaters = set()

    def _subscribe(self, updater: OnlineUpdater):
        self._online_updaters.add(updater)

    def _unsubscribe(self, updater: OnlineUpdater):
        self._online_updaters.remove(updater)

    def _notify_online_updaters(self):
        for updater in self._online_updaters:
            updater._notify(self)

    def add_data(self, data):
        if self.preprocess_fn:
            data = self.preprocess_fn(data, self.preprocess_storage)
            if data is None:
                return

        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        old_data = self.data

        if old_data is None:
            self.data = data
            self._notify_online_updaters()
            return
        
        end_time = self.data.index[-1]
        start_time = data.index[0]

        if end_time >= start_time:
            raise ValueError(f"New data overlaps with existing data. End time of existing data: {end_time}, start time of new data: {start_time}")

        self.data = pd.concat([old_data, data], axis=0)

        # Truncate to memory
        if self.memory is not None and len(self.data) > self.memory:
            self.data = self.data.iloc[-self.memory:]

        self._notify_online_updaters()

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