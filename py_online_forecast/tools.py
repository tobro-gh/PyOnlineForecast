from __future__ import annotations
import pandas as pd
import numpy as np
from .core import Transformation, DEFAULT_SOURCE, DEFAULT_INDEX, MEMORY, ForgettingMean, ForgettingVariance, subset_columns
from typing import Literal

class Reindexer(Transformation):

    def __init__(self, freq, data = DEFAULT_SOURCE):
        super().__init__(data)
        self.freq = freq

    def evaluate(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Reindexer can only be applied to pandas DataFrame.")

        if data.index.freq != self.freq:
            expected_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=self.freq)
            data = data.reindex(expected_index)

        return data

class DataCleaner(Transformation):

    def __init__(self, forgetting, data = DEFAULT_SOURCE, z_thresh = 3, forward_fill = True, track_memory = True, freq: str = None):

        if freq is not None:
            data = Reindexer(freq, data)

        mean = ForgettingMean(forgetting, track_memory = track_memory, data = data)
        variance = ForgettingVariance(forgetting, track_memory = track_memory, center = mean, covariance = False, data = data)
        super().__init__(data, variance = variance, mean = mean, last_state = MEMORY)
        self.z_thresh = z_thresh
        self.forward_fill = forward_fill
        self.freq = freq

    def evaluate(self, data, variance, mean, last_state = None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("DataCleaner can only be applied to pandas DataFrame.")

        std = np.sqrt(variance)
        z_scores = np.abs((data - mean) / std)

        # Identify outliers
        outliers = z_scores > self.z_thresh

        # Replace outliers with NaN
        data_cleaned = data.mask(outliers)

        if last_state is not None:
            # Fill first row with last state
            filled_first_row = data_cleaned.iloc[[0]].fillna(value=last_state)
            data_cleaned.iloc[0] = filled_first_row.iloc[0]

        if self.forward_fill:
            data_cleaned = data_cleaned.ffill()

        # Store last state
        last_state = data_cleaned.iloc[-1]

        return data_cleaned, last_state

class AlignIndex(Transformation):
    # Align dataframes/series to expected index

    def __init__(self, *data, expected_index = DEFAULT_INDEX, method: str = None):
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

    def __init__(self, data = DEFAULT_SOURCE, var_scales: dict[str, float] = None):
        super().__init__(data, state = MEMORY)
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

    def __init__(self, freq, data = DEFAULT_SOURCE, agg_type: Literal["sum", "mean"] = "mean"):
        self.freq = freq
        if agg_type not in ["sum", "mean"]:
            raise ValueError("agg_type must be either 'sum' or 'mean'.")
        self.agg_type = agg_type
        super().__init__(data)

    def evaluate(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Aggregator can only be applied to pandas dataframe.")

        if self.agg_type == "sum":
            aggregated_data = data.resample(self.freq, closed = "right", label = "right").sum()
        else:
            aggregated_data = data.resample(self.freq, closed = "right", label = "right").mean()
                
        return aggregated_data

