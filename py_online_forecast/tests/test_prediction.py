import numpy as np
import unittest

from py_online_forecast.core import DEFAULT_SOURCE
from py_online_forecast.prediction import BackShift, RRR, rmse


class PredictionTests(unittest.TestCase):
    def test_backshift(self):
        result = BackShift([1, 0], initial_value=0)(np.arange(1, 5).reshape(-1, 1))

        np.testing.assert_array_equal(result, [[0, 1], [1, 2], [2, 3], [3, 4]])

    def test_rrr_returns_forecasts_and_covariance(self):
        data = np.arange(1, 7, dtype=float).reshape(-1, 1)
        model = RRR(DEFAULT_SOURCE, DEFAULT_SOURCE, memory=0.9)

        result = model(data)

        self.assertEqual(result["mean"].shape, data.shape)
        self.assertEqual(result["cov"].shape[0], len(data))
        self.assertTrue(np.isfinite(rmse(np.array([1.0, -1.0]))))