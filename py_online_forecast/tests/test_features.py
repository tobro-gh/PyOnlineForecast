import numpy as np
import unittest

from py_online_forecast.core import DEFAULT_SOURCE
from py_online_forecast.features import Combine, DesignMatrix, One, SlidingMean, SlidingSum


class FeatureTests(unittest.TestCase):
    def test_feature_transforms(self):
        data = np.arange(1, 7).reshape(3, 2)

        np.testing.assert_array_equal(One()(data), [1, 1, 1])
        np.testing.assert_array_equal(
            DesignMatrix(DEFAULT_SOURCE, DEFAULT_SOURCE)(data), np.hstack([data, data])
        )

        combined = Combine(DEFAULT_SOURCE, DEFAULT_SOURCE + 1, names=["x", "y"])(data)
        np.testing.assert_array_equal(combined["x"], data)
        np.testing.assert_array_equal(combined["y"], data + 1)

    def test_stateful_sliding_transforms(self):
        data = np.arange(1, 5).reshape(-1, 1)
        sliding_sum = SlidingSum(DEFAULT_SOURCE, 2)
        sliding_mean = SlidingMean(DEFAULT_SOURCE, 2)

        sums = sliding_sum(data).ravel()
        means = sliding_mean(data).ravel()
        self.assertTrue(np.isnan(sums[0]))
        self.assertTrue(np.isnan(means[0]))
        np.testing.assert_array_equal(sums[1:], [3, 5, 7])
        np.testing.assert_array_equal(means[1:], [1.5, 2.5, 3.5])