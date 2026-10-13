import numpy as np
import pandas as pd
import unittest

from py_online_forecast.core import DEFAULT_SOURCE
from py_online_forecast.forecast_tools import ForecastFormat, Subset


def make_data():
    columns = pd.MultiIndex.from_tuples(
        [("y", 0), ("y", 1), ("x", 0)], names=["Variable", "Horizon"]
    )
    return pd.DataFrame(np.arange(9).reshape(3, 3), columns=columns)


class ForecastToolsTests(unittest.TestCase):
    def test_forecast_matrix_accessors(self):
        data = make_data()

        self.assertEqual(data.fc.variables, ["y", "x"])
        self.assertEqual(data.fc.horizons, (0, 1))
        lagged = data.fc.lag().iloc[:, 1].to_numpy()
        self.assertTrue(np.isnan(lagged[0]))
        np.testing.assert_array_equal(lagged[1:], [1, 4])

    def test_subset_transformation_and_format(self):
        data = make_data()
        subset = Subset(DEFAULT_SOURCE, "y", horizons=(1,))(data)
        formatted = ForecastFormat(DEFAULT_SOURCE)(subset)

        self.assertEqual(subset.shape, (3, 1))
        self.assertTrue(formatted.equals(subset))