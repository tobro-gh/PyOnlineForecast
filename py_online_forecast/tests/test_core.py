import pickle
import unittest

import numpy as np

from py_online_forecast.core import CircularBuffer, DEFAULT_SOURCE, Param, Transformation


class CoreTests(unittest.TestCase):
    def test_transformation_composes_sources(self):
        source = DEFAULT_SOURCE * 2 + 1

        result = source(np.array([1, 2, 3]))

        np.testing.assert_array_equal(result, [3, 5, 7])

    def test_default_source_preserves_identity_when_pickled(self):
        loaded = pickle.loads(pickle.dumps(DEFAULT_SOURCE))

        self.assertIs(loaded, DEFAULT_SOURCE)

    def test_param_and_circular_buffer(self):
        self.assertEqual(Param(4)(None), 4)

        buffer = CircularBuffer(3, 1)
        buffer.append(np.array([[1], [2]]))
        buffer.append(np.array([[3], [4]]))

        np.testing.assert_array_equal(buffer.get(3).ravel(), [2, 3, 4])