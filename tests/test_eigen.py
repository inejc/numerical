from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal

from numerical.eigen import inverse_iteration


class PowerIterationTest(TestCase):

    def setUp(self):
        self.A = np.array([
            [3, 3, 0],
            [3, 2, -7],
            [0, -8, 6]])

        self.eigen_value_1_expected = -4.52

        self.eigen_vector_1_expected = np.array([-0.30266, 0.75866, 0.57692])

    def test_inverse_iteration(self):
        eigen_vector, eigen_value = inverse_iteration(self.A, -5)

        assert_almost_equal(
            eigen_vector,
            self.eigen_vector_1_expected,
            decimal=4
        )

        self.assertAlmostEqual(
            eigen_value,
            self.eigen_value_1_expected,
            places=2
        )
