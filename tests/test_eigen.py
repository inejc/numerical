from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.linalg import norm

from numerical.eigen import inverse_iteration, householder_reduce


class PowerIterationTest(TestCase):

    def test_inverse_iteration(self):
        A = np.array([
            [3, 3, 0],
            [3, 2, -7],
            [0, -8, 6]]).astype(float)

        eigval_1_expected = -4.52
        eigvec_1_expected = np.array([-0.30266, 0.75866, 0.57692])
        eigvec, eigval = inverse_iteration(A, -5)

        self._assert_collinear(eigvec, eigvec_1_expected)

        self.assertAlmostEqual(
            eigval,
            eigval_1_expected,
            places=2
        )

    def test_hessenberg(self):
        A = np.array([
            [5, 1, 2, 0, 4],
            [1, 4, 2, 1, 3],
            [2, 2, 5, 4, 0],
            [0, 1, 4, 1, 3],
            [4, 3, 0, 3, 4]]).astype(float)

        H_expected = np.array([
            [5, -4.58258, 0, 0, 0],
            [-4.58258, 5.71429, -5.60733, 0, 0],
            [0, -5.60733, 1.8636, -2.36985, 0],
            [0, 0, -2.36985, 3.7425, -1.51171],
            [0, 0, 0, -1.51171, 2.67961]])

        H, _ = householder_reduce(A)
        assert_array_almost_equal(H, H_expected, decimal=5)

        A = np.array([
            [1, -4, 0, 3],
            [-1, 3, 0, -2],
            [3, -7, -2, 6],
            [0, 4, 0, -2]]).astype(float)

        H_expected = np.array([
            [1, 1.26491, -0.507093, 4.8107],
            [3.16228, 0.6, -1.17595, 9.97282],
            [0, 1.49666, 0.4, 3.79473],
            [0, 0, 0, -2]])

        H, _ = householder_reduce(A)
        assert_array_almost_equal(H, H_expected, decimal=5)

    def test_reduction_no_reduction_same(self):
        A = np.array([
            [2, -1, 1],
            [-1, 2, 1],
            [1, -1, 2]]).astype(float)

        eigvec_0, eigval_0 = inverse_iteration(A, 2.1, reduce=True)
        eigvec_1, eigval_1 = inverse_iteration(A, 2.1, reduce=False)

        self._assert_collinear(eigvec_0, eigvec_1)
        self.assertAlmostEqual(eigval_0, eigval_1, places=4)

    def _assert_collinear(self, x, y):
        abs_cos = np.abs(x.dot(y) / (norm(x) * norm(y)))
        self.assertAlmostEqual(abs_cos, 1)
