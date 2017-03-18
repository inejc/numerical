from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from numerical.linear_equations import construct_M, iterative
from numerical.linear_equations import iterative_gauss_seidel, _mul_sparse
from numerical.linear_equations import iterative_jacobi, iterative_sor


class IterativeMethodsTest(TestCase):

    def setUp(self):
        self.A = np.array([
            [2, -1, 0, 0],
            [-1, 2, -1, 0],
            [0, -1, 2, -1],
            [0, 0, -1, 2]])

        self.b = np.array([1, 0, 0, 1])
        self.x0 = np.zeros(self.b.shape)
        self.acc = 1e-7
        self.omega = 1.3

    def test_construct_M(self):
        M_expected = np.array([
            [0, 2, -1],
            [-1, 2, -1],
            [-1, 2, -1],
            [-1, 2, 0]])

        M = construct_M(self.A)
        assert_array_equal(M_expected, M)

    def test_same_solutions(self):
        M = construct_M(self.A)

        x0 = self.x0.copy()
        x_j, _ = iterative_jacobi(M, self.b, x0, self.acc)

        x0 = self.x0.copy()
        x_gs, _ = iterative_gauss_seidel(M, self.b, x0, self.acc)

        x0 = self.x0.copy()
        x_sor, _ = iterative_sor(M, self.b, x0, self.acc, self.omega)

        assert_array_almost_equal(x_j, x_gs)
        assert_array_almost_equal(x_gs, x_sor)

    def test_num_iter(self):
        M = construct_M(self.A)
        res = iterative(M, self.b, self.x0, self.acc, self.omega)

        self.assertTrue(res['j_num_iter'] > res['gs_num_iter'])
        self.assertTrue(res['gs_num_iter'] > res['sor_num_iter'])

    def test_correct_solution(self):
        x_expected = np.array([1, 1, 1, 1])
        M = construct_M(self.A)
        res = iterative(M, self.b, self.x0, self.acc, self.omega)

        assert_array_almost_equal(x_expected, res['x'])

    def test_mul_sparse(self):
        x = np.array([1, 1, 1, 1])
        M = construct_M(self.A)
        expected_b = _mul_sparse(M, x)

        assert_array_equal(expected_b, self.b)
