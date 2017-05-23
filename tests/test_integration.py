from random import uniform
from unittest import TestCase

from scipy.stats import norm

from numerical.integration import std_norm_cdf, gauss_quad_2


class IntegrationTest(TestCase):

    def test_cdf_negative_input(self):
        self.assertAlmostEqual(norm.cdf(-1), std_norm_cdf(-1), places=10)

    def test_cdf_zero_input(self):
        self.assertAlmostEqual(norm.cdf(0), std_norm_cdf(0), places=10)

    def test_cdf_positive_input(self):
        self.assertAlmostEqual(norm.cdf(1), std_norm_cdf(1), places=10)

    def test_cdf_one(self):
        self.assertAlmostEqual(norm.cdf(1e8), std_norm_cdf(1e8), places=10)

    def test_cdf_zero(self):
        self.assertAlmostEqual(norm.cdf(-1e8), std_norm_cdf(-1e8), places=10)

    def test_cdf_half(self):
        self.assertAlmostEqual(0.5, std_norm_cdf(0), places=10)

    def test_cdf_minus_three(self):
        self.assertAlmostEqual(norm.cdf(-3), std_norm_cdf(-3), places=10)

    def test_cdf_three(self):
        self.assertAlmostEqual(norm.cdf(3), std_norm_cdf(3), places=10)

    def test_cdf_rand(self):
        for _ in range(1000):
            x = uniform(-1e8, 1e8)
            self.assertAlmostEqual(norm.cdf(x), std_norm_cdf(x), places=10)

    def test_gauss_quad_2(self):
        self.assertEqual(gauss_quad_2(lambda x: x ** 2, -1, 1), 2 / 3)
        self.assertEqual(gauss_quad_2(lambda x: x ** 3, -1, 1), 0)

        def f(x): return x ** 3 - x ** 2 + x - 1
        self.assertEqual(gauss_quad_2(f, -1, 1), - 8 / 3)
