from random import uniform
from unittest import TestCase

from scipy.stats import norm

from numerical.integration import std_norm_cdf


class StandardNormalCdfTest(TestCase):

    def test_negative_input(self):
        self.assertAlmostEqual(norm.cdf(-1), std_norm_cdf(-1), places=10)

    def test_zero_input(self):
        self.assertAlmostEqual(norm.cdf(0), std_norm_cdf(0), places=10)

    def test_positive_input(self):
        self.assertAlmostEqual(norm.cdf(1), std_norm_cdf(1), places=10)

    def test_one(self):
        self.assertAlmostEqual(norm.cdf(1e6), std_norm_cdf(1e6), places=10)

    def test_zero(self):
        self.assertAlmostEqual(norm.cdf(-1e6), std_norm_cdf(-1e6), places=10)

    def test_half(self):
        self.assertAlmostEqual(0.5, std_norm_cdf(0), places=10)

    def test_minus_three(self):
        self.assertAlmostEqual(norm.cdf(-3), std_norm_cdf(-3), places=10)

    def test_three(self):
        self.assertAlmostEqual(norm.cdf(3), std_norm_cdf(3), places=10)

    def test_rand(self):
        for _ in range(1000):
            x = uniform(-1e6, 1e6)
            self.assertAlmostEqual(norm.cdf(x), std_norm_cdf(x), places=10)
