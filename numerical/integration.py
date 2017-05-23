from math import pi, sqrt
from os.path import join, dirname

import numpy as np

_DATA_DIR = join(dirname(dirname(__file__)), 'data')
with open(join(_DATA_DIR, 'legendre.csv'), 'r') as f:
    def line_to_tuple(l):
        split = l.split(',')
        return float(split[0]), float(split[1])
    _LEGENDRE = [line_to_tuple(l) for l in list(f)]


def std_norm_cdf(x):
    """Computes a cumulative distribution function of the standard normal
    distribution with precision of ten decimal places.

    Parameters
    ----------
    x: float
        Value of the random variable.

    Returns
    -------
    F: float
        Value of the cdf, i.e. F(X) = P(X <= x).
    """
    const = 1 / sqrt(2 * pi)

    if x <= -3:
        u = 1 / x
        return const * _gauss_quad(_propto_pdf_changed_var, 0, u, n=2)
    elif -3 < x <= 0:
        return .5 - const * _gauss_quad(_propto_pdf, x, 0, n=1)
    elif 0 < x <= 3:
        return .5 + const * _gauss_quad(_propto_pdf, 0, x, n=1)
    else:
        # can't compute 1 - F because of catastrophic cancellation
        F_0_3 = _gauss_quad(_propto_pdf, 0, 3, n=1)
        F_3_x = _gauss_quad(_propto_pdf_changed_var, 1 / 3, 1 / x, n=2)
        return .5 + const * (F_0_3 + F_3_x)


def _propto_pdf(x):
    return np.exp(- .5 * x ** 2)


def _propto_pdf_changed_var(u):
    # u = 1 / x
    return - np.exp(- .5 * u ** -2) / u ** 2


def _gauss_quad(func, a, b, n=2):
    """Approximates a definite integral of func on interval [a, b] using
    Legendre-Gauss quadrature integral approximation (of degree 10).

    Parameters
    ----------
    func: callable
        A function to integrate.

    a: float
        Lower bound of the integration.

    b: float
        Upper bound of the integration.

    n: int
        Number of regions in which [a, b] is split.

    Returns
    -------
    I_hat: float
        Estimate of the area under the curve.
    """
    h_half = 0.5 * (b - a) / n

    I_hat = 0
    for i in range(n):
        for w_j, x_j in _LEGENDRE:
            I_hat += w_j * func(h_half * (x_j + 2 * i + 1) + a)

    return h_half * I_hat
