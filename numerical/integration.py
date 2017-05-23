from math import pi, sqrt
from os.path import join, dirname

import numpy as np


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
        return const * gauss_quad(_propto_pdf_changed_var, 0, u, num_points=15)
    elif -3 < x <= 0:
        return .5 - const * gauss_quad(_propto_pdf, x, 0, num_points=9)
    elif 0 < x <= 3:
        return .5 + const * gauss_quad(_propto_pdf, 0, x, num_points=9)
    else:
        # can't compute 1 - F because of catastrophic cancellation
        F_0_3 = gauss_quad(_propto_pdf, 0, 3, num_points=9)
        F_3_x = gauss_quad(_propto_pdf_changed_var, 1 / 3, 1 / x, num_points=15)
        return .5 + const * (F_0_3 + F_3_x)


def _propto_pdf(x):
    return np.exp(- .5 * x ** 2)


def _propto_pdf_changed_var(u):
    # u = 1 / x
    return - np.exp(- .5 * u ** -2) / u ** 2


def gauss_quad(func, a, b, num_points, n=1):
    """Approximates a definite integral of func on interval [a, b] using
    Legendre-Gauss quadrature integral approximation.

    Parameters
    ----------
    func: callable
        A function to integrate.

    a: float
        Lower bound of the integration.

    b: float
        Upper bound of the integration.

    num_points: int, one of (9, 15)
        Number of weights and abscissae to use (i.e. Legendre roots
        and coefficients).

    n: int
        Number of regions in which [a, b] is split.

    Returns
    -------
    I_hat: float
        Estimate of the area under the curve.
    """
    assert num_points in _LEGENDRE.keys()
    legendre = _LEGENDRE[num_points]

    h_half = 0.5 * (b - a) / n

    I_hat = 0
    for i in range(n):
        for w_j, x_j in legendre:
            I_hat += w_j * func(h_half * (x_j + 2 * i + 1) + a)

    return h_half * I_hat


def gauss_quad_2(func, a, b, n=1):
    """Approximates a definite integral of func on interval [a, b] using
    a vectorized Legendre-Gauss quadrature integral approximation of degree 2.
    For parameters and return values see gauss_quad_9 above, but note that
    func has to be a vectorized function.
    """
    h_half = 0.5 * (b - a) / n
    x = sqrt(1 / 3)
    i = 2 * np.arange(n) + 1
    regions = func(h_half * (x + i) + a) + func(h_half * (-x + i) + a)

    return h_half * np.sum(regions)


def _line_to_tuple(l):
    split = l.split(',')
    return float(split[0]), float(split[1])

_DATA_DIR = join(dirname(dirname(__file__)), 'data')
_LEGENDRE = {}

with open(join(_DATA_DIR, 'legendre_9.csv'), 'r') as f:
    _LEGENDRE[9] = [_line_to_tuple(l) for l in list(f)]

with open(join(_DATA_DIR, 'legendre_15.csv'), 'r') as f:
    _LEGENDRE[15] = [_line_to_tuple(l) for l in list(f)]
