"""
Iterative methods for solving linear equation systems of form Ax = b,
where A is a tridiagonal, diagonally dominant matrix.
"""

import numpy as np
from numpy.linalg import norm


def iterative(M, b, x0, accuracy, omega):
    """Solves a linear equations system Ax = b with Jacobi, Gauss-Seidel
    and SOR iterative methods where M is a sparse representation of a square
    tridiagonal matrix A.

    Parameters
    ----------
    M: array-like, shape (n, 3)
        A sparse representation of A with subdiagonal of A with zero as the
        first element in the first column, main diagonal in the second column
        and superdiagonal with zero as the last element in the third column.

    b: array-like, shape (n,)
        Right-hand side of the Ax = b equation.

    x0: array-like, shape (n,)
        Initial arbitrary approximation for the solution of the Ax = b system.

    accuracy: float
        Desired accuracy of the solution (i.e. the stopping condition).

    omega: float
        Weight parameter for the SOR method.

    Returns
    -------
    {
        'x': solution of the Ax = b system,
        'j_num_iter': number of iterations required by the Jacobi method,
        'gs_num_iter': number of iterations required by the Gauss-Seidel method,
        'sor_num_iter': number of iterations required by the SOR method
    }
    """
    x, j_num_iter = iterative_jacobi(M, b, x0.copy(), accuracy)
    _, gs_num_iter = iterative_gauss_seidel(M, b, x0.copy(), accuracy)
    _, sor_num_iter = iterative_sor(M, b, x0.copy(), accuracy, omega)

    return {
        'x': x,
        'j_num_iter': j_num_iter,
        'gs_num_iter': gs_num_iter,
        'sor_num_iter': sor_num_iter
    }


def iterative_jacobi(M, b, x0, accuracy):
    """Solves a linear equations system Ax = b with a Jacobi method.
    For parameters see the iterative(...) method."""
    def jacobi(i, sum_before, sum_after, _):
        return 1 / M[i, 1] * (b[i] - sum_before - sum_after)

    # previous and current approximations need to be different arrays
    # and they must not be changed inplace
    return _iterate(
        M, b,
        x_prev=x0,
        x_curr=x0.copy(),
        iteration_formula=jacobi,
        accuracy=accuracy,
        inplace=False
    )


def iterative_gauss_seidel(M, b, x0, accuracy):
    """Solves a linear equations system Ax = b with a Gauss-Seidel method.
    For parameters see the iterative(...) method."""
    def gauss_seidel(i, sum_before, sum_after, _):
        return 1 / M[i, 1] * (b[i] - sum_before - sum_after)

    # previous and current approximations need to be same arrays
    # and they must be changed inplace
    return _iterate(
        M, b,
        x_prev=x0,
        x_curr=x0,
        iteration_formula=gauss_seidel,
        accuracy=accuracy,
        inplace=True
    )


def iterative_sor(M, b, x0, accuracy, omega):
    """Solves a linear equations system Ax = b with a SOR method.
    For parameters see the iterative(...) method."""
    def sor(i, sum_before, sum_after, x_prev):
        relaxed = omega / M[i, 1] * (b[i] - sum_before - sum_after)
        return relaxed + (1 - omega) * x_prev[i]

    # previous and current approximations need to be same arrays and they
    # must be changed inplace like in _iterative_gauss_seidel(...) but
    # relaxation is used in the iteration formula
    return _iterate(
        M, b,
        x_prev=x0,
        x_curr=x0,
        iteration_formula=sor,
        accuracy=accuracy,
        inplace=True
    )


def _iterate(M, b, x_prev, x_curr, iteration_formula, accuracy, inplace):
    num_rows = M.shape[0]
    num_iter = 0

    while True:
        num_iter += 1

        for i in range(num_rows):
            if i == 0:
                sum_before = 0
                sum_after = M[i, 2] * x_prev[i + 1]
            elif i == num_rows - 1:
                sum_before = M[i, 0] * x_prev[i - 1]
                sum_after = 0
            else:
                sum_before = M[i, 0] * x_prev[i - 1]
                sum_after = M[i, 2] * x_prev[i + 1]

            x_curr[i] = iteration_formula(i, sum_before, sum_after, x_prev)

        if not inplace:
            x_prev = x_curr.copy()

        if norm(_mul_sparse(M, x_curr) - b, np.inf) < accuracy:
            return x_curr, num_iter


def _mul_sparse(M, x):
    num_rows = M.shape[0]
    b = np.empty(x.shape)

    for i in range(num_rows):
        if i == 0:
            b[i] = M[i, 1] * x[i] + M[i, 2] * x[i + 1]
        elif i == num_rows - 1:
            b[i] = M[i, 1] * x[i] + M[i, 0] * x[i - 1]
        else:
            b[i] = M[i, 1] * x[i] + M[i, 2] * x[i + 1] + M[i, 0] * x[i - 1]

    return b


def construct_M(A):
    """Constructs a sparse representation of a square tridiagonal matrix.

    Parameters
    ----------
    A: array-like, shape (n, n)
        A tridiagonal square matrix.

    Returns
    -------
    M: array-like, shape (n, 3)
        A sparse representation of A with subdiagonal of A with zero as the
        first element in the first column, main diagonal in the second column
        and superdiagonal with zero as the last element in the third column.
    """
    sub_diagonal = np.hstack((0, np.diag(A, -1))).reshape(-1, 1)
    main_diagonal = np.diag(A).reshape(-1, 1)
    super_diagonal = np.hstack((np.diag(A, -1), 0)).reshape(-1, 1)
    return np.hstack((sub_diagonal, main_diagonal, super_diagonal))
