"""
Iterative methods for solving linear equation systems of the form Ax = b.
"""

import numpy as np
from numpy.linalg import norm


def to_sparse(A):
    """Constructs a sparse representation of a tridiagonal square matrix A.

    Parameters
    ----------
    A: array-like, shape (n, n)
        A tridiagonal square matrix (i.e. a coefficient matrix).

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


def iterative(C, b, x0, accuracy, omega):
    """Solves a linear equations system Cx = b with the Jacobi, Gauss-Seidel
    and SOR iterative methods where C is a diagonally dominant square matrix
    (i.e. a coefficient matrix) or a sparse representation of it.

    Parameters
    ----------
    C: array-like, shape (n, n) or (n, 3)
        Diagonally dominant square matrix A (i.e. a coefficient matrix) or a
        sparse representation M.

    b: array-like, shape (n,)
        Right-hand side of the Cx = b equation.

    x0: array-like, shape (n,)
        Initial arbitrary approximation for the solution of the Cx = b system.

    accuracy: float
        Desired accuracy of the solution (i.e. the stopping condition).

    omega: float
        Weight parameter for the SOR method.

    Returns
    -------
    {
        'x': solution of the Cx = b system,
        'j_num_iter': number of iterations required by the Jacobi method,
        'gs_num_iter': number of iterations required by the Gauss-Seidel method,
        'sor_num_iter': number of iterations required by the SOR method
    }
    """
    x, j_num_iter = iterative_jacobi(C, b, x0.copy(), accuracy)
    _, gs_num_iter = iterative_gauss_seidel(C, b, x0.copy(), accuracy)
    _, sor_num_iter = iterative_sor(C, b, x0.copy(), accuracy, omega)

    return {
        'x': x,
        'j_num_iter': j_num_iter,
        'gs_num_iter': gs_num_iter,
        'sor_num_iter': sor_num_iter
    }


def iterative_jacobi(C, b, x0, accuracy):
    """Solves a linear equations system Cx = b with the Jacobi method.
    For parameters see the iterative(...) function above.
    """
    def jacobi_formula(i, is_sparse, sum_before, sum_after, _):
        a_i = C[i, 1] if is_sparse else C[i, i]
        return 1 / a_i * (b[i] - sum_before - sum_after)

    # previous and current approximations need to be different arrays
    # and current solution must not be changed inplace
    return _iterate(
        C, b,
        x_prev=x0,
        x_curr=x0.copy(),
        iteration_formula=jacobi_formula,
        accuracy=accuracy,
        inplace=False
    )


def iterative_gauss_seidel(C, b, x0, accuracy):
    """Solves a linear equations system Cx = b with the Gauss-Seidel method.
    For parameters see the iterative(...) function above.
    """
    def gauss_seidel_formula(i, is_sparse, sum_before, sum_after, _):
        a_i = C[i, 1] if is_sparse else C[i, i]
        return 1 / a_i * (b[i] - sum_before - sum_after)

    # previous and current approximations need to be same arrays
    # and current solution must be changed inplace
    return _iterate(
        C, b,
        x_prev=x0,
        x_curr=x0,
        iteration_formula=gauss_seidel_formula,
        accuracy=accuracy,
        inplace=True
    )


def iterative_sor(C, b, x0, accuracy, omega):
    """Solves a linear equations system Cx = b with the SOR method.
    For parameters see the iterative(...) function above.
    """
    def sor_formula(i, is_sparse, sum_before, sum_after, x_prev):
        a_i = C[i, 1] if is_sparse else C[i, i]
        gauss_seidel = 1 / a_i * (b[i] - sum_before - sum_after)
        return omega * gauss_seidel + (1 - omega) * x_prev[i]

    # previous and current approximations need to be same arrays and current
    # solution must be changed inplace like in _iterative_gauss_seidel(...) but
    # relaxation is used in the iteration formula above
    return _iterate(
        C, b,
        x_prev=x0,
        x_curr=x0,
        iteration_formula=sor_formula,
        accuracy=accuracy,
        inplace=True
    )


def _iterate(C, b, x_prev, x_curr, iteration_formula, accuracy, inplace):
    is_sparse = _is_sparse(C)
    num_rows = C.shape[0]
    num_iter = 0

    while True:
        num_iter += 1

        for i in range(num_rows):
            if is_sparse:
                sums = _sum_all_but_ith_sparse(i, C, x_prev, num_rows)
                sum_before, sum_after = sums
            else:
                sum_before, sum_after = _sum_all_but_ith(i, C, x_prev)

            x_curr[i] = iteration_formula(
                i, is_sparse,
                sum_before, sum_after, x_prev
            )

        if not inplace:
            x_prev = x_curr.copy()

        b_curr = _dot_sparse(C, x_curr) if is_sparse else np.dot(C, x_curr)
        if norm(b_curr - b, np.inf) < accuracy:
            return x_curr, num_iter


def _is_sparse(C):
    if C.shape[0] != C.shape[1]:
        return True
    elif C[0, 0] == 0 and C[-1, -1] == 0 and np.count_nonzero(C) == C.size - 2:
        return True

    return False


def _sum_all_but_ith_sparse(i, M, x_prev, num_rows):
    if i == 0:
        sum_before = 0
        sum_after = M[i, 2] * x_prev[i + 1]
    elif i == num_rows - 1:
        sum_before = M[i, 0] * x_prev[i - 1]
        sum_after = 0
    else:
        sum_before = M[i, 0] * x_prev[i - 1]
        sum_after = M[i, 2] * x_prev[i + 1]

    return sum_before, sum_after


def _sum_all_but_ith(i, A, x_prev):
    sum_before = np.dot(A[i, :i], x_prev[:i])
    sum_after = np.dot(A[i, i + 1:], x_prev[i + 1:])
    return sum_before, sum_after


def _dot_sparse(M, x):
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
