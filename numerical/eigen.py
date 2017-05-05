import numpy as np
from numpy.random import rand
from scipy.linalg import lu, solve_triangular, norm


def inverse_iteration(A, eigval_approx, reduce=True, acc=1e-6):
    """Finds an approximate eigenvector for an approximation to a corresponding
    eigenvalue.

    Parameters
    ----------
    A: array-like, shape (n, n)
        Matrix for which the eigenvector and a corresponding
        eigenvalue will be computed.

    eigval_approx: float
        Initial approximation for the eigenvalue. Should be closer to the
        eigenvalue in interest than to any other eigenvalue.

    reduce: boolean
        If true A is reduced to a similar Hessenberg matrix (or tridiagonal
        matrix if A is symmetric) with Householder reflections prior to
        calculating the eigenvector and the eigenvalue.

    acc: float
        Desired accuracy of the solution (i.e. the stopping condition).

    Returns
    -------
    eigenvector: array-like, shape (n,)
        Approximate eigenvector.

    eigenvalue: float
        Approximate eigenvalue.
    """
    if reduce:
        A, vectors_u = householder_reduce(A)

    I = np.identity(A.shape[0])
    A_shifted = A - eigval_approx * I

    eigvec = rand(A.shape[0])

    P, L, U = lu(A_shifted)

    while True:
        # avoid computing the inverse of a matrix (use LU decomposition instead)
        x = solve_triangular(L, P.dot(eigvec), lower=True)
        x = solve_triangular(U, x)

        eigvec = x / norm(x)
        eigval = eigvec.dot(A).dot(eigvec)

        if norm((A - eigval * I).dot(eigvec), np.inf) < acc * norm(A, np.inf):
            break

    if reduce:
        eigvec = _dot_Q_x(vectors_u, eigvec)

    return eigvec, eigval


def householder_reduce(A):
    """Computes the reduction of a matrix to a similar Hessenberg matrix using
    householder reflections. If input is symmetric reduction results in a
    tridiagonal matrix.

    Parameters
    ----------
    A: array-like, shape (n, n)
        Matrix to reduce.

    Returns
    -------
    H: array-like, shape (n, n)
        Upper Hessenberg or tridiagonal matrix similar to A.

    vectors_u: list, length n - 2
        Vectors u that determine the Qi matrices.
    """
    H = A.copy()
    vectors_u = []

    for i in range(H.shape[0] - 2):
        u = H[i + 1:, i].copy()
        u[0] += np.sign(u[0]) * norm(u)
        u = (u / norm(u)).reshape((-1, 1))
        vectors_u.append(u)

        # H = Q_i.T.dot(H)
        H[i + 1:, i:] = H[i + 1:, i:] - 2 * u.dot(u.T.dot(H[i + 1:, i:]))
        # H = H.dot(Q_i)
        H[:, i + 1:] = H[:, i + 1:] - 2 * H[:, i + 1:].dot(u.dot(u.T))

    return H, vectors_u


def _dot_Q_x(vectors_u, x):
    reversed_ = zip(range(len(vectors_u), 0, -1), reversed(vectors_u))
    for i, u in reversed_:
        x[i:] = x[i:] - 2 * u.dot(u.T.dot(x[i:]))
    return x
