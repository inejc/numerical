import numpy as np
from numpy.random import rand
from scipy.linalg import lu, solve_triangular, norm


def inverse_iteration(A, eigval_approx, acc=1e-5):
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
    
    acc: float
        Desired accuracy of the solution (i.e. the stopping condition).

    Returns
    -------
    eigenvector: array-like, shape (n,)
        Approximate eigenvector.

    eigenvalue: float
        Approximate eigenvalue.
    """
    I = np.identity(A.shape[0])
    A_shifted = A - eigval_approx * I

    eigvec = rand(A.shape[0])

    P, L, U = lu(A_shifted)

    while True:
        # avoid computing the inverse of a shifted matrix
        x = solve_triangular(L, P.dot(eigvec), lower=True)
        x = solve_triangular(U, x)

        eigvec = x / norm(x)
        eigval = eigvec.dot(A).dot(eigvec)

        if norm((A - eigval * I).dot(eigvec), np.inf) < acc * norm(A, np.inf):
            break

    return eigvec, eigval
