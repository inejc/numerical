import numpy as np
from numpy.random import uniform

from numerical.linear_equations import iterative
from numerical.linear_equations import iterative_sor, iterative_gauss_seidel


def demo_0():
    # Matt and Jane are selling fruit. Buyers can purchase small boxes of
    # apples and large boxes of apples. Matt sold 3 small boxes and 14 large
    # boxes for a total of 203€. Jane sold 11 small boxes and 11 large boxes
    # for a total of 220€. How much do the boxes cost? (the correct solution
    # is 7€ and 13€)

    # 3  x_0 + 14 x_1 = 203
    # 11 x_0 + 11 x_1 = 220
    A = np.array([
        [3, 14],
        [11, 11]])
    b = np.array([203, 220])

    result = iterative(A, b, x0=np.zeros(b.shape), acc=1e-7, omega=1)

    # let's take a look if the returned solution is correct
    print("Small box price: {:.2f}€".format(result['x'][0]))
    print("Large box price: {:.2f}€".format(result['x'][1]))

    # how many iterations were needed with different iterative methods?
    print("Number of iterations needed to find the solution:")
    print("Jacobi: {:d}".format(result['j_num_iter']))
    print("Gauss-Seidel: {:d}".format(result['gs_num_iter']))
    print("SOR: {:d}".format(result['sor_num_iter']))


def demo_1():
    A = np.array([
        [2, -1, 0, 0],
        [-1, 2, -1, 0],
        [0, -1, 2, -1],
        [0, 0, -1, 2]])
    b = np.array([1, 0, 0, 1])

    # let's compare number of iterations for the Gauss-Seidel
    # method and the SOR method
    _, gs_num_iter = iterative_gauss_seidel(
        A, b,
        x0=np.zeros(b.shape),
        acc=1e-7,
    )
    _, sor_num_iter = iterative_sor(
        A, b,
        x0=np.zeros(b.shape),
        acc=1e-7,
        omega=1
    )
    print("Number of iterations needed to find the solution:")
    print("Gauss-Seidel: {:d}".format(gs_num_iter))
    print("SOR with omega = 1: {:d}".format(sor_num_iter))

    # can we do better? -> random search
    fastest_convergence = (sor_num_iter, 1)

    for _ in range(100):
        omega = uniform(0, 2)
        _, sor_num_iter = iterative_sor(
            A, b,
            x0=np.zeros(b.shape),
            acc=1e-7,
            omega=omega
        )

        if sor_num_iter < fastest_convergence[0]:
            fastest_convergence = (sor_num_iter, omega)

    str_ = "SOR with omega = {:f}: {:d}"
    print(str_.format(fastest_convergence[1], fastest_convergence[0]))


if __name__ == '__main__':
    demo_0()
    print()
    demo_1()
