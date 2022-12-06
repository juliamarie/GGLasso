import numpy as np
from numpy.testing import assert_array_almost_equal

from gglasso.solver.tagl_admm_solver import la_admm_tagl
from gglasso.solver.single_admm_solver import ADMM_SGL


def test_example_tree_big_lambda2():
    A = np.zeros((5, 8))
    A[:5, :5] = np.identity(5)
    v = [1, 1, 1, 0, 0]
    w = [0, 0, 0, 1, 1]
    A[:, 5] = v
    A[:, 6] = w
    A[:, 7] = np.ones(5)
    W = np.random.rand(2)
    V = np.random.rand(3)
    S = np.zeros((5, 5))
    S[:3, :3] = W @ W.T
    S[3:, 3:] = V @ V.T
    Omega, Gamma, D = la_admm_tagl(S, A, 1, 100, 1, 100, 10, 1e-5, 1e-5)
    for i in range(5):
        Omega[i][i] = 1
    assert_array_almost_equal(Omega, np.identity(5), 5)
    return


def test_example_tree_big_lambda1():
    A = np.zeros((5, 8))
    A[:5, :5] = np.identity(5)
    v = [1, 1, 1, 0, 0]
    w = [0, 0, 0, 1, 1]
    A[:, 5] = v
    A[:, 6] = w
    A[:, 7] = np.ones(5)
    W = np.random.rand(2)
    V = np.random.rand(3)
    S = np.zeros((5, 5))
    S[:3, :3] = W @ W.T
    S[3:, 3:] = V @ V.T
    Omega, Gamma, D = la_admm_tagl(S, A, 100, 1, 1, 100, 10, 1e-5, 1e-5)
    Omega2 = np.zeros((5, 5))
    Omega2.fill(Gamma[1][1])
    for i in range(5):
        Omega[i][i] = Gamma[1][1]
    assert_array_almost_equal(Omega, Omega2, 5)
    return


def test_example_tree_glasso():
    A = np.zeros((5, 8))
    A[:5, :5] = np.identity(5)
    v = [1, 1, 1, 0, 0]
    w = [0, 0, 0, 1, 1]
    A[:, 5] = v
    A[:, 6] = w
    A[:, 7] = np.ones(5)
    W = np.random.rand(2)
    V = np.random.rand(3)
    S = np.zeros((5, 5))
    S[:3, :3] = W @ W.T
    S[3:, 3:] = V @ V.T
    Omega, Gamma, D = la_admm_tagl(S, A, 1e-7, 1, 1, 100, 10, 1e-7, 1e-4)
    sol, info = ADMM_SGL(S, 1, np.zeros((5, 5)))
    Omega2 = sol.get('Omega')
    for i in range(5):
        Omega[i][i] = 1
        Omega2[i][i] = 1
    assert_array_almost_equal(Omega, Omega2, 5)
    return


def test():
    A = np.zeros((5, 8))
    A[:5, :5] = np.identity(5)
    v = [1, 1, 1, 0, 0]
    w = [0, 0, 0, 1, 1]
    A[:, 5] = v
    A[:, 6] = w
    A[:, 7] = np.ones(5)
    W = np.random.rand(2)
    V = np.random.rand(3)
    S = np.zeros((5, 5))
    S[:3, :3] = W @ W.T
    S[3:, 3:] = V @ V.T
    print(la_admm_tagl(S, A, 1, 1, 1, 100, 10, 1e-7, 1e-4))
    return


test()
test_example_tree_big_lambda2()
test_example_tree_big_lambda1()
test_example_tree_glasso()
