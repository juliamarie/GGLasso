import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
#import csv

from rpy2.robjects.conversion import localconverter

from gglasso.solver.tagl_admm_solver import la_admm_tagl
from gglasso.solver.single_admm_solver import ADMM_SGL

usethis = importr('usethis')
devtools = importr('devtools')
taglasso = importr('taglasso')

def test_example_tree_big_lambda2():
    A = np.zeros((5, 8))
    A[:5, :5] = np.identity(5)
    v = [1, 1, 1, 0, 0]
    w = [0, 0, 0, 1, 1]
    A[:, 5] = v
    A[:, 6] = w
    A[:, 7] = np.ones(5)
    W = np.random.rand(2,2)
    V = np.random.rand(3,3)
    S = np.zeros((5, 5))
    S[:3, :3] = V @ V.T
    S[3:, 3:] = W @ W.T
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
    W = np.random.rand(2, 2)
    V = np.random.rand(3, 3)
    S = np.zeros((5, 5))
    S[:3, :3] = V @ V.T
    S[3:, 3:] = W @ W.T
    Omega, Gamma, D = la_admm_tagl(S, A, 100, 0.1, 1, 100, 10, 1e-5, 1e-5)
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
    np.random.seed(10)
    W = np.random.rand(2,2)
    np.random.seed(10)
    V = np.random.rand(3,3)
    S = np.zeros((5, 5))
    S[:3, :3] = V @ V.T
    S[3:, 3:] = W @ W.T
    Omega, Gamma, D = la_admm_tagl(S, A, 0, 0.1, 1, 100, 40, 1e-7, 1e-5, verbose=True)
    sol, info = ADMM_SGL(S, 1, np.zeros((5, 5)), verbose=True)
    Omega2 = sol.get('Omega')
    for i in range(5):
        Omega[i][i] = 1
        Omega2[i][i] = 1
    for i in range(5):
        for j in range(5):
            if np.abs(Omega[i][j]) <= 1e-5:
                Omega[i][j] = 0
            if np.abs(Omega2[i][j]) <= 1e-5:
                Omega2[i][j] = 0
    print("Omega tag-lasso: ", Omega)
    print("Omega SGL: ", Omega2)
    assert_array_almost_equal(Omega, Omega2, 5)
    return


def test_example_tree_Rcode(lam1, lam2):
    p = 5
    A = np.zeros((5, 8))
    A[:5, :5] = np.identity(5)
    v = [1, 1, 1, 0, 0]
    w = [0, 0, 0, 1, 1]
    A[:, 5] = v
    A[:, 6] = w
    A[:, 7] = np.ones(5)
    np.random.seed(10)
    W = np.random.rand(2, 2)
    np.random.seed(10)
    V = np.random.rand(3, 3)
    np.random.seed(10)
    S = np.zeros((5, 5))
    S[:3, :3] = V @ V.T
    S[3:, 3:] = W @ W.T
    X = np.random.multivariate_normal(np.zeros(p), S, 100)
    #print(X)
    sdf = pd.DataFrame(X)
    S = np.cov(X.T)
    #print(S)
    #np.savetxt("matrix.csv", X, delimiter=",")
    # sdf = pd.DataFrame(S)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_s = robjects.conversion.py2rpy(sdf)
    A_vec = A.flatten()
    r_A_vec = robjects.FloatVector(A_vec)
    r_a = robjects.r['matrix'](r_A_vec, nrow=5)

    tag_lasso = robjects.r['taglasso']
    #print(S)
    print(la_admm_tagl(S, A, lam1, lam2, 1, 100, 10, 1e-7, 1e-20, verbose=False))
    print(tag_lasso(X=r_s, A=r_a, lambda1=lam1, lambda2=lam2, rho=1, it_in=100, it_out=10))


def test_example_tree_Rcode_glasso():
    A = np.zeros((5, 8))
    A[:5, :5] = np.identity(5)
    v = [1, 1, 1, 0, 0]
    w = [0, 0, 0, 1, 1]
    A[:, 5] = v
    A[:, 6] = w
    A[:, 7] = np.ones(5)
    np.random.seed(10)
    W = np.random.rand(2, 2)
    np.random.seed(10)
    V = np.random.rand(3, 3)
    S = np.zeros((5, 5))
    S[:3, :3] = V @ V.T
    S[3:, 3:] = W @ W.T
    sdf = pd.DataFrame(S)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_s = robjects.conversion.py2rpy(sdf)
    A_vec = A.flatten()
    r_A_vec = robjects.FloatVector(A_vec)
    r_a = robjects.r['matrix'](r_A_vec, nrow=5)

    tag_lasso = robjects.r['taglasso']
    print(tag_lasso(X=r_s, A=r_a, lambda1=0, lambda2=0.1, rho=1, it_in=100, it_out=10))
    sol, info = ADMM_SGL(S, 0.1, np.zeros((5, 5)), verbose=True)
    print(sol)
    return



def test():
    A = np.zeros((5, 8))
    A[:5, :5] = np.identity(5)
    v = [1, 1, 1, 0, 0]
    w = [0, 0, 0, 1, 1]
    A[:, 5] = v
    A[:, 6] = w
    A[:, 7] = np.ones(5)
    W = np.random.rand(2,2)
    V = np.random.rand(3,3)
    S = np.zeros((5, 5))
    S[:3, :3] = V @ V.T
    S[3:, 3:] = W @ W.T
    print(la_admm_tagl(S, A, 1, 0.1, 1, 100, 10, 1e-7, 1e-4))
    return


#test()
#test_example_tree_big_lambda2()
#test_example_tree_big_lambda1()
#test_example_tree_glasso()
test_example_tree_Rcode(0.1,0.1)
#test_example_tree_Rcode_glasso()
