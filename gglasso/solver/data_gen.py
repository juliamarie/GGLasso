import numpy as np
from gglasso.solver.tagl_admm_solver import la_admm_tagl
from gglasso.solver.tagl_admm_solver import admm_tagl
from numpy import genfromtxt
from gglasso.solver.tagl_admm_solver import function



data_admm1 = np.zeros((8, 1000))
data_admm2 = np.zeros((8, 1000))
data_la_admm1 = np.zeros((8, 1000))
data_la_admm2 = np.zeros((8, 1000))

l = 0
for k in [0.01, 0.1, 1, 2]:
    A = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/A.csv', delimiter=',')
    A = A[1:, :]
    X = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/data.csv', delimiter=',')
    X = X[1:, :]
    S = np.cov(X.T)
    p = len(S)
    T = len(A[0])
    U1 = np.zeros((p, p))
    U2 = np.zeros((p, p))
    U3 = np.zeros((p, p))
    U4 = np.zeros((T, p))
    U5 = np.zeros((T, p))


    lam1 = 0.1
    lam2 = 0.1
    i = 0


    Om, Gam, D, Om1, Om2, Om3, Gam1, Gam2, U1, U2, U3, U4, U5, stat, it, data_admm1[l*2:l*2+2] = admm_tagl(data_admm1[l*2:l*2+2], i, S, A, lam1,
                                                                                               lam2, rho=k,
                                                                                               t=1000,
                                                                                               Om0=np.zeros((p, p)),
                                                                                               Gam0=np.zeros((T, p)),
                                                                                               U1_init=U1, U2_init=U2,
                                                                                               U3_init=U3, U4_init=U4,
                                                                                               U5_init=U5,
                                                                                               tol=0, rtol=0)

    A = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/A.csv', delimiter=',')
    A = A[1:, :]
    X = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/data.csv', delimiter=',')
    X = X[1:, :]
    S = np.cov(X.T)
    p = len(S)
    T = len(A[0])
    U1 = np.zeros((p, p))
    U2 = np.zeros((p, p))
    U3 = np.zeros((p, p))
    U4 = np.zeros((T, p))
    U5 = np.zeros((T, p))
    i = 0

    Om, Gam, D, data_la_admm1[l*2:l*2+2] = la_admm_tagl(data_la_admm1[l*2:l*2+2], S, A, lam1, lam2, rho=k, t=100, K=10, tol=0, rtol=0)
    A = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/A.csv', delimiter=',')
    A = A[1:, :]
    X = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/data.csv', delimiter=',')
    X = X[1:, :]
    S = np.cov(X.T)
    p = len(S)
    T = len(A[0])
    U1 = np.zeros((p, p))
    U2 = np.zeros((p, p))
    U3 = np.zeros((p, p))
    U4 = np.zeros((T, p))
    U5 = np.zeros((T, p))


    lam1 = 0.2
    lam2 = 0.5
    i = 0

    Om, Gam, D, Om1, Om2, Om3, Gam1, Gam2, U1, U2, U3, U4, U5, stat, it, data_admm2[l*2:l*2+2] = admm_tagl(data_admm2[l*2:l*2+2], i, S, A, lam1,
                                                                                               lam2, rho=k,
                                                                                               t=1000,
                                                                                               Om0=np.zeros((p, p)),
                                                                                               Gam0=np.zeros((T, p)),
                                                                                               U1_init=U1, U2_init=U2,
                                                                                               U3_init=U3, U4_init=U4,
                                                                                               U5_init=U5,
                                                                                               tol=0, rtol=0)
    A = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/A.csv', delimiter=',')
    A = A[1:, :]
    X = genfromtxt('/Users/julia/Documents/Uni/Bachelor_Arbeit/code/GGLasso/gglasso/tests/data.csv', delimiter=',')
    X = X[1:, :]
    S = np.cov(X.T)
    p = len(S)
    T = len(A[0])
    U1 = np.zeros((p, p))
    U2 = np.zeros((p, p))
    U3 = np.zeros((p, p))
    U4 = np.zeros((T, p))
    U5 = np.zeros((T, p))
    i = 0

    Om, Gam, D, data_la_admm2[l*2:l*2+2] = la_admm_tagl(data_la_admm2[l*2:l*2+2], S, A, lam1, lam2, rho=k, t=100, K=10, tol=0, rtol=0)


    l = l+1
np.savetxt("data_for_plot_admm_pensens1.csv", data_admm1, delimiter=",")
np.savetxt("data_for_plot_admm_pensens2.csv", data_admm2, delimiter=",")
np.savetxt("data_for_plot_la_admm_pensens1.csv", data_la_admm1, delimiter=",")
np.savetxt("data_for_plot_la_admm_pensens2.csv", data_la_admm2, delimiter=",")
