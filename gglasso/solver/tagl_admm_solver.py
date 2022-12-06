import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

from gglasso.helper import tagl_admm_helper as tl
from gglasso.solver.ggl_helper import prox_1norm

# from gglasso.solver.single_admm_solver import ADMM_SGL


usethis = importr('usethis')
devtools = importr('devtools')
taglasso = importr('taglasso')


def admm_tagl(S, A, p, T, lam1, lam2, rho, t, Om0, Gam0, A_for_gamma, A_for_B, C, C_for_D, stat, eps_abs, eps_rel=1e-4):
    assert (np.linalg.norm(p) >= 1e-10), "Dimension of the problem should be greater than 0"
    # initialize
    U1 = Om0.copy()
    U2 = Om0.copy()
    U3 = Om0.copy()
    U4 = Gam0.copy()
    U5 = Gam0.copy()
    Om1 = np.zeros((p, p))  # Om0.copy()
    Om2 = np.zeros((p, p))  # Om0.copy()
    Om3 = np.zeros((p, p))  # Om0.copy()
    Gam1 = np.zeros((T, p))  # Gam0.copy()
    Gam2 = np.zeros((T, p))  # Gam0.copy()
    Om = Om0.copy() #np.zeros((p, p))  # np.identity(p)#Om0.copy()
    Gam = Gam0.copy() #np.zeros((T, p))  # Gam0.copy()#np.ones((T,p))
    D = np.zeros((p, p))
    it = 100000

    # \tilde A for \Gamma^{(2)} and D
    # Atilde = np.zeros((p + T, T))
    # Atilde[:p] = A
    # Atilde[p:] = np.identity(T)

    # (\tilde A^T * \tilde A)^-1 * \tilde A for \Gamma^{(2)} and D
    # AtildeInv = np.linalg.inv(np.transpose(Atilde) @ Atilde)  # TxT

    # C for D
    # J = np.zeros((p + T, p))
    # J[:p] = np.identity(p)
    # C = J - Atilde @ AtildeInv @ np.transpose(A)  # p+TxT = p+TxT TxT Txp

    # Atilde, A_for_gamma, A_for_B, C, C_for_D = tl.A_precompute(A, p, T)

    for k in range(t):
        # print(rho, k)

        # copy of last iterate
        Om_old = Om.copy()
        Gam_old = Gam.copy()

        # Update for \Omega^{(1)}
        rhs = (rho * Om) - U1 - S
        rhs = (rhs + rhs.T) / 2
        # print(np.transpose((rho * Om) - U1 - S))
        assert (np.linalg.norm(rhs - np.transpose(rhs)) <= 1e-10), "Expected symmetry"
        w, Q = np.linalg.eigh(rhs)
        delta_omega = tl.get_eigenvalues(w, p, rho)
        Om1 = Q @ delta_omega @ Q.T
        # print("Omega 1:", Om1)

        # Udate \Omega^{(3)} by soft-thresholding
        Om3 = prox_1norm(Om - U3 / rho, lam2 / rho)
        o = Om - U3 / rho
        for i in range(p):
            Om3[i][i] = o[i][i]
        # print("Omega 3: ", Om3)

        # Update \Gamma^{(1)} by groupwise soft-thresholding
        if lam1 == 0:
            Gam1 = Gam - U4 / rho
        else:
            for i in range(0, p - 1):
                Gam1[i] = tl.groupwise_st(Gam[i] - U4[i] / rho, lam1 / rho)
        Gam1[p - 1] = tl.vector_avrg(Gam[p-1] - U4[p-1] / rho, p)
        # print("Gamma 1: ", Gam1)

        # Update D, \Gamma^{(2)} and \Omega^{(2)}
        Mtilde = np.zeros((T + p, p))
        Mtilde[:p] = Om - U2 / rho
        Mtilde[p:] = Gam - U5 / rho
        B = A_for_B @ Mtilde
        #B = (np.identity(p + T) - (Atilde @ AtildeInv @ Atilde.T)) @ Mtilde
        #D = np.linalg.inv(tl.diag(C.T @ C, p)) * tl.diag_max(B.T @ C, p)
        D = C_for_D @ tl.diag_max(B.T @ C, p)
        # print("D:", D)
        assert (D >= 0).all(), "D should have only positive entries"
        Dtilde = np.zeros((p + T, p))
        Dtilde[:p] = D
        #Gam2 = AtildeInv @ Atilde.T @ (Mtilde - Dtilde)
        Gam2 = A_for_gamma @ (Mtilde - Dtilde)
        # print("Gamma 2: ", Gam2)
        Om2 = (A @ Gam2) + D
        # print("Omega 2: ", Om2)

        # Update \Omega
        Om = (Om1 + Om2 + Om3) / 3
        # print("Omega: ", Om)

        # Update \Gamma
        Gam = (Gam1 + Gam2) / 2
        # print("Gamma: ", Gam)

        # Update U1, U2, U3, U4 and U5
        U1 = U1 + rho * (Om1 - Om)
        # print("U 1: ", U1)
        U2 = U2 + rho * (Om2 - Om)
        # print("U 2: ", U2)
        U3 = U3 + rho * (Om3 - Om)
        # print("U 3: ", U3)
        U4 = U4 + rho * (Gam1 - Gam)
        # print("U 4: ", U4)
        U5 = U5 + rho * (Gam2 - Gam)
        # print("U 5: ", U5)

        s = tl.s_k(Om, Gam, Om_old, Gam_old, rho)
        r = tl.r_k(Om1, Om2, Om3, Gam1, Gam2, Om, Gam)
        eps_pri = tl.eps_pri(Om1, Om2, Om3, Gam1, Gam2, Om, Gam, eps_abs, eps_rel)
        eps_dual = tl.eps_dual(U1, U2, U3, U4, U5, eps_abs, eps_rel)
        # print(rho)
        # print(np.linalg.norm(r))

        if k>=1:
            if np.linalg.norm(r) <= eps_pri and np.linalg.norm(s) <= eps_dual:
                stat = 1
                it = k
                break

    return Om, Gam, D, stat, it


def la_admm_tagl(S, A, p, T, lam1, lam2, rho, t, K, eps_abs, eps_rel=1e-3):
    Om = np.zeros((p, p))
    Gam = np.zeros((T, p))
    D = np.zeros((p, p))
    Atilde, A_for_gamma, A_for_B, C, C_for_D = tl.A_precompute(A, p, T)
    stat = 0
    for l in range(K):
        Om, Gam, D, stat, it = admm_tagl(S, A, p, T, lam1, lam2, rho, t, Om, Gam, A_for_gamma, A_for_B, C, C_for_D, stat, eps_abs, eps_rel)

        # Update penalty parameter
        rho = 2 * rho
        if stat == 1:
            print("ended in iteration: ", l, it)
            break
    return Om, Gam, D


# p = 4
# T = p + 4
# rho = 1
# lam1 = 1000
# lam2 = 1000
# t = 100
# K = 10
#
# M = np.random.rand(p, p)
# U = np.tril(M)
# S = np.matmul(U, np.transpose(U))
# sdf = pd.DataFrame(S)
#
# with localconverter(robjects.default_converter + pandas2ri.converter):
#     r_s = robjects.conversion.py2rpy(sdf)
# r_vec = robjects.FloatVector([1.0, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])
# r_a = robjects.r['matrix'](r_vec, nrow=4)
#
# A = np.zeros((p, T))
# A[:, :p] = np.identity(p)
# A[:, p + 3] = np.ones(p)
# A[0][p] = 1
# A[1][p + 1] = 1
# for k in range(2, p):
#     A[k][p + 2] = 1
#
# adf = pd.DataFrame(A)
# # with localconverter(robjects.default_converter + pandas2ri.converter):
# #     r_a = robjects.conversion.py2rpy(adf)
#
#
# tag_lasso = robjects.r['taglasso']
#
# print("tag-lasso")
# #print(la_admm_tagl(S, np.identity(p), p, p, lam1, lam2, rho, t, K, 1e-19, 1e-4))
# print(la_admm_tagl(S, A, p, T, lam1, lam2, rho, t, K, 1e-19))
#
# print("................")
#
# print(np.linalg.inv(S))
#
#
# #print(tag_lasso(X=r_s, A=r_a, lambda1=lam1, lambda2=lam2, rho=rho))
#
# # print("glasso")
# # ADMM_SGL(S, lam2, np.zeros((p,p)))
