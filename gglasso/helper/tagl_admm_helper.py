import numpy as np


def get_eigenvalues(evals, p, rho):
    # get eigenvalues of next iterate of \Omega^{(1)}, input are eigenvalues of \rho\Omega_k-U_k-S and dimension p
    omega = np.zeros((p, p))
    np.fill_diagonal(omega, ((evals + np.sqrt(evals ** 2 + 4 * rho)) / (2 * rho)))
    # for i in range(p):
    # omega[i][i] = (evals[i] + np.sqrt(evals[i] ** 2 + 4 * rho)) / (2 * rho)
    return omega


def groupwise_st(gamma, lamb):
    # groupwise soft thresholding, gamma np.array 1D, lamb scalar
    u = np.linalg.norm(gamma)
    if u <= 1e-10:
        res = np.zeros(len(gamma))
    else:
        a = 1 - lamb / u  # eucl. norm for vector
        res = gamma * np.maximum(a, 0.)
    return res


def vector_avrg(gamma, p):
    # \sum_{i=1}^p gamma_i/p with gamma as a vector and p the matching dimension
    s = 0
    for i in range(p):
        s = s + gamma[i]
    return s / p


def diag(C, p):
    # get only diagonal entries of C into a diagonal matrix with dimension p
    D = np.zeros((p, p))
    d = np.diagonal(C)
    np.fill_diagonal(D, d)
    # for i in range(p):
    #     D[i][i] = C[i][i]
    return D


def diag_max(C, p):
    D = np.zeros((p, p))
    np.fill_diagonal(D, np.diagonal(np.maximum(0., C)))
    # for i in range(p):
    #     D[i][i] = np.maximum(0, C[i][i])
    return D


######### functions for stopping criterion from Boyd
def s_k(Om, Gam, Om_old, Gam_old, rho):
    p = len(Om)
    T = len(Gam)
    A = np.identity((p * 3 + T * 2))
    B = np.zeros(((p * 3 + T * 2), p + T))
    B[:p, :p] = np.identity(p)
    B[p:(p + p), :p] = np.identity(p)
    B[(2 * p):(3 * p), :p] = np.identity(p)
    B[(3 * p):(3 * p + T), p:(p + T)] = np.identity(T)
    B[(3 * p + T):(3 * p + 2 * T), p:(p + T)] = np.identity(T)
    B = -1 * B
    z_old = np.zeros((p + T, p))
    z_old[:p] = Om_old
    z_old[p:p + T] = Gam_old
    z = np.zeros((p + T, p))
    z[:p] = Om
    z[p:p + T] = Gam
    res = rho * (A.T @ B) @ (z - z_old)
    return res


def r_k(Om1, Om2, Om3, Gam1, Gam2, Om, Gam):
    p = len(Om1)
    T = len(Gam1)
    A = np.identity((p * 3 + T * 2))
    B = np.zeros(((p * 3 + T * 2), p + T))
    B[:p, :p] = np.identity(p)
    B[p:(p + p), :p] = np.identity(p)
    B[(2 * p):(3 * p), :p] = np.identity(p)
    B[(3 * p):(3 * p + T), p:(p + T)] = np.identity(T)
    B[(3 * p + T):(3 * p + 2 * T), p:(p + T)] = np.identity(T)
    B = -1 * B
    x = np.zeros((3 * p + 2 * T, p))
    x[:p] = Om1
    x[p:2 * p] = Om2
    x[2 * p:3 * p] = Om3
    x[3 * p:3 * p + T] = Gam1
    x[3 * p + T:] = Gam2
    z = np.zeros((p + T, p))
    z[:p] = Om
    z[p:p + T] = Gam
    res = A @ x + B @ z
    return res


def eps_pri(Om1, Om2, Om3, Gam1, Gam2, Om, Gam, eps_abs, eps_rel=1e-3):
    assert (eps_abs > 0), "eps_abs should be positive"
    assert (eps_rel > 0), "eps_rel should be positive"
    p = len(Om1)
    T = len(Gam1)
    n = 3 * p + 2 * T
    A = np.identity((p * 3 + T * 2))
    B = np.zeros(((p * 3 + T * 2), p + T))
    B[:p, :p] = np.identity(p)
    B[p:(p + p), :p] = np.identity(p)
    B[(2 * p):(3 * p), :p] = np.identity(p)
    B[(3 * p):(3 * p + T), p:(p + T)] = np.identity(T)
    B[(3 * p + T):(3 * p + 2 * T), p:(p + T)] = np.identity(T)
    B = -1 * B
    x = np.zeros((3 * p + 2 * T, p))
    x[:p] = Om1
    x[p:2 * p] = Om2
    x[2 * p:3 * p] = Om3
    x[3 * p:3 * p + T] = Gam1
    x[3 * p + T:] = Gam2
    z = np.zeros((p + T, p))
    z[:p] = Om
    z[p:p + T] = Gam
    res = np.sqrt(n) * eps_abs + eps_rel * np.maximum(np.linalg.norm(A @ x), np.linalg.norm(B @ z))
    return res


def eps_dual(U1, U2, U3, U4, U5, eps_abs, eps_rel=1e-3):
    assert (eps_abs > 0), "eps_abs should be positive"
    assert (eps_rel > 0), "eps_rel should be positive"
    p = len(U1)
    T = len(U4)
    n = 3 * p + 2 * T
    A = np.identity((p * 3 + T * 2))
    y = np.zeros((3 * p + 2 * T, p))
    y[:p] = U1
    y[p:2 * p] = U2
    y[2 * p:3 * p] = U3
    y[3 * p:3 * p + T] = U4
    y[3 * p + T:] = U5
    res = np.sqrt(n) * eps_abs + eps_rel * np.linalg.norm(A @ y)
    return res


def A_precompute(A, p, T):
    Atilde = np.zeros((p + T, T))
    Atilde[:p] = A
    Atilde[p:] = np.identity(T)

    A_inv = np.linalg.inv(A.T @ A + np.identity(T))

    A_for_gamma = A_inv @ Atilde.T

    A_for_B = np.identity(p+T) - Atilde @ A_inv @ Atilde.T

    J = np.zeros((p + T, p))
    J[:p] = np.identity(p)
    C = J - Atilde @ A_inv @ A.T

    C_for_D = np.linalg.inv(diag(C, p))

    return Atilde, A_for_gamma, A_for_B, C, C_for_D

# # p=5
# # rho=0.5
# # eval = [1,2,3,4,5]
# # evals = np.array(eval)
# # print(eval*2)
# ma = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
# man = np.array(ma)
# print(man)
#
# w, v = np.linalg.eig(man)
# D = np.zeros((3, 3))
# for i in range(3):
#     D[i][i] = w[i]
# print("eigenvalues: ", w)
# print("eigenvektoren: ", v)
#
# mon = v * D * np.matrix.transpose(v)
# D2 = np.matmul(np.matmul(np.matrix.transpose(v), man), v)
# print("urspruengliche Matrix: ", D2)
#
# # print(get_eigenvalues(evals, p, rho))
#
#
# A = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
# An = np.array(A)
# B = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
# Bn = np.array(B)
#
# # print("normal: ",A*B)
# print("numpy: ", An * Bn)
