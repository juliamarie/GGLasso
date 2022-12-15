import numpy as np


from gglasso.solver.ggl_helper import prox_1norm


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


def diag_max(C):
    D = np.diag(np.maximum(np.zeros(len(C)), np.diag(C)))
    # for i in range(p):
    #     D[i][i] = np.maximum(0, C[i][i])
    return D


# functions for stopping criterion from Boyd
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
    A_t = np.zeros((T, p+T))
    A_t[:,:p] = A.T
    A_t[:,p:] = np.identity(T)

    A_inv = np.linalg.inv(A.T @ A + np.identity(T))
    Atilde_inv = np.linalg.inv(Atilde.T @ Atilde)
    #print("Atilde_inv: ")
    #print(Atilde_inv)
    #print("A_inv: ")
    #print(A_inv)

    A_for_gamma = A_inv @ A_t
    #print("Atilde @ Atilde_inv @ Atilde.T")
    #print(np.diag(Atilde @ Atilde_inv @ Atilde.T))
    A_for_B = np.identity(p + T) - Atilde @ Atilde_inv @ Atilde.T
    #print("A_for_B")
    #print(np.diag(A_for_B))

    J = np.zeros((p + T, p))
    J[:p] = np.identity(p)
    #print("Atilde @ Atilde_inv @ A.T")
    #print(Atilde @ Atilde_inv @ A.T)
    C = J - (Atilde @ Atilde_inv @ A.T)
    #print("C")
    #print(C)

    #C_for_D = np.linalg.inv(np.diag(np.diag(C.T@C)))

    C_for_D = np.diag(1/np.diag(C.T @ C))

    return Atilde, A_for_gamma, A_for_B, C, C_for_D


def admm_tagl(S, A, lam1, lam2, rho, t, Om0, Gam0, A_for_gamma=None, A_for_B=None, C=None, C_for_D=None, tol=1e-5, rtol=1e-4, verbose=False):
    # initialize
    stat = 0
    p = len(S)
    T = len(A[0])
    U1 = np.zeros((p,p)) #Om0.copy()
    U2 = np.zeros((p,p)) #Om0.copy()
    U3 = np.zeros((p,p)) #Om0.copy()
    U4 = np.zeros((T,p)) #Gam0.copy()
    U5 = np.zeros((T,p)) #Gam0.copy()
    Om1 = np.zeros((p, p))  # Om0.copy()
    Om2 = np.zeros((p, p))  # Om0.copy()
    Om3 = np.zeros((p, p))  # Om0.copy()
    Gam1 = np.zeros((T, p))  # Gam0.copy()
    Gam2 = np.zeros((T, p))  # Gam0.copy()
    Om = Om0.copy()  # np.zeros((p, p))  # np.identity(p)#Om0.copy()
    Gam = Gam0.copy()  # np.zeros((T, p))  # Gam0.copy()#np.ones((T,p))
    D = np.zeros((p, p))
    it = 100000

    if A_for_gamma is None:
        Atilde, A_for_gamma, A_for_B, C, C_for_D = A_precompute(A, p, T)

    hdr_fmt = "%4s\t%10s\t%10s\t%10s\t%10s"
    out_fmt = "%4d\t%10.4g\t%10.4g\t%10.4g\t%10.4g"
    if verbose:
        print("------------ADMM Algorithm for tag-lasso----------------")
        print(hdr_fmt % ("iter", "r_t", "s_t", "eps_pri", "eps_dual"))

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
        rhs = rho * Om - U1 - S
        rhs = (rhs + rhs.T) / 2
        # print(np.transpose((rho * Om) - U1 - S))
        # assert (np.linalg.norm(((rho * Om) - U1 - S) - np.transpose((rho * Om) - U1 - S)) <= 1e-10), "Expected symmetry"
        w, Q = np.linalg.eigh(rhs)
        L = np.zeros((p, p))
        #print("rhs: ", rhs)
        np.fill_diagonal(L, w)
        #print("Versuch: ", Q @ L @ Q.T)
        delta_omega = get_eigenvalues(w, p, rho)
        Om1 = Q @ delta_omega @ Q.T
        #print("Omega 1:", Om1)

        # Udate \Omega^{(3)} by soft-thresholding
        Om3 = prox_1norm(Om - U3 / rho, lam2 / rho)
        o = Om - U3 / rho
        for i in range(p):
            Om3[i][i] = o[i][i]
        #print("Omega 3: ", Om3)

        # Update \Gamma^{(1)} by groupwise soft-thresholding
        if lam1 == 0:
            Gam1 = Gam - U4 / rho
        else:
            for i in range(0, T - 1):
                Gam1[i] = groupwise_st(Gam[i] - U4[i] / rho, lam1 / rho)
        Gam1[T - 1] = (Gam[T - 1] - U4[T - 1] / rho).mean()
        #print("Gamma 1: ", Gam1)

        # Update D, \Gamma^{(2)} and \Omega^{(2)}
        Mtilde = np.zeros((T + p, p))
        Mtilde[:p] = Om - U2 / rho
        Mtilde[p:] = Gam - U5 / rho
        B = A_for_B @ Mtilde
        # B = (np.identity(p + T) - (Atilde @ AtildeInv @ Atilde.T)) @ Mtilde
        # D = np.linalg.inv(tl.diag(C.T @ C, p)) * tl.diag_max(B.T @ C, p)
        D = C_for_D @ diag_max(B.T @ C)
        #print("D:", D)
        assert (D >= 0).all(), "D should have only positive entries"
        Dtilde = np.zeros((p + T, p))
        Dtilde[:p] = D
        # Gam2 = AtildeInv @ Atilde.T @ (Mtilde - Dtilde)
        Gam2 = A_for_gamma @ (Mtilde - Dtilde)
        #print("Gamma 2: ", Gam2)
        Om2 = (A @ Gam2) + D
        #print("Omega 2: ", Om2)

        # Update \Omega
        Om = (Om1 + Om2 + Om3) / 3
        #print("Omega: ", Om)

        # Update \Gamma
        Gam = (Gam1 + Gam2) / 2
        #print("Gamma: ", Gam)

        # Update U1, U2, U3, U4 and U5
        U1 = U1 + rho * (Om1 - Om)
        #print("U 1: ", U1)
        U2 = U2 + rho * (Om2 - Om)
        #print("U 2: ", U2)
        U3 = U3 + rho * (Om3 - Om)
        #print("U 3: ", U3)
        U4 = U4 + rho * (Gam1 - Gam)
        #print("U 4: ", U4)
        U5 = U5 + rho * (Gam2 - Gam)
        #print("U 5: ", U5)

        s = s_k(Om, Gam, Om_old, Gam_old, rho)
        s_k_n = np.linalg.norm(s)
        r = r_k(Om1, Om2, Om3, Gam1, Gam2, Om, Gam)
        r_k_n = np.linalg.norm(r)
        eps_p = eps_pri(Om1, Om2, Om3, Gam1, Gam2, Om, Gam, tol, rtol)
        eps_d = eps_dual(U1, U2, U3, U4, U5, tol, rtol)
        # print(rho)
        # print(np.linalg.norm(r))

        if verbose:
            print(out_fmt % (k, r_k_n, s_k_n, eps_p, eps_d))

        if k >= 1:
            if r_k_n <= eps_p and s_k_n <= eps_d:
                stat = 1
                it = k
                break

    return Om, Gam, D, Om1, Om2, Om3, Gam1, Gam2, U1, U2, U3, U4, U5, stat, it


def la_admm_tagl(S, A, lam1, lam2, rho, t, K, tol, rtol=1e-3, verbose=False):
    p = len(S)
    T = len(A[0])
    Om = np.zeros((p, p))
    Gam = np.zeros((T, p))
    D = np.zeros((p, p))
    Atilde, A_for_gamma, A_for_B, C, C_for_D = A_precompute(A, p, T)
    stat = 0
    for l in range(K):
        Om, Gam, D, Om1, Om2, Om3, Gam1, Gam2, U1, U2, U3, U4, U5,stat, it = admm_tagl(S, A, lam1, lam2, rho, t, Om, Gam, A_for_gamma, A_for_B, C, C_for_D, tol, rtol,
                                         verbose)

        # Update penalty parameter
        rho = 2 * rho
        if stat == 1:
            print("ended in iteration: ", l, it)
            break
    return Om, Gam, D
