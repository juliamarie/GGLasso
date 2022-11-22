import numpy as np

from gglasso.helper import tagl_admm_helper as tl
from gglasso.solver.ggl_helper import prox_1norm


def admm_tagl(S, A, p, T, lam1, lam2, rho, t, Om0, Gam0):
    # initialize
    U1 = Om0
    U2 = Om0
    U3 = Om0
    U4 = Gam0
    U5 = Gam0
    Om1 = U1
    Om2 = U2
    Om3 = U3
    Gam1 = U4
    Gam2 = U5
    Om = Om0
    Gam = Gam0#np.ones((T,p))
    D = np.zeros((p, p))
    J = np.zeros((p + T, p))
    J[:p] = np.identity(p)
    Atilde = np.zeros((p + T, T))
    Atilde[:p] = A
    Atilde[p:] = np.identity(T)
    AtildeInv = np.linalg.inv(np.transpose(Atilde) @ Atilde) # TxT
    C = J - Atilde @ AtildeInv @ np.transpose(A) # p+TxT = p+TxT TxT Txp

    for k in range(t):
        w, Q = np.linalg.eigh(rho * Om - U1 - S)
        #assert np.all(w>=0), "Expected positive eigenvalues"
        delta_omega = tl.get_eigenvalues(w, p, rho)
        Om1 = Q @ delta_omega @  np.transpose(Q)
        Om3 = prox_1norm(Om - U3 / rho, lam2 / rho)
        for i in range(0, p-1):
            Gam1[i] = tl.groupwise_st(Gam[i] - U4[i] / rho, lam1 / rho)
        Gam1[p-1] = tl.vector_avrg(Gam[0] - U4[0] / rho, p)

        Mtilde = np.zeros((T + p, p))
        print(rho)
        print(Om)
        Mtilde[:p] = Om - U2 / rho
        Mtilde[p:] = Gam - U5 / rho
        B = np.matmul(np.identity(p + T) - np.matmul(Atilde, np.matmul(AtildeInv, np.transpose(Atilde))), Mtilde)

        D = np.linalg.inv(tl.diag(np.matmul(np.transpose(C), C), p)) * tl.diag_max(np.matmul(np.transpose(B), C), p)
        Dtilde = np.zeros((p + T, p))
        Dtilde[:p] = D
        Gam2 = np.matmul(AtildeInv, np.matmul(np.transpose(Atilde), Mtilde - Dtilde))
        Om2 = np.matmul(A, Gam2) + D
        Om = (Om1 + Om2 + Om3) / 3
        Gam = (Gam1 + Gam2) / 2
        U1 = U1 + rho * (Om1 - Om)
        U2 = U2 + rho * (Om2 - Om)
        U3 = U3 + rho * (Om3 - Om)
        U4 = U4 + rho * (Gam1 - Gam)
        U5 = U5 + rho * (Gam2 - Gam)

    return Om, Gam, D


def la_admm_tagl(S, A, p, T, lam1, lam2, rho, t, K):
    Om = np.zeros((p, p))
    print(1)
    Gam = np.zeros((T, p))
    print(2)
    D = np.zeros((p, p))
    print(3)
    for l in range(K):
        Om, Gam, D = admm_tagl(S, A, p, T, lam1, lam2, rho, t, Om, Gam)
        rho = 2 * rho
    return Om, Gam, D



p=10
T = p+4
rho = 1
lam1 = 0
lam2 = 1
t = 10
K = 10

M=np.random.rand(p,p)
U = np.tril(M)
S = np.matmul(U,np.transpose(U))



A = np.zeros((p,T))
A[:, :p] = np.identity(p)
A[:, p+3] =np.ones(p)
A[0][p] = 1
A[1][p+1] = 1
for k in range(2,p):
    A[k][p+2] = 1



la_admm_tagl(S, A, p, T, lam1, lam2, rho, t, K)