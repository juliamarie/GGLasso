import numpy as np

from gglasso.helper import tagl_admm_helper as tl
from gglasso.solver.ggl_helper import prox_1norm
from gglasso.solver.single_admm_solver import ADMM_SGL


def admm_tagl(S, A, p, T, lam1, lam2, rho, t, Om0, Gam0):
    # initialize
    U1 = np.zeros((p,p))#Om0.copy()
    U2 = np.zeros((p,p))#Om0.copy()
    U3 = np.zeros((p,p))#Om0.copy()
    U4 = np.zeros((T,p))#Gam0.copy()
    U5 = np.zeros((T,p))#Gam0.copy()
    Om1 = Om0.copy()
    Om2 = Om0.copy()
    Om3 = Om0.copy()
    Gam1 = Gam0.copy()
    Gam2 = Gam0.copy()
    Om = Om0.copy()#np.identity(p)#Om0.copy()
    Gam = Gam0.copy()#np.ones((T,p))
    D = np.zeros((p, p))
    J = np.zeros((p + T, p))
    J[:p] = np.identity(p)
    Atilde = np.zeros((p + T, T))
    Atilde[:p] = A
    Atilde[p:] = np.identity(T)
    AtildeInv = np.linalg.inv(np.transpose(Atilde) @ Atilde) # TxT
    C = J - Atilde @ AtildeInv @ np.transpose(A) # p+TxT = p+TxT TxT Txp

    for k in range(t):
        print(rho, k)
        print((rho * Om) - U1 - S)
        #print(np.transpose((rho * Om) - U1 - S))
        assert(np.linalg.norm((rho * Om) - U1 - S - np.transpose((rho * Om) - U1 - S)) <= 1e-2), "Expected symmetry"
        # assert ((rho * Om) - U1 - S == np.transpose((rho * Om) - U1 - S)).all(), "Expected symmetry"
        w, Q = np.linalg.eigh((rho * Om) - U1 - S)
        #assert np.all(w>=0), "Expected positive eigenvalues"
        delta_omega = tl.get_eigenvalues(w, p, rho)
        Om1 = Q @ delta_omega @  Q.T
        assert (np.linalg.norm(Om1-Om1.T)<= 1e-10), "Expected symmetry Om1"
        #print("Omega 1:", Om1)
        Om3 = prox_1norm(Om - U3 / rho, lam2 / rho)
        assert (np.linalg.norm(Om3-Om3.T)<= 1e-10), "Expected symmetry Om3"
        #print("Omega 3: ",Om3)
        for i in range(0, p-1):
            Gam1[i] = tl.groupwise_st(Gam[i] - U4[i] / rho, lam1 / rho)
        Gam1[p-1] = tl.vector_avrg(Gam[0] - U4[0] / rho, p)
        #print("Gamma 1: ",Gam1)

        Mtilde = np.zeros((T + p, p))
        #print(rho)
        #print(Om)
        Mtilde[:p] = Om - U2 / rho
        Mtilde[p:] = Gam - U5 / rho
        B = np.matmul(np.identity(p + T) - np.matmul(Atilde, np.matmul(AtildeInv, np.transpose(Atilde))), Mtilde)

        D = np.linalg.inv(tl.diag(np.matmul(np.transpose(C), C), p)) * tl.diag_max(np.matmul(np.transpose(B), C), p)
        #print("D:",D)
        assert (D >= -1e-12).all(), "D should have only positive entries"
        Dtilde = np.zeros((p + T, p))
        Dtilde[:p] = D
        Gam2 = np.matmul(AtildeInv, np.matmul(np.transpose(Atilde), Mtilde - Dtilde))
        #print("Gamma 2: ",Gam2)
        Om2 = np.matmul(A, Gam2) + D
        assert (np.linalg.norm(Om2 - Om2.T) <= 1e-8), "Expected symmetry Om2"
        #print("Omega 2: ",Om2)
        Om = (Om1 + Om2 + Om3) / 3
        print("Omega: ", Om)
        Gam = (Gam1 + Gam2) / 2
        #print("Gamma: ",Gam)
        U1 = U1 + rho * (Om1 - Om)
        #print("U 1: ",U1)
        U2 = U2 + rho * (Om2 - Om)
        #print("U 2: ",U2)
        U3 = U3 + rho * (Om3 - Om)
        #print("U 3: ", U3)
        U4 = U4 + rho * (Gam1 - Gam)
        #print("U 4: ", U4)
        U5 = U5 + rho * (Gam2 - Gam)
        #print("U 5: ", U5)

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



p=5
T = p+4
rho = 1
lam1 = 0
lam2 = 1
t = 500
K = 1

M=np.random.rand(p, p)
U = np.tril(M)
S = np.matmul(U, np.transpose(U))



A = np.zeros((p,T))
A[:, :p] = np.identity(p)
A[:, p+3] =np.ones(p)
A[0][p] = 1
A[1][p+1] = 1
for k in range(2,p):
    A[k][p+2] = 1


B = [[-1.09630064, -7.74071667, -0.61274897, -0.69654005, -0.27135812],
     [-0.7370941,  -2.13471069, -1.04108949, -1.33654979, -0.68054737],
     [-0.60936192, -1.04786931, -1.86430678, -1.04665832, -0.51015774],
     [-0.69276482, -1.33883327, -1.04332615, -2.41221536, -0.69161099],
     [-0.27746328, -0.69095481, -0.51254993, -0.69733535, -0.98852835]]

print(np.linalg.eigh(B))


print("tag-lasso")
la_admm_tagl(S, A, p, T, lam1, lam2, rho, t, K)

print("glasso")
ADMM_SGL(S, lam2, np.zeros((p,p)))