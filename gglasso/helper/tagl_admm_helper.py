import numpy as np


def get_eigenvalues(evals, p, rho):
    # get eigenvalues of next iterate of \Omega^{(1)}, input are eigenvalues of \rho\Omega_k-U_k-S and dimension p
    omega = np.zeros((p, p))
    for i in range(p):
        omega[i][i] = (evals[i] + np.sqrt(evals[i] ** 2 + 4 * rho)) / 2 * rho
    return omega


def groupwise_st(gamma, lamb):
    # groupwise soft thresholding, gamma np.array 1D, lamb scalar
    u = np.linalg.norm(gamma)
    if u <= 1e-10:
        res = np.zeros(len(gamma))
    else:
        a = 1 - lamb / u  # eucl. norm for vector
        res = gamma * np.maximum(a, 0)
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
    for i in range(p):
        D[i][i] = C[i][i]
    return D


def diag_max(C, p):
    D = np.zeros((p, p))
    for i in range(p):
        D[i][i] = np.maximum(0, C[i][i])
    return D


# p=5
# rho=0.5
# eval = [1,2,3,4,5]
# evals = np.array(eval)
# print(eval*2)
ma = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
man = np.array(ma)
print(man)

w, v = np.linalg.eig(man)
D = np.zeros((3, 3))
for i in range(3):
    D[i][i] = w[i]
print("eigenvalues: ", w)
print("eigenvektoren: ", v)

mon = v * D * np.matrix.transpose(v)
D2 = np.matmul(np.matmul(np.matrix.transpose(v), man), v)
print("urspruengliche Matrix: ", D2)

# print(get_eigenvalues(evals, p, rho))


A = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
An = np.array(A)
B = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
Bn = np.array(B)

# print("normal: ",A*B)
print("numpy: ", An * Bn)
