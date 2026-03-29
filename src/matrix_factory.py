import numpy as np

def build_A_matrix_single(E, V0, a, hbar=1.0, m=0.5):
    k = np.sqrt(2*m*E) / hbar
    if E < V0:
        kappa = np.sqrt(2*m*(V0 - E)) / hbar
    else:
        kappa = 1j * np.sqrt(2*m*(E - V0)) / hbar
    exp_kappa_a = np.exp(kappa * a)
    exp_minus_kappa_a = np.exp(-kappa * a)
    exp_ik_a = np.exp(1j * k * a)
    A = np.zeros((4,5), dtype=complex)
    A[0,:5] = [1, 1, -1, -1, 0]
    A[1,:5] = [1j*k, -1j*k, -kappa, kappa, 0]
    A[2,:5] = [0, 0, exp_kappa_a, exp_minus_kappa_a, -exp_ik_a]
    A[3,:5] = [0, 0, kappa*exp_kappa_a, -kappa*exp_minus_kappa_a, -1j*k*exp_ik_a]
    return A

def build_A_matrix_double(E, V0, a, b, hbar=1.0, m=0.5):
    k = np.sqrt(2*m*E) / hbar
    if E < V0:
        kappa = np.sqrt(2*m*(V0 - E)) / hbar
    else:
        kappa = 1j * np.sqrt(2*m*(E - V0)) / hbar
    exp = np.exp
    x0 = 0.0
    x1 = a
    x2 = a + b
    x3 = 2*a + b
    A = np.zeros((8,9), dtype=complex)
    A[0,0] = exp(1j*k*x0); A[0,1] = exp(-1j*k*x0); A[0,2] = -exp(kappa*x0); A[0,3] = -exp(-kappa*x0)
    A[1,0] = 1j*k*exp(1j*k*x0); A[1,1] = -1j*k*exp(-1j*k*x0); A[1,2] = -kappa*exp(kappa*x0); A[1,3] = kappa*exp(-kappa*x0)
    A[2,2] = exp(kappa*x1); A[2,3] = exp(-kappa*x1); A[2,4] = -exp(1j*k*x1); A[2,5] = -exp(-1j*k*x1)
    A[3,2] = kappa*exp(kappa*x1); A[3,3] = -kappa*exp(-kappa*x1); A[3,4] = -1j*k*exp(1j*k*x1); A[3,5] = 1j*k*exp(-1j*k*x1)
    A[4,4] = exp(1j*k*x2); A[4,5] = exp(-1j*k*x2); A[4,6] = -exp(kappa*x2); A[4,7] = -exp(-kappa*x2)
    A[5,4] = 1j*k*exp(1j*k*x2); A[5,5] = -1j*k*exp(-1j*k*x2); A[5,6] = -kappa*exp(kappa*x2); A[5,7] = kappa*exp(-kappa*x2)
    A[6,6] = exp(kappa*x3); A[6,7] = exp(-kappa*x3); A[6,8] = -exp(1j*k*x3)
    A[7,6] = kappa*exp(kappa*x3); A[7,7] = -kappa*exp(-kappa*x3); A[7,8] = -1j*k*exp(1j*k*x3)
    return A
