import numpy as np

def analytic_transmission_rect(E, V0, a, hbar=1.0, m=0.5):
    k = np.sqrt(2*m*E) / hbar
    if E < V0:
        kappa = np.sqrt(2*m*(V0 - E)) / hbar
        denom = 1 + (V0**2 / (4 * E * (V0 - E))) * (np.sinh(kappa * a)**2)
        T = 1.0 / denom
    else:
        q = np.sqrt(2*m*(E - V0)) / hbar
        denom = 1 + (V0**2 / (4 * E * (E - V0))) * (np.sin(q * a)**2)
        T = 1.0 / denom
    return float(np.real(T))

def analytic_transmission_double(E, V0, a, b, hbar=1.0, m=0.5):
    k = np.sqrt(2*m*E) / hbar
    if E < V0:
        kappa = np.sqrt(2*m*(V0 - E)) / hbar
        Mbar = np.array([[np.cosh(kappa*a), (1/kappa)*np.sinh(kappa*a)],
                         [kappa*np.sinh(kappa*a), np.cosh(kappa*a)]], dtype=complex)
    else:
        q = np.sqrt(2*m*(E - V0)) / hbar
        Mbar = np.array([[np.cos(q*a), (1/q)*np.sin(q*a)],
                         [-q*np.sin(q*a), np.cos(q*a)]], dtype=complex)
    P = np.array([[np.exp(1j*k*b), 0],[0, np.exp(-1j*k*b)]], dtype=complex)
    M = Mbar @ P @ Mbar
    denom = M[0,0] + M[0,1]*(1j*k) - M[1,0]/(1j*k) - M[1,1]
    T = 1.0 / np.abs(denom)**2
    return float(np.real(T))
