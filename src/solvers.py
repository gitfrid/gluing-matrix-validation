# src/solvers.py
import numpy as np

def analytic_transmission_rect(E, V0, a, hbar=1.0, m=0.5):
    """
    Analytic transmission for a single rectangular barrier.
    Returns a real float in [0,1].
    """
    E = float(E)
    V0 = float(V0)
    a = float(a)
    # avoid division by zero for E==0 or E==V0 by small eps
    eps = 1e-300
    if E <= 0:
        return 0.0
    k = np.sqrt(2.0 * m * E) / hbar
    if E < V0:
        kappa = np.sqrt(max(0.0, 2.0 * m * (V0 - E))) / hbar
        # use sinh for evanescent region
        sinh_term = np.sinh(kappa * a)
        denom = 1.0 + (V0**2 / (4.0 * E * (V0 - E) + eps)) * (sinh_term**2)
        T = 1.0 / denom
    else:
        q = np.sqrt(max(0.0, 2.0 * m * (E - V0))) / hbar
        sin_term = np.sin(q * a)
        denom = 1.0 + (V0**2 / (4.0 * E * (E - V0) + eps)) * (sin_term**2)
        T = 1.0 / denom
    # ensure real and in [0,1]
    T = float(np.real(T))
    if T < 0.0:
        T = 0.0
    if T > 1.0:
        T = 1.0
    return T

def analytic_transmission_double(E, V0, a, b, hbar=1.0, m=0.5):
    """
    Analytic transmission for a symmetric double rectangular barrier using transfer matrices.
    Returns a real float in [0,1].
    """
    E = float(E)
    V0 = float(V0)
    a = float(a)
    b = float(b)
    eps = 1e-300
    if E <= 0:
        return 0.0
    k = np.sqrt(2.0 * m * E) / hbar
    if E < V0:
        kappa = np.sqrt(max(0.0, 2.0 * m * (V0 - E))) / hbar
        Mbar = np.array([[np.cosh(kappa * a), (1.0 / (kappa + eps)) * np.sinh(kappa * a)],
                         [kappa * np.sinh(kappa * a), np.cosh(kappa * a)]], dtype=complex)
    else:
        q = np.sqrt(max(0.0, 2.0 * m * (E - V0))) / hbar
        Mbar = np.array([[np.cos(q * a), (1.0 / (q + eps)) * np.sin(q * a)],
                         [-q * np.sin(q * a), np.cos(q * a)]], dtype=complex)
    P = np.array([[np.exp(1j * k * b), 0.0], [0.0, np.exp(-1j * k * b)]], dtype=complex)
    M = Mbar @ P @ Mbar
    # For a transfer matrix M, transmission T = 1 / |M[0,0]|^2 for unit incident flux (symmetric leads)
    denom = M[0, 0]
    denom_abs2 = np.abs(denom)**2 + eps
    T = 1.0 / denom_abs2
    T = float(np.real(T))
    if T < 0.0:
        T = 0.0
    if T > 1.0:
        T = 1.0
    return T
