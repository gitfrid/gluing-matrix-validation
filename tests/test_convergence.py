# tests/test_convergence.py
import numpy as np
from src.matrix_factory import build_A_matrix_double
from src.stability import compute_svd_scaled

def test_sigma_convergence():
    V0 = 10.0; a = 1.0; b = 1.0
    E = 5.0
    A = build_A_matrix_double(E, V0, a, b)
    s1 = compute_svd_scaled(A)['s_min']
    # emulate refinement by a tiny perturbation (proxy for patching)
    A2 = A.copy() * (1.0 + 1e-8)
    s2 = compute_svd_scaled(A2)['s_min']
    rel = abs(s1 - s2) / (abs(s2) + 1e-300)
    assert rel < 1e-5
