import numpy as np
from scipy.linalg import svd

def row_scale_matrix(A):
    row_max = np.max(np.abs(A), axis=1)
    row_max = np.where(row_max == 0, 1.0, row_max)
    D_inv = np.diag(1.0 / row_max)
    A_s = D_inv @ A
    return A_s, row_max

def compute_svd_scaled(A, use_longdouble=False):
    """
    Compute SVD on row-scaled matrix. Returns sorted singulars and relative singulars.
    """
    if use_longdouble:
        try:
            A = A.astype(np.complex128)
        except Exception:
            pass
    A_s, row_scales = row_scale_matrix(A)
    s = svd(A_s, compute_uv=False)
    s_sorted = np.sort(s)
    s_rel = s_sorted / (np.max(s_sorted) + 1e-300)
    return {
        "s_sorted": s_sorted,
        "s_rel": s_rel,
        "s_min": float(s_sorted[0]),
        "s_max": float(s_sorted[-1]),
        "row_scales": row_scales
    }
