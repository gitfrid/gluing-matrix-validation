# src/analytics.py
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Tuple, Dict

def gap_protocol(s_rel, G_thresh: float = 1e3, delta: float = 1e-6) -> Tuple[int, np.ndarray]:
    """
    Identify a large gap in the relative singular spectrum s_rel (ascending order).
    Returns (k, gaps) where k is the multiplicity index (0 if none found).
    k is the number of small singulars (i.e., gap after index k-1).
    """
    s_rel = np.asarray(s_rel, dtype=float)
    if s_rel.size < 2:
        return 0, np.array([])
    # avoid division by zero
    denom = s_rel[:-1] + 1e-300
    gaps = s_rel[1:] / denom
    candidates = np.where((gaps > G_thresh) & (s_rel[:-1] < delta))[0]
    if candidates.size == 0:
        return 0, gaps
    k = int(candidates[0]) + 1
    return k, gaps

def regression_loglog(x, y) -> Dict[str, float]:
    """
    Perform linear regression in log10-log10 space for positive x,y.
    Returns slope, intercept, r2, pearson, spearman (NaN if not computable).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan, pearson=np.nan, spearman=np.nan)
    lx = np.log10(x[mask])
    ly = np.log10(y[mask])
    # handle degenerate case where lx or ly have zero variance
    if np.allclose(lx, lx[0]) or np.allclose(ly, ly[0]):
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan, pearson=np.nan, spearman=np.nan)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = np.sum((ly - pred)**2)
    ss_tot = np.sum((ly - np.mean(ly))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    try:
        pearson = pearsonr(lx, ly)[0]
    except Exception:
        pearson = np.nan
    try:
        spearman = spearmanr(lx, ly)[0]
    except Exception:
        spearman = np.nan
    return dict(slope=float(slope), intercept=float(intercept), r2=float(r2), pearson=float(pearson), spearman=float(spearman))
