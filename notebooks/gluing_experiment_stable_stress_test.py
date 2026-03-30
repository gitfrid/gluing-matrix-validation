#!/usr/bin/env python3
# gluing_experiment_stable_with_stress.py

from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr

# -------------------------
# Imports
# -------------------------
try:
    from src.matrix_factory import build_A_matrix_double
    from src.stability import compute_svd_scaled
    from src.solvers import analytic_transmission_double
except Exception as exc:
    raise ImportError(
        "Failed to import project modules. Run `pip install -e .` or set PYTHONPATH."
    ) from exc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("gluing_experiment")


# =========================================================
# 🔥 STRESS TEST SUITE
# =========================================================
def stress_test_suite(E_grid, T_list, sigma_min_list, s_spectra_all):
    print("\n================ STRESS TEST ================\n")

    T = np.array(T_list, dtype=float)
    sigma = np.array(sigma_min_list, dtype=float)
    sigma_sq = sigma**2

    valid = (T > 0) & (sigma_sq > 0) & np.isfinite(T) & np.isfinite(sigma_sq)

    if valid.sum() < 10:
        print("Not enough valid data for stress test.")
        return

    logT = np.log10(T[valid])
    logsig = np.log10(sigma_sq[valid])

    # -------------------------
    # 1. Core correlation
    # -------------------------
    corr = pearsonr(logsig, logT)[0]
    print(f"[1] log-log correlation: {corr:.4f}")

    # -------------------------
    # 2. Power law scan
    # -------------------------
    print("\n[2] Power scan:")
    for p in [1, 2, 3, 4]:
        corr_p = pearsonr(np.log10(sigma[valid]**p), logT)[0]
        print(f"  power {p}: corr = {corr_p:.4f}")

    # -------------------------
    # 3. Shuffle test
    # -------------------------
    shuffled = np.random.permutation(sigma_sq[valid])
    corr_shuffle = pearsonr(np.log10(shuffled), logT)[0]
    print(f"\n[3] shuffled correlation: {corr_shuffle:.4f}")

    # -------------------------
    # 4. Condition number
    # -------------------------
    conds = []
    for s in s_spectra_all:
        if len(s) >= 2 and s[0] > 0:
            conds.append(s[-1] / s[0])
        else:
            conds.append(np.nan)

    conds = np.array(conds)
    bad = np.sum(conds > 1e12)
    print(f"\n[4] ill-conditioned points (>1e12): {bad}/{len(conds)}")

    # -------------------------
    # 5. Proxy test (independence)
    # -------------------------
    proxy = []
    for s in s_spectra_all:
        if len(s) >= 2 and s[0] > 0:
            proxy.append((s[0] / s[-1])**2)

    proxy = np.array(proxy)

    if len(proxy) > 10:
        proxy_valid = np.isfinite(proxy[:len(logT)])
        corr_proxy = pearsonr(
            np.log10(proxy[:len(logT)][proxy_valid]),
            logT[:len(proxy)][proxy_valid]
        )[0]
        print(f"\n[5] proxy correlation: {corr_proxy:.4f}")

    print("\n===========================================\n")


# =========================================================
# MAIN EXPERIMENT
# =========================================================
def run_double_suite(
    V0=10.0,
    a_list=None,
    b_list=None,
    E_grid=None,
    eps_factors=None,
    results_dir=None,
):
    if a_list is None:
        a_list = [1.0, 2.0, 3.0, 5.0]
    if b_list is None:
        b_list = [0.5, 1.0, 2.0, 4.0]
    if E_grid is None:
        E_grid = np.linspace(0.1, V0 - 0.1, 200)
    if eps_factors is None:
        eps_factors = [1e-4, 1e-6, 1e-8]
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results"

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for a in a_list:
        for b in b_list:
            logger.info("Running a=%s, b=%s", a, b)

            rows = []
            T_list = []
            sigma_min_list = []
            s_spectra_all = []

            for E in E_grid:
                try:
                    A = build_A_matrix_double(E, V0, a, b)
                    stats = compute_svd_scaled(A)

                    s_sorted = np.asarray(stats.get("s_sorted", []), dtype=float)

                    sigma_min = float(s_sorted[0]) if len(s_sorted) else np.nan
                    sigma_max = float(s_sorted[-1]) if len(s_sorted) else np.nan

                    T = float(analytic_transmission_double(E, V0, a, b))

                    T_list.append(T)
                    sigma_min_list.append(sigma_min)
                    s_spectra_all.append(s_sorted.tolist())

                    rows.append({
                        "E": E,
                        "T": T,
                        "sigma_min": sigma_min,
                        "sigma_max": sigma_max,
                    })

                except Exception as exc:
                    logger.warning("Failure at E=%g: %s", E, exc)
                    T_list.append(np.nan)
                    sigma_min_list.append(np.nan)
                    s_spectra_all.append([])

            # -------------------------
            # Save CSV
            # -------------------------
            df = pd.DataFrame(rows)
            df.to_csv(results_dir / f"double_a_{a}_b_{b}.csv", index=False)

            # -------------------------
            # Run stress test 🔥
            # -------------------------
            stress_test_suite(E_grid, T_list, sigma_min_list, s_spectra_all)

            # -------------------------
            # Basic summary
            # -------------------------
            T_arr = np.array(T_list)
            sigma_sq = np.array(sigma_min_list)**2

            valid = (T_arr > 0) & (sigma_sq > 0) & np.isfinite(T_arr)

            if valid.sum() > 5:
                slope, intercept = np.polyfit(
                    np.log10(sigma_sq[valid]),
                    np.log10(T_arr[valid]),
                    1
                )
            else:
                slope = np.nan

            summary[f"a={a}_b={b}"] = {
                "slope": float(slope) if not np.isnan(slope) else None
            }

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done.")
    return summary


# =========================================================
if __name__ == "__main__":
    run_double_suite()