#!/usr/bin/env python3
# gluing_experiment_stable.py
"""
Batch runner for double-barrier resonance suite using src/ helpers.
Produces CSVs, PNGs and a JSON summary in results/.
Requires: editable install (pip install -e .) or PYTHONPATH pointing to project root.
"""
from __future__ import annotations
from pathlib import Path
import sys
# ensure project root (one level above notebooks/) is on sys.path
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


# Import project helpers (must exist in src/)
try:
    from src.matrix_factory import build_A_matrix_double
    from src.stability import compute_svd_scaled
    from src.solvers import analytic_transmission_double
    from src.analytics import gap_protocol
except Exception as exc:
    raise ImportError(
        "Failed to import project modules. Run `pip install -e .` or set PYTHONPATH to project root. "
        f"Original error: {exc}"
    ) from exc

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("gluing_experiment")

def run_double_suite(
    V0: float = 10.0,
    a_list: list[float] | None = None,
    b_list: list[float] | None = None,
    E_grid: np.ndarray | None = None,
    eps_factors: list[float] | None = None,
    results_dir: Path | None = None,
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
            C_counts = {eps: [] for eps in eps_factors}

            use_longdouble_flag = (a >= 5.0)

            for E in E_grid:
                try:
                    A = build_A_matrix_double(float(E), float(V0), float(a), float(b))
                    stats = compute_svd_scaled(A, use_longdouble=use_longdouble_flag)
                    s_sorted = np.asarray(stats.get("s_sorted", []), dtype=float)
                    s_rel = np.asarray(stats.get("s_rel", []), dtype=float)

                    sigma_min = float(s_sorted[0]) if s_sorted.size > 0 else float("nan")
                    sigma_max = float(s_sorted[-1]) if s_sorted.size > 0 else float("nan")
                    T = float(analytic_transmission_double(float(E), float(V0), float(a), float(b)))

                    T_list.append(T)
                    sigma_min_list.append(sigma_min)
                    s_spectra_all.append(s_sorted.tolist())

                    for eps in eps_factors:
                        C_counts[eps].append(int(np.sum(s_rel < eps)))

                    # physical kappa (with hbar=1, m=0.5): sqrt(2*m*(V0-E))
                    kappa = np.sqrt(np.maximum(0.0, 2.0 * 0.5 * max(0.0, V0 - E)))
                    rows.append({
                        "E": float(E),
                        "T": float(T),
                        "sigma_min": float(sigma_min),
                        "sigma_min_sq": float(sigma_min**2 if not np.isnan(sigma_min) else np.nan),
                        "sigma_max": float(sigma_max),
                        "kappa": float(kappa),
                        "kappa_a": float(kappa * a)
                    })
                except Exception as exc:
                    logger.warning("Failure at E=%g (a=%s,b=%s): %s", E, a, b, exc)
                    T_list.append(float("nan"))
                    sigma_min_list.append(float("nan"))
                    s_spectra_all.append([])
                    for eps in eps_factors:
                        C_counts[eps].append(0)
                    rows.append({
                        "E": float(E),
                        "T": float("nan"),
                        "sigma_min": float("nan"),
                        "sigma_min_sq": float("nan"),
                        "sigma_max": float("nan"),
                        "kappa": float("nan"),
                        "kappa_a": float("nan")
                    })

            # DataFrame + multiplicity columns
            df = pd.DataFrame(rows)
            for eps in eps_factors:
                df[f"C_eps_{eps}"] = C_counts[eps]

            # Peak detection
            T_arr = np.asarray(T_list, dtype=float)
            peaks, props = find_peaks(T_arr, height=0.5)
            resonance_flags = np.zeros(len(E_grid), dtype=int)
            resonance_flags[peaks] = 1
            df["resonance_flag"] = resonance_flags

            csv_name = f"double_a_{a}_b_{b}.csv"
            df.to_csv(results_dir / csv_name, index=False)
            logger.info("Saved CSV: %s", results_dir / csv_name)

            # log-log regression and correlations
            sigma_sq = np.asarray(sigma_min_list, dtype=float)**2
            valid_mask = (T_arr > 0) & (sigma_sq > 0) & np.isfinite(T_arr) & np.isfinite(sigma_sq)
            pearson_corr = float(pearsonr(np.log10(sigma_sq[valid_mask]), np.log10(T_arr[valid_mask]))[0]) if valid_mask.sum() > 1 else float("nan")
            spearman_corr = float(spearmanr(np.log10(sigma_sq[valid_mask]), np.log10(T_arr[valid_mask]))[0]) if valid_mask.sum() > 1 else float("nan")
            if valid_mask.sum() > 1:
                slope, intercept = np.polyfit(np.log10(sigma_sq[valid_mask]), np.log10(T_arr[valid_mask]), 1)
                pred = slope * np.log10(sigma_sq[valid_mask]) + intercept
                ss_res = np.sum((np.log10(T_arr[valid_mask]) - pred)**2)
                ss_tot = np.sum((np.log10(T_arr[valid_mask]) - np.mean(np.log10(T_arr[valid_mask])))**2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
            else:
                slope = intercept = r2 = float("nan")

            # asymptotic subset (kappa*a > 2)
            kappa_vals = np.sqrt(np.maximum(0.0, 2.0 * 0.5 * np.maximum(0.0, V0 - E_grid)))
            asym_mask = (kappa_vals * a) > 2.0
            asym_valid = valid_mask & asym_mask
            if asym_valid.sum() > 1:
                logsig_asym = np.log10(sigma_sq[asym_valid])
                logT_asym = np.log10(T_arr[asym_valid])
                slope_asym, intercept_asym = np.polyfit(logsig_asym, logT_asym, 1)
                ss_res_asym = np.sum((logT_asym - (slope_asym * logsig_asym + intercept_asym))**2)
                ss_tot_asym = np.sum((logT_asym - np.mean(logT_asym))**2)
                r2_asym = 1.0 - ss_res_asym / ss_tot_asym if ss_tot_asym != 0 else float("nan")
            else:
                slope_asym = intercept_asym = r2_asym = float("nan")

            # resonance info and multiplicity zooms
            resonance_info = []
            for p in peaks:
                Eres = float(E_grid[p])
                stats_res = compute_svd_scaled(build_A_matrix_double(Eres, V0, a, b), use_longdouble=(a >= 5.0))
                svals = np.sort(np.asarray(stats_res.get("s_sorted", []), dtype=float))
                s_rel_res = np.asarray(stats_res.get("s_rel", []), dtype=float)
                counts = {str(eps): int(np.sum(s_rel_res < eps)) for eps in eps_factors}
                resonance_info.append({"Eres": Eres, "counts": counts, "svals": svals.tolist()})

                # multiplicity zoom
                E_window = np.linspace(max(0.1, Eres - 0.5), min(V0 - 0.1, Eres + 0.5), 200)
                s_small = []
                for Ewin in E_window:
                    stats_win = compute_svd_scaled(build_A_matrix_double(Ewin, V0, a, b), use_longdouble=(a >= 5.0))
                    s_sorted_win = np.sort(np.asarray(stats_win.get("s_sorted", []), dtype=float))
                    s_small.append(np.pad(s_sorted_win[:3], (0, max(0, 3 - s_sorted_win[:3].size)), constant_values=np.nan))
                s_small = np.array(s_small, dtype=float)

                plt.figure(figsize=(6, 4))
                for col in range(s_small.shape[1]):
                    plt.plot(E_window, s_small[:, col], label=f"s[{col+1}]")
                plt.yscale("log")
                plt.axvline(Eres, color="r", linestyle="--", label="E_res")
                plt.xlabel("E")
                plt.ylabel("sigma (log)")
                plt.title(f"Multiplicity zoom a={a}, b={b}, Eres={Eres:.4f}")
                plt.legend()
                plt.grid(True, which="both", ls="--", alpha=0.4)
                plt.tight_layout()
                plt.savefig(results_dir / f"multiplicity_a{a}_b{b}_E{Eres:.4f}.png")
                plt.close()

            # summary plots
            plt.figure(figsize=(10, 6))
            plt.semilogy(E_grid, T_list, label="T(E)")
            plt.semilogy(E_grid, np.array(sigma_min_list)**2, label=r"$\sigma_{min}^2$")
            if peaks.size > 0:
                plt.scatter(E_grid[peaks], np.array(T_list)[peaks], c="r", label="resonances")
            plt.xlabel("E")
            plt.ylabel("Amplitude (log)")
            plt.title(f"Double barrier a={a}, b={b}")
            plt.legend()
            plt.grid(True, which="both", ls="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(results_dir / f"double_a_{a}_b_{b}.png")
            plt.close()

            # log-log scatter + fit
            if valid_mask.sum() > 1:
                plt.figure(figsize=(6, 6))
                logsig = np.log10(sigma_sq[valid_mask])
                logT = np.log10(T_arr[valid_mask])
                plt.scatter(logsig, logT, alpha=0.5)
                xfit = np.linspace(np.min(logsig), np.max(logsig), 100)
                plt.plot(xfit, slope * xfit + intercept, "r--", label=f"slope={slope:.3f}")
                plt.xlabel(r"$\log_{10}(\sigma_{min}^2)$")
                plt.ylabel(r"$\log_{10}(T)$")
                plt.title(f"log-log fit a={a}, b={b}, R2={r2:.3f}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(results_dir / f"double_loglog_a_{a}_b_{b}.png")
                plt.close()

            key = f"a={a}_b={b}"
            summary[key] = {
                "pearson": float(pearson_corr) if not np.isnan(pearson_corr) else None,
                "spearman": float(spearman_corr) if not np.isnan(spearman_corr) else None,
                "slope": float(slope) if not np.isnan(slope) else None,
                "intercept": float(intercept) if not np.isnan(intercept) else None,
                "r2": float(r2) if not np.isnan(r2) else None,
                "slope_asym": float(slope_asym) if not np.isnan(slope_asym) else None,
                "intercept_asym": float(intercept_asym) if not np.isnan(intercept_asym) else None,
                "r2_asym": float(r2_asym) if not np.isnan(r2_asym) else None,
                "num_resonances": int(len(peaks)),
                "resonance_energies": [float(E_grid[p]) for p in peaks],
                "resonance_info": resonance_info
            }

    # write summary JSON and README
    with open(results_dir / "double_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(results_dir / "README.txt", "w") as f:
        f.write("Double Barrier Resonance Suite\n")
        f.write(f"Parameters: V0={V0}, a_list={a_list}, b_list={b_list}, E_grid points={len(E_grid)}\n")
        f.write(f"eps_factors: {eps_factors}\n")
        f.write("Outputs: CSVs, PNGs, double_summary.json\n")

    logger.info("Run complete. Results in %s", results_dir)
    return summary

if __name__ == "__main__":
    run_double_suite()
