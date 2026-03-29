# notebooks/02_double_barrier_resonances.py
"""
Minimal script to run double barrier resonance suite (stabilized).
Saves CSV and a multiplicity plot for the first resonance (if any).
Robust imports and path handling so it runs from the notebooks folder or as a module.
"""
from __future__ import annotations
from pathlib import Path
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Ensure project root is on sys.path so "src" imports resolve when running from notebooks/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try package imports; if they fail, raise a clear error
try:
    from src.matrix_factory import build_A_matrix_double
    from src.stability import compute_svd_scaled
    from src.solvers import analytic_transmission_double
    from src.analytics import gap_protocol
except Exception as exc:
    raise ImportError(
        "Failed to import project modules. Make sure you ran `pip install -e .` "
        "or set PYTHONPATH to the project root. Original error: " + str(exc)
    ) from exc

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_double_barrier(out_dir: Path | None = None) -> Path:
    """Sweep energy, compute σ_min(E) and analytic T(E), save CSV and multiplicity plot."""
    if out_dir is None:
        out_dir = PROJECT_ROOT / "results"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    V0 = 10.0
    a = 1.0
    b = 4.0
    E_grid = np.linspace(0.1, V0 - 0.1, 400)

    T_list: list[float] = []
    s_min_list: list[float] = []
    s_spectra: list[np.ndarray] = []

    for E in E_grid:
        try:
            A = build_A_matrix_double(float(E), float(V0), float(a), float(b))
            stats = compute_svd_scaled(A, use_longdouble=False)
            s_sorted = np.asarray(stats.get("s_sorted", []), dtype=float)
            s_min = float(stats.get("s_min", float("nan")))
            T = float(analytic_transmission_double(float(E), float(V0), float(a), float(b)))
            T_list.append(T)
            s_min_list.append(s_min)
            s_spectra.append(s_sorted)
        except Exception as exc:
            logger.warning("Failed at E=%g: %s", float(E), exc)
            T_list.append(float("nan"))
            s_min_list.append(float("nan"))
            s_spectra.append(np.array([], dtype=float))

    df = pd.DataFrame({"E": E_grid, "T": T_list, "sigma_min": s_min_list})
    csv_path = out_dir / "double_a1_b4.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV to: %s", csv_path)

    # find peaks and plot multiplicity zoom for first peak
    try:
        peaks, _ = find_peaks(np.array(T_list), height=0.5)
        if peaks.size > 0:
            p = int(peaks[0])
            Eres = float(E_grid[p])
            window_mask = (E_grid > Eres - 0.5) & (E_grid < Eres + 0.5)
            if np.any(window_mask):
                # collect up to first 3 smallest singulars in the window (pad if necessary)
                spectra_arr = np.array([s[:3] if s.size >= 3 else np.pad(s, (0, 3 - s.size), constant_values=np.nan)
                                        for s in s_spectra])
                s_small = spectra_arr[window_mask]
                plt.figure(figsize=(6, 4))
                for col in range(s_small.shape[1]):
                    plt.plot(E_grid[window_mask], s_small[:, col], label=f"s[{col}]")
                plt.yscale("log")
                plt.xlabel("E")
                plt.ylabel("small singulars (log)")
                plt.title("Multiplicity near first resonance")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plot_path = out_dir / "multiplicity_a1_b4.png"
                plt.savefig(plot_path, dpi=150)
                plt.close()
                logger.info("Saved multiplicity plot to: %s", plot_path)
            else:
                logger.info("No energy window found around first peak.")
        else:
            logger.info("No peaks found in transmission above threshold.")
    except Exception as exc:
        logger.warning("Failed to generate multiplicity plot: %s", exc)

    return csv_path


def main() -> None:
    csv_path = run_double_barrier()
    logger.info("Done: %s", csv_path)


if __name__ == "__main__":
    main()
