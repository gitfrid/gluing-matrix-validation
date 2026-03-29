# notebooks/01_single_barrier.py
"""
Minimal script to run single barrier validation and save results to results/.
Robust imports (works with editable install, PYTHONPATH, or direct execution from notebooks/).
"""
from __future__ import annotations
from pathlib import Path
import sys
import logging

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so "src" imports resolve when running from notebooks/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try package imports; if they fail, raise a clear error
try:
    from src.matrix_factory import build_A_matrix_single
    from src.stability import compute_svd_scaled
    from src.solvers import analytic_transmission_rect
except Exception as exc:
    raise ImportError(
        "Failed to import project modules. Make sure you ran `pip install -e .` "
        "or set PYTHONPATH to the project root. Original error: " + str(exc)
    ) from exc

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_single_barrier(out_dir: Path | None = None) -> Path:
    """Run single-barrier sweep and save CSV to results/."""
    if out_dir is None:
        out_dir = PROJECT_ROOT / "results"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    V0 = 10.0
    a = 3.0
    E_grid = np.linspace(0.1, V0 - 0.1, 200)
    rows: list[dict] = []

    for E in E_grid:
        try:
            A = build_A_matrix_single(float(E), float(V0), float(a))
            stats = compute_svd_scaled(A, use_longdouble=False)
            sigma_min = float(stats.get("s_min", float("nan")))
            T = float(analytic_transmission_rect(float(E), float(V0), float(a)))
            rows.append({"E": float(E), "T": T, "sigma_min": sigma_min})
        except Exception as exc:
            logger.warning("Failed at E=%g: %s", float(E), exc)
            rows.append({"E": float(E), "T": float("nan"), "sigma_min": float("nan")})

    df = pd.DataFrame(rows)
    out_path = out_dir / "single_barrier_a3.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved results to: %s", out_path)
    return out_path


def main() -> None:
    run_single_barrier()


if __name__ == "__main__":
    main()
