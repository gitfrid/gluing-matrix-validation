# Gluing Matrix Validation

**Goal**  
Validate the hypothesis **\(T(E) \propto \sigma_{\min}^2(E)\)** for 1D quantum scattering by analyzing the singular-value spectrum of the gluing matrix. The repo provides stabilized matrix builders, robust SVD wrappers, a gap protocol for multiplicity detection, and runnable examples.

## Quickstart

1. Create and activate a virtual environment  
   macOS / Linux: `python -m venv .venv` then `source .venv/bin/activate`  
   Windows PowerShell: `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`

2. Install dependencies  
   `pip install --upgrade pip setuptools wheel`  
   `pip install -r requirements.txt`  
   or for development: `pip install -e .`

3. Run examples  
   `cd /path/to/gluing-matrix-validation`  
   `python notebooks/01_single_barrier.py`  
   `python notebooks/02_double_barrier_resonances.py`  
   optional batch runner: `python gluing_experiment_stable.py`  
   Outputs are written to `results/` (CSV, PNG, JSON).

## Contents

- **src/** — core code: `matrix_factory.py`, `stability.py`, `solvers.py`, `analytics.py`  
- **notebooks/** — runnable scripts reproducing experiments  
- **gluing_experiment_stable.py** — optional batch runner for sweeps  
- **results/** — generated outputs  
- **tests/** — (optional) smoke/unit tests

## Key points

- **Numerical stabilization:** analytic preconditioning (row/column scaling) removes exponential overflow/underflow (e.g. `exp(kappa*a)`) so SVD reliably reveals small singular values.  
- **Gap protocol:** detects multiplicity (count of near-zero singular values) to identify resonant near-kernel modes.  
- **Interpretation:** high correlation between `T(E)` and `sigma_min^2(E)` holds once numerical artifacts are removed; SVD also reveals multiplicity and mode structure at resonances.

## VS Code tips

Add `.vscode/settings.json` with `python.analysis.extraPaths` set to `["./"]` and ensure the workspace uses the intended interpreter. Optionally add a `.env` with `PYTHONPATH=.`. Reload the window and restart the language server.

## Recommended next steps

- Use the stabilized builder `build_A_matrix_double_stabilized` if not already present.  
- Re-run sweeps, save multiplicity zoom plots (log scale for smallest singular values), and archive CSV/PNG/JSON outputs.  
- Add a short methods note describing preconditioning and gap thresholds for reproducibility.

## License & Contributing

Check `LICENSE`. Contributions: fork → branch → PR with tests and documentation.
