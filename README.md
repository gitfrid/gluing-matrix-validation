# Gluing Matrix Validation

**Goal**  
Validate the hypothesis **`T(E) ∝ σ_min²(E)`** for 1D quantum scattering by analyzing the singular-value spectrum of the gluing matrix. The repo provides stabilized matrix builders, robust SVD wrappers, a gap protocol for multiplicity detection, and runnable examples.

[**Full Methods & Results:**](https://github.com/gitfrid/gluing-matrix-validation/blob/main/Gluing%20Matrix%20Validation%20Methods%20and%20Results.MD)

---

## Quickstart

1. Create and activate a virtual environment  

   **macOS / Linux**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   **Windows (PowerShell)**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies  

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

   or for development:

   ```bash
   pip install -e .
   ```

3. Run examples  

   ```bash
   cd /path/to/gluing-matrix-validation
   python notebooks/01_single_barrier.py
   python notebooks/02_double_barrier_resonances.py
   ```

   Optional batch runner:

   ```bash
   python gluing_experiment_stable.py
   ```

   Outputs are written to `results/` (CSV, PNG, JSON).

---

## Contents

- **src/** — core code  
  `matrix_factory.py`, `stability.py`, `solvers.py`, `analytics.py`  

- **notebooks/** — runnable scripts reproducing experiments  

- **gluing_experiment_stable.py** — optional batch runner for sweeps  

- **results/** — generated outputs  

- **tests/** — (optional) smoke/unit tests  

---

## Key Points

- **Numerical stabilization**  
  Analytic preconditioning (row/column scaling) removes exponential overflow/underflow (e.g. `exp(κ·a)`) so SVD reliably reveals small singular values.

- **Gap protocol**  
  Detects multiplicity (count of near-zero singular values) to identify resonant near-kernel modes.

- **Interpretation**  
  Strong correlation between `T(E)` and `σ_min²(E)` appears once numerical artifacts are removed.  
  SVD additionally reveals:
  - multiplicity  
  - mode structure  
  - resonance coupling  

---

## VS Code Tips

Add `.vscode/settings.json`:

```json
{
  "python.analysis.extraPaths": ["./"]
}
```

Ensure the workspace uses the correct interpreter.

Optional `.env`:

```bash
PYTHONPATH=.
```

Then reload VS Code and restart the language server.

---

## Recommended Next Steps

- Use the stabilized builder  
  `build_A_matrix_double_stabilized`

- Re-run parameter sweeps and:
  - save multiplicity zoom plots (log scale)
  - archive CSV / PNG / JSON outputs

- Add a short methods note describing:
  - analytic preconditioning
  - gap thresholds
  - SVD workflow

---

## Summary

This project shows that:

```
T(E) ∝ σ_min²(E)
```

holds robustly when numerical instability is removed.

The SVD of the gluing matrix provides deeper structure than transmission alone:
- detects resonances
- reveals mode multiplicity
- quantifies coherence of solutions
