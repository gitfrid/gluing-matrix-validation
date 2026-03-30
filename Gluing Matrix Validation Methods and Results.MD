# Gluing Matrix Validation — Methods and Results

## Abstract

We present a new, intuitive view on 1D quantum tunneling: **transmission measures how well local solutions globally fit together**. Numerically, this coherence is captured by the smallest singular value of the gluing matrix.  

Central hypothesis:

T(E) ∝ σ_min²(E)

We introduce an analytic preconditioning strategy to stabilize the gluing matrix against exponential ill-conditioning caused by evanescent modes, a robust SVD workflow, and a gap-detection protocol for identifying multiplicity at resonances. This provides a reproducible framework for analyzing tunneling and resonances.

---

## Introduction — A New View on Tunneling

Classical tunneling approaches treat transmission as a probability amplitude across barriers. Here we reinterpret it:

**Transmission = "How well local solutions globally fit together."**

- Local solutions exist in each segment of a piecewise-constant potential.  
- Continuity of the wavefunction and its derivative ensures these local solutions glue into a global scattering state.  
- If the local solutions do not fit well, transmission is suppressed.

Numerically:

- Build a **gluing matrix** encoding matching conditions between regions.  
- Compute the **singular value decomposition (SVD)**.  
- The **smallest singular value** measures the degree of coherence — the better the fit, the larger the transmission.

---

## Results — Key Findings

- **Correlation between T(E) and σ_min²(E)** is restored when using analytic preconditioning, even for large barriers.  
- **Resonance multiplicity**: multiple small singular values appear at resonant energies, corresponding to symmetric and antisymmetric modes.  
- **Robustness**: analytic patch solutions eliminate discretization errors; only the SVD tolerance affects counting near-zero singulars.

In plain terms: **the SVD lets you "see" how local solutions combine into global behavior**, revealing both tunneling and mode multiplicity.

---

## Mathematical Framework

### Gluing Matrix Construction

For each segment:

- Propagating modes: exp(± i k x)  
- Evanescent modes: exp(± κ x) with κ = sqrt(V0 - E)

Enforcing continuity of ψ and ψ' at boundaries produces:

A(E) x = 0

Where:

- A(E) — the gluing matrix  
- x — coefficients of local basis functions  

Physical solutions correspond to **near-kernel vectors** of A(E).

### Singular Values as Coherence Measure

- Let σ₁ ≤ σ₂ ≤ … ≤ σ_n be the singular values.  
- σ_min = σ₁ measures distance to singularity.  
- Small σ_min → weak global coherence → low transmission.  
- In tunneling regime (E < V0):

T(E) ≈ C(E) * σ_min²(E)

Where C(E) is a slowly varying factor absorbing normalization choices.

### Multiplicity and Resonances

- Resonances show multiple small singular values:

σ₁ ≈ σ₂ ≈ … ≈ σ_C << σ_C+1

- Multiplicity C is detected using a **gap criterion**:

g_i = σ_{i+1} / (σ_i + ε_tiny)

- If g_i >> 1 and σ_i < δ, set C = i.  
- This counts coexisting modes at a resonance.

---

## Numerical Methods

### Problem: Ill-Conditioning

Evanescent terms (exp(± κ x)) can make matrix entries vary over many orders of magnitude, causing unstable SVD results.

### Analytic Preconditioning

- Scale rows or columns to remove exponential growth:

A → D * A, where D ≈ diag(exp(-κ x_i))

- Preserves kernel structure and singular value ratios.  
- Restores numerical stability.

### Stabilized Double-Barrier Matrix Example (Python)

import numpy as np

def build_A_matrix_double_stabilized(E, V0, a, b):
    k = np.sqrt(E + 0j)
    kappa = np.sqrt(V0 - E + 0j)
    exp = np.exp

    A = np.zeros((8, 9), dtype=complex)

    # x = 0
    A[0,0], A[0,1], A[0,2], A[0,3] = 1, 1, -1, -1
    A[1,0], A[1,1], A[1,2], A[1,3] = 1j*k, -1j*k, -kappa, kappa

    # x = a (scaled)
    scale_a = exp(-kappa * a)
    A[2,2], A[2,3] = 1.0, exp(-2*kappa*a)
    A[2,4], A[2,5] = -exp(1j*k*a)*scale_a, -exp(-1j*k*a)*scale_a
    A[3,2], A[3,3] = kappa, -kappa*exp(-2*kappa*a)
    A[3,4], A[3,5] = -1j*k*exp(1j*k*a)*scale_a, 1j*k*exp(-1j*k*a)*scale_a

    # x = a + b
    A[4,4], A[4,5] = exp(1j*k*(a+b)), exp(-1j*k*(a+b))
    A[4,6], A[4,7] = -exp(kappa*(a+b)), -exp(-kappa*(a+b))
    A[5,4], A[5,5] = 1j*k*exp(1j*k*(a+b)), -1j*k*exp(-1j*k*(a+b))
    A[5,6], A[5,7] = -kappa*exp(kappa*(a+b)), kappa*exp(-kappa*(a+b))

    # x = 2a + b (scaled)
    scale_2ab = exp(-kappa*(2*a+b))
    A[6,6], A[6,7] = 1.0, exp(-2*kappa*(2*a+b))
    A[6,8] = -exp(1j*k*(2*a+b))*scale_2ab
    A[7,6], A[7,7] = kappa, -kappa*exp(-2*kappa*(2*a+b))
    A[7,8] = -1j*k*exp(1j*k*(2*a+b))*scale_2ab

    return A

### SVD Workflow

1. Compute A = U Σ V*  
2. Normalize σ_i / σ_max  
3. Detect multiplicity with gap rule  
4. Perform log-log regression: log(T(E)) vs 2*log(σ_min(E))  
5. Restrict to tunneling regime (κ*a > 2)  
6. Report R², Pearson, Spearman

---

## Conclusion

- **Transmission = global coherence of local solutions**  
- SVD provides a **quantitative measure of this coherence**  
- Analytic preconditioning is **essential for stability**  
- Reveals multiplicity and mode structure at resonances  
- Results are **reproducible and numerically robust**

---

## Key Takeaways

- Intuitive interpretation: transmission is not just probability, it's a **measure of global fit**  
- Numerically: **SVD gives you the “coherence meter”**  
- Conceptual link: Peirce’s continuum → gluing of local solutions → emergent global behavior
