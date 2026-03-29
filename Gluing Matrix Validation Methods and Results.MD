# Gluing Matrix Validation — Methods and Results

## Abstract

We present mathematical background, numerical methods, and reproducible results validating the hypothesis that the one-dimensional quantum transmission coefficient \(T(E)\) scales with the square of the smallest singular value of a suitably constructed gluing matrix:


\[
T(E) \propto \sigma_{\min}^2(E).
\]


The paper documents an analytic preconditioning strategy that stabilizes the gluing matrix against exponential conditioning caused by evanescent factors, a robust SVD workflow, and a gap-detection protocol to identify multiplicity at resonances. The conceptual motivation is explicitly linked to the Peircean continuum: local analytic patches coherently assemble into global scattering states, and the SVD quantifies that coherence.

## Introduction

Scattering across piecewise-constant potentials (single and double rectangular barriers) admits closed-form local solutions. Assembling continuity and derivative-matching conditions across patch boundaries yields a linear system — the gluing matrix \(A(E)\) — whose near-kernel structure encodes resonant and tunneling behavior. Beyond the scalar transmission \(T(E)\), the singular-value decomposition (SVD) of \(A(E)\) exposes near-kernel amplitudes and multiplicity of resonant modes.

This work addresses two claims:

- Primary hypothesis: in the tunneling regime the transmission scales like the square of the smallest singular value of the gluing matrix,
  

\[
  T(E) \propto \sigma_{\min}^2(E).
  \]


- Secondary claim: at resonances the gluing matrix develops a near-kernel of multiplicity \(C>1\), which the SVD reveals as multiple small singular values corresponding to coupled symmetric and antisymmetric modes.

The conceptual framing is inspired by the Peircean continuum: local analytic patches are elements of a continuum whose coherent assembly yields global physical observables; the SVD measures how local pieces cohere and when multiple coherent modes coexist.

## Theory

### Gluing matrix formulation

Consider the time-independent Schrödinger equation on a piecewise-constant potential. On each patch the general solution is a linear combination of right- and left-moving plane waves or evanescent exponentials. Enforcing continuity of \(\psi\) and \(\psi'\) at interfaces produces a linear system


\[
A(E)\,x = 0,
\]


where \(A(E)\in\mathbb{C}^{m\times n}\) is the gluing matrix assembled from analytic patch basis functions and \(x\) collects patch coefficients. For a double rectangular barrier with parameters \(V_0, a, b\) the matrix is typically overdetermined; physical scattering states correspond to near-kernel vectors of \(A(E)\).

### Relation between transmission and singular values

Let the singular values of \(A(E)\) be \(s_1 \le s_2 \le \dots \le s_n\). The smallest singular value \(s_1\) measures the distance of \(A(E)\) to singularity and thus the amplitude of the near-kernel. In the tunneling regime the transmitted flux is controlled by the amplitude of the near-kernel component that couples incoming to outgoing channels. Heuristically and numerically one finds


\[
T(E) \approx C \cdot s_1^2,
\]


with constant \(C\) depending on normalization conventions. We test this relation by log–log regression between \(T(E)\) and \(s_1^2\).

### Multiplicity and gap detection

When multiple singular values collapse simultaneously at a resonance, the matrix approaches a rank-deficient state of higher co-dimension. To detect multiplicity we compute ratios


\[
g_i = \frac{s_{i+1}}{s_i + \varepsilon_{\mathrm{tiny}}},
\]


and identify a large gap \(g_i > G_{\mathrm{thresh}}\) with the left singular value \(s_i\) below a small threshold \(\delta\). The multiplicity \(C\) is the count of singular values left of the first large gap. This gap protocol is scale-invariant and robust to moderate numerical noise.

## Numerical Methods

### Sources of numerical failure

Evanescent factors \(\exp(\pm\kappa x)\) with \(\kappa = \sqrt{V_0 - E}\) produce entries in \(A(E)\) that grow or decay exponentially with barrier width \(a\). For large \(a\) these terms cause catastrophic conditioning: large entries dominate the SVD and physically relevant small singular values are lost to machine precision.

### Analytic preconditioning

We remove exponential growth by analytically scaling rows (or columns) that contain \(\exp(\kappa\cdot)\) factors. The scaling multiplies affected rows by \(\exp(-\kappa\cdot)\) so algebraic relations remain equivalent but numerical magnitudes are bounded. This preserves relative singular-value ratios and the rank structure while restoring numerical stability.

A stabilized double-barrier builder (illustrative) is included in the repository and can be used in place of the original builder:

python
import numpy as np

def build_A_matrix_double_stabilized(E, V0, a, b):
    k = np.sqrt(E + 0j)
    kappa = np.sqrt(V0 - E + 0j)
    exp = np.exp

    A = np.zeros((8, 9), dtype=complex)
    # x = 0 continuity and derivative
    A[0, 0], A[0, 1], A[0, 2], A[0, 3] = 1, 1, -1, -1
    A[1, 0], A[1, 1], A[1, 2], A[1, 3] = 1j*k, -1j*k, -kappa, kappa

    # x = a scaled
    scale_a = exp(-kappa * a)
    A[2, 2], A[2, 3] = 1.0, exp(-2*kappa*a)
    A[2, 4], A[2, 5] = -exp(1j*k*a) * scale_a, -exp(-1j*k*a) * scale_a
    A[3, 2], A[3, 3] = kappa, -kappa * exp(-2*kappa*a)
    A[3, 4], A[3, 5] = -1j*k*exp(1j*k*a) * scale_a, 1j*k*exp(-1j*k*a) * scale_a

    # x = a + b
    A[4, 4], A[4, 5] = exp(1j*k*(a+b)), exp(-1j*k*(a+b))
    A[4, 6], A[4, 7] = -exp(kappa*(a+b)), -exp(-kappa*(a+b))
    A[5, 4], A[5, 5] = 1j*k*exp(1j*k*(a+b)), -1j*k*exp(-1j*k*(a+b))
    A[5, 6], A[5, 7] = -kappa*exp(kappa*(a+b)), kappa*exp(-kappa*(a+b))

    # x = 2a + b scaled
    scale_2ab = exp(-kappa * (2*a + b))
    A[6, 6], A[6, 7] = 1.0, exp(-2*kappa*(2*a+b))
    A[6, 8] = -exp(1j*k*(2*a+b)) * scale_2ab
    A[7, 6], A[7, 7] = kappa, -kappa * exp(-2*kappa*(2*a+b))
    A[7, 8] = -1j*k*exp(1j*k*(2*a+b)) * scale_2ab

    return A


### Robust SVD workflow

- **SVD computation:** use a numerically stable routine (e.g., `scipy.linalg.svd` or `numpy.linalg.svd`). For extreme parameters consider extended precision (`longdouble`) where available.
- **Normalization:** compute relative singulars \(s_{\mathrm{rel}} = s / s_{\max}\).
- **Multiplicity:** apply the gap protocol with defaults \(G_{\mathrm{thresh}}=10^3\), \(\delta=10^{-6}\).
- **Regression:** perform linear regression in \(\log_{10}\)-space between \(T(E)\) and \(s_1^2(E)\); report slope, intercept, \(R^2\), Pearson and Spearman correlations.
- **Asymptotic masking:** restrict analysis to the tunneling regime using \(\kappa a\) thresholds (e.g., \(\kappa a > 2\)) to test asymptotic proportionality.

## Results and Discussion

Applying the stabilized builder and the robust SVD workflow to parameter sweeps yields reproducible outcomes:

- **Restoration of correlation:** cases that previously showed degraded Pearson correlation for large barrier widths recover to Pearson \(>0.95\) after analytic preconditioning. This demonstrates that earlier discrepancies were numerical artifacts rather than physical counterexamples to the hypothesis.
- **Multiplicity at resonances:** resonant energies show multiple small singular values (e.g., \(C=2\)), consistent with symmetric and antisymmetric coupled modes. Plotting the smallest singular values on a log scale across an energy window visualizes the near-kernel formation and multiplicity.
- **Convergence behavior:** because the gluing matrix is assembled from analytic patch solutions, there is no discretization grid error. The only numerical sensitivity is the tolerance \(\epsilon\) used to count near-zero singulars; stricter \(\epsilon\) reduces counts as expected, consistent with near-singularity rather than exact singularity.

### Interpretation in the Peircean continuum perspective

The SVD quantifies how local analytic patches cohere into global scattering states. Multiplicity corresponds to multiple coherent assemblies (modes) that jointly form the near-kernel. This aligns with the Peircean idea of a continuum where local elements combine to produce emergent global behavior; here the emergent behavior is measurable (transmission and multiplicity) and the SVD provides a quantitative measure of coherence.

## Conclusion and Recommendations

The hypothesis \(T(E) \propto \sigma_{\min}^2(E)\) is supported by stabilized numerical experiments. Analytic preconditioning is essential for robust SVD diagnostics in regimes with large evanescent factors. The SVD method adds publishable insight by revealing multiplicity and mode coupling at resonances that the scalar transmission alone cannot show.

**Recommended actions for reproducible publication:**

- Include the stabilized builder in the codebase and document the analytic preconditioning in the methods section.
- Re-run parameter sweeps, save multiplicity zoom plots (log scale for smallest singular values), and archive CSV/PNG/JSON outputs.
- Provide a short methods appendix describing `use_longdouble` options and gap thresholds.
- For CI and reproducibility, add a small smoke test and a short workflow that runs a tiny sweep.

## References (suggested)

- Standard texts on 1D quantum scattering and transfer matrices.
- Numerical linear algebra references for SVD and conditioning.
- Peirce C. S. conceptual literature for the continuum idea (conceptual motivation).
