# Gluing Matrix Validation — Methods and Results

## Abstract

We present a new perspective on one-dimensional quantum tunneling: instead of viewing tunneling as a mysterious quantum effect, it is interpreted as a measure of **global coherence of local solutions**.  

In piecewise-constant potentials, the wavefunction is constructed from local solutions on each region. A physical scattering state exists only if these local solutions match at interfaces. We encode these compatibility conditions in a **gluing matrix** \(A(E)\).  

Using **Singular Value Decomposition (SVD)**, we find:

\[
T(E) \propto \sigma_{\min}^2(E)
\]

where \(T(E)\) is the transmission coefficient and \(\sigma_{\min}\) is the smallest singular value of \(A(E)\). The SVD also reveals **resonance multiplicity** (multiple near-zero singular values).  

We provide **analytic preconditioning** to remove exponential ill-conditioning from evanescent modes, a robust SVD workflow, and reproducible numerical experiments validating these claims.

---

## 1. Introduction — Tunneling as Coherence

### 1.1 Physical intuition

Classically, a particle either passes a barrier or not. In quantum mechanics, the transmission of a wavefunction across a potential barrier can be understood more intuitively as:

> **Transmission measures how well local solutions combine into a global solution.**

- Local solutions in each potential region are **analytic**.  
- Matching conditions at interfaces enforce **continuity** of the wavefunction and its derivative.  
- Only when the pieces “fit together” globally does a physical transmission occur.  

### 1.2 Conceptual framing

- The **gluing matrix** encodes these matching conditions.  
- **SVD** quantifies the “fit” or **coherence**.  
- Near-zero singular values correspond to physically allowed scattering states; multiple small singular values indicate **resonances**.

This perspective links naturally to the **Peircean continuum**, where local sections (patches) glue to form a coherent global object.

---

## 2. Main Result

### 2.1 Empirical relation

Through stabilized numerical experiments, we find:

\[
T(E) \propto \sigma_{\min}^2(E)
\]

- \(\sigma_{\min}\) measures the “distance to singularity” of the gluing matrix.  
- Small \(\sigma_{\min}\) → poor coherence → low transmission.  
- Large \(\sigma_{\min}\) → strong coherence → high transmission.

### 2.2 Resonances and multiplicity

At certain energies, multiple near-zero singular values appear:

\[
\sigma_1 \approx \sigma_2 \approx \dots \approx \sigma_C \ll \sigma_{C+1}
\]

- \(C\) is the multiplicity (number of coherent modes).  
- Detected numerically using a **gap protocol** in the normalized singular-value spectrum.

---

## 3. Mathematical Framework

### 3.1 Gluing matrix construction

For a piecewise-constant potential:

\[
-\frac{\hbar^2}{2m} \psi''(x) + V(x) \psi(x) = E \psi(x)
\]

- In each segment: \(\psi(x) = A e^{i k x} + B e^{-i k x}\) (propagating)  
  or \(\psi(x) = C e^{\kappa x} + D e^{-\kappa x}\) (evanescent), \(\kappa = \sqrt{V_0 - E}\)
- At interfaces, enforce continuity of \(\psi\) and \(\psi'\):

\[
A(E) x = 0
\]

where \(x\) collects local coefficients and \(A(E)\) is the gluing matrix.

### 3.2 Relation to singular values

Let the singular values of \(A(E)\) be:

\[
\sigma_1 \le \sigma_2 \le \dots \le \sigma_n
\]

- \(\sigma_{\min} = \sigma_1\) measures **compatibility/coherence**.  
- Transmission is controlled by the amplitude of the near-kernel vector:

\[
T(E) \approx C(E) \sigma_{\min}^2
\]

with slowly varying \(C(E)\) absorbing normalization factors.

### 3.3 Resonance detection

- Compute ratios: \(g_i = \frac{\sigma_{i+1}}{\sigma_i + \varepsilon_{\text{tiny}}}\)  
- Define multiplicity \(C\) as the number of singular values left of the first large gap \(g_i > G_\text{thresh}\) and below \(\delta\).

---

## 4. Numerical Methods

### 4.1 Challenges

- Evanescent terms \(\exp(\pm \kappa x)\) cause entries to span **many orders of magnitude** → SVD instability.  

### 4.2 Analytic preconditioning

- Rescale rows or columns to remove exponential growth:

\[
A \mapsto D A, \quad D \sim \mathrm{diag}(e^{-\kappa x_i})
\]

- Preserves kernel structure and singular-value ratios.  
- Restores numerical stability for large barriers.

### 4.3 Stabilized double-barrier builder (Python example)

```python
import numpy as np

def build_A_matrix_double_stabilized(E, V0, a, b):
    k = np.sqrt(E + 0j)
    kappa = np.sqrt(V0 - E + 0j)
    exp = np.exp

    A = np.zeros((8, 9), dtype=complex)

    # x = 0
    A[0,0], A[0,1], A[0,2], A[0,3] = 1,1,-1,-1
    A[1,0], A[1,1], A[1,2], A[1,3] = 1j*k,-1j*k,-kappa,kappa

    # x = a (scaled)
    scale_a = exp(-kappa*a)
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
