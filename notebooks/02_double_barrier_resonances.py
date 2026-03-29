# Minimal script to run double barrier resonance suite (stabilized)
from src.matrix_factory import build_A_matrix_double
from src.stability import compute_svd_scaled
from src.solvers import analytic_transmission_double
from src.analytics import gap_protocol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
os.makedirs('results', exist_ok=True)

V0 = 10.0
a = 1.0; b = 4.0
E_grid = np.linspace(0.1, V0-0.1, 400)
T_list = []
s_min_list = []
s_spectra = []
for E in E_grid:
    A = build_A_matrix_double(E, V0, a, b)
    stats = compute_svd_scaled(A, use_longdouble=False)
    s_sorted = stats['s_sorted']
    T = analytic_transmission_double(E, V0, a, b)
    T_list.append(T); s_min_list.append(s_sorted[0]); s_spectra.append(s_sorted)
# save CSV
df = pd.DataFrame({'E':E_grid, 'T':T_list, 'sigma_min':s_min_list})
df.to_csv('results/double_a1_b4.csv', index=False)
# find peaks and plot multiplicity zoom for first peak
peaks, _ = find_peaks(np.array(T_list), height=0.5)
if peaks.size>0:
    p = peaks[0]; Eres = E_grid[p]
    window = (E_grid > Eres-0.5) & (E_grid < Eres+0.5)
    s_small = np.array(s_spectra)[window][:,:3]
    plt.figure(); plt.plot(E_grid[window], s_small); plt.yscale('log'); plt.savefig('results/multiplicity_a1_b4.png')
print('Done: results/double_a1_b4.csv and multiplicity plot')
