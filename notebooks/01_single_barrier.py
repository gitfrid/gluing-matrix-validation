# Minimal script to run single barrier validation
from src.matrix_factory import build_A_matrix_single
from src.stability import compute_svd_scaled
from src.solvers import analytic_transmission_rect
import numpy as np
import pandas as pd

V0 = 10.0; a = 3.0
E_grid = np.linspace(0.1, V0-0.1, 200)
rows = []
for E in E_grid:
    A = build_A_matrix_single(E, V0, a)
    stats = compute_svd_scaled(A, use_longdouble=False)
    sigma_min = stats['s_min']
    T = analytic_transmission_rect(E, V0, a)
    rows.append({'E':E, 'T':T, 'sigma_min':sigma_min})
df = pd.DataFrame(rows)
df.to_csv('results/single_barrier_a3.csv', index=False)
print('Done: results/single_barrier_a3.csv')
