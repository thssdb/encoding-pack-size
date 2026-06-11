import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from integer_datasets import FIGURE_DIR, RESULTS_DIR

_DATA = RESULTS_DIR
_OUT = FIGURE_DIR
df1 = pd.read_csv(_DATA / 'features_and_best_p.csv')
df2 = pd.read_csv(_DATA / 'features_and_best_p_sprintz.csv')
fontsize = 22
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.subplots_adjust(wspace=0.25)
n0, _, _ = axes[0].hist(df1['optimal_pack_size'], bins=1024, color='C0')
print(df1['optimal_pack_size'].describe())
print(f"Mode: {df1['optimal_pack_size'].mode()[0]}")
axes[0].set_title('(a) Optimal pack sizes (BP)', fontsize=fontsize, x=0.38)
axes[0].set_xlabel('Pack size', fontsize=fontsize)
axes[0].set_ylabel('Frequency', fontsize=fontsize)
axes[0].tick_params(axis='both', labelsize=fontsize)
axes[0].set_xlim(0.9, 128)
axes[0].set_ylim(0, max(float(n0.max()), 1.0) * 1.05)
n1, _, _ = axes[1].hist(df2['optimal_pack_size'], bins=1024, color='C1')
print(df2['optimal_pack_size'].describe())
print(f"Mode: {df2['optimal_pack_size'].mode()[0]}")
axes[1].set_title('(b) Optimal pack sizes (Sprintz)', fontsize=fontsize, x=0.4)
axes[1].set_xlabel('Pack size', fontsize=fontsize)
axes[1].set_ylabel('Frequency', fontsize=fontsize)
axes[1].tick_params(axis='both', labelsize=fontsize)
axes[1].set_xlim(0.9, 128)
axes[1].set_ylim(0, max(float(n1.max()), 1.0) * 1.05)
plt.tight_layout()
os.makedirs(_OUT, exist_ok=True)
out = _OUT / 'optimal_pack_size_distribution_combined.png'
plt.savefig(out, dpi=400, bbox_inches='tight')
plt.savefig(out.with_suffix('.eps'), format='eps', dpi=400, bbox_inches='tight')
print(f'Saved plot: {out}')
