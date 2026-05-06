import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent
_DATA = _REPO / "data"

df1 = pd.read_csv(_DATA / "features_and_best_p.csv")
df2 = pd.read_csv(_DATA / "features_and_best_p_sprintz.csv")
fontsize = 22

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.subplots_adjust(wspace=0.25)


axes[0].hist(df1['optimal_pack_size'], bins=1024,  color='C0')
print(df1['optimal_pack_size'].describe())
print(f"众数: {df1['optimal_pack_size'].mode()[0]}")
axes[0].set_title('(a) Optimal pack sizes (BP)', fontsize=fontsize, x=0.38)
axes[0].set_xlabel('Pack size', fontsize=fontsize)
axes[0].set_ylabel('Frequency', fontsize=fontsize)
axes[0].tick_params(axis='both', labelsize=fontsize)
axes[0].set_xlim(0.9, 128)
axes[0].set_ylim(0, 520)


axes[1].hist(df2['optimal_pack_size'], bins=1024, color='C1')
print(df2['optimal_pack_size'].describe())
print(f"众数: {df2['optimal_pack_size'].mode()[0]}")
axes[1].set_title('(b) Optimal pack sizes (Sprintz)', fontsize=fontsize, x=0.4)
axes[1].set_xlabel('Pack size', fontsize=fontsize)
axes[1].set_ylabel('Frequency', fontsize=fontsize)
axes[1].tick_params(axis='both', labelsize=fontsize)
axes[1].set_xlim(0.9, 128)
axes[1].set_ylim(0, 520)

plt.tight_layout()
os.makedirs('./figure_for_paper', exist_ok=True)
plt.savefig('./figure_for_paper/optimal_pack_size_distribution_combined.png', dpi=400, bbox_inches='tight')
plt.savefig('./figure_for_paper/optimal_pack_size_distribution_combined.eps', format='eps', dpi=400, bbox_inches='tight')

dataset_mapping = {

    'City-temp.csv': 'CT',
    'Wind-Speed.csv': 'WS',
    'IR-bio-temp.csv': 'IR',
    'PM10-dust.csv': 'PM10',
    'Air-pressure.csv': 'AP',
    'Dew-point-temp.csv': 'DT',
    'Stocks-UK.csv': 'SUK',
    'Stocks-USA.csv': 'SUA',
    'Stocks-DE.csv': 'SDE',
    'Bird-migration.csv': 'BM',

    'Food-price.csv': 'FP',
    'Blockchain-tr.csv': 'BTR',
    'City-lat.csv': 'CLT',
    'City-lon.csv': 'CLN',
}


fig, axes = plt.subplots(1, 2, figsize=(6, 3))


dataset_file=list(dataset_mapping.items())[3][0]
dataset_abbr=list(dataset_mapping.items())[3][1]
df1_subset = df1[df1['Dataset'] == dataset_file]
axes[0].hist(df1_subset['optimal_pack_size'], bins=128, edgecolor='black', alpha=0.5, label=dataset_abbr)
axes[0].set_title('(a) Dataset '+dataset_abbr, fontsize=fontsize)
axes[0].set_xlabel('Pack size', fontsize=fontsize)
axes[0].set_ylabel('Frequency', fontsize=fontsize)
axes[0].set_ylim(0, 40)
axes[0].tick_params(axis='both', labelsize=fontsize)
axes[0].set_xlim(0.9, 128)
# axes[0].legend(fontsize=fontsize)
dataset_file=list(dataset_mapping.items())[2][0]
dataset_abbr=list(dataset_mapping.items())[2][1]
df1_subset = df1[df1['Dataset'] == dataset_file]
axes[1].hist(df1_subset['optimal_pack_size'], bins=128, edgecolor='black', alpha=0.5, label=dataset_abbr)
axes[1].set_title('(b) Dataset '+dataset_abbr, fontsize=fontsize)
axes[1].set_xlabel('Pack size', fontsize=fontsize)
axes[1].set_ylabel('Frequency', fontsize=fontsize)
axes[1].set_ylim(0, 40)
axes[1].tick_params(axis='both', labelsize=fontsize)
axes[1].set_xlim(0.9, 128)


plt.tight_layout()
os.makedirs('./figure_for_paper', exist_ok=True)
plt.savefig('./figure_for_paper/optimal_pack_size_distribution_selected_datasets_bp.png', dpi=400, bbox_inches='tight')
plt.savefig('./figure_for_paper/optimal_pack_size_distribution_selected_datasets_bp.eps', format='eps', dpi=400, bbox_inches='tight')
