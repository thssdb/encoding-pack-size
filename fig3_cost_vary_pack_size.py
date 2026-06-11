import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from integer_datasets import FIGURE_DIR, INTEGER_SRC, RESULTS_DIR

# MW = MeteoNet-Weather.csv (MeteoNet weather integer series)
MW_DATASET_FILE = 'MeteoNet-Weather.csv'
MW_COST_STEM = 'MeteoNet-Weather_cost'
CHUNK_SIZE = 1024
CSV_DIR = RESULTS_DIR / 'packsize_cost_analysis'
OUTPUT_DIR = FIGURE_DIR
FIG_FIGSIZE = (8, 4)
FIG_LEGEND_FONTSIZE = 16
FIG4_MARKER_SIZE = 9


def _bit_width(value: int) -> int:
    v = max(1, value)
    return v.bit_length()


def _scale_integers(values: list[int]) -> list[int]:
    if not values:
        return []
    m = min(values)
    return [v - m for v in values]


def _load_first_chunk(path: Path, chunk_size: int = CHUNK_SIZE) -> list[int]:
    values: list[int] = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                values.append(int(float(s)))
            except ValueError:
                continue
            if len(values) >= chunk_size:
                break
    if len(values) < chunk_size:
        raise ValueError(f'{path} has only {len(values)} values; need {chunk_size}')
    return values


def compute_chunk_cost_csv(chunk: list[int]) -> pd.DataFrame:
    """Mirror OptimizePackSizeTest.testPackSizeCostAnalysis for one chunk."""
    n = len(chunk)
    global_max = max(chunk)
    z = int(math.ceil(math.log(_bit_width(global_max) + 1) / math.log(2)))
    rows: list[dict[str, int]] = []
    for pack_size in range(1, min(n, CHUNK_SIZE) + 1):
        num_packs = (n + pack_size - 1) // pack_size
        value_cost = 0
        for pack_idx in range(num_packs):
            start = pack_idx * pack_size
            end = min(start + pack_size, n)
            max_bw = max(_bit_width(chunk[i]) for i in range(start, end))
            value_cost += (end - start) * max_bw
        bit_width_cost = num_packs * z
        cost = value_cost + bit_width_cost
        rows.append(
            {
                'pack size': pack_size,
                'value_cost': value_cost,
                'bitwidth_cost': bit_width_cost,
                'cost': cost,
            }
        )
    return pd.DataFrame(rows)


def ensure_mw_cost_csv(chunk_size: int = CHUNK_SIZE) -> Path:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out = CSV_DIR / f'{MW_COST_STEM}.csv'
    src = INTEGER_SRC / MW_DATASET_FILE
    if not src.is_file():
        raise FileNotFoundError(f'Missing MW dataset: {src}')
    chunk = _scale_integers(_load_first_chunk(src, chunk_size))
    df = compute_chunk_cost_csv(chunk)
    df.to_csv(out, index=False)
    print(f'Wrote MW first-block cost CSV: {out} ({len(df)} rows)')
    return out


def _mw_cost_csv() -> Path:
    path = CSV_DIR / f'{MW_COST_STEM}.csv'
    if not path.is_file():
        return ensure_mw_cost_csv()
    df = pd.read_csv(path, nrows=CHUNK_SIZE)
    if len(df) < CHUNK_SIZE:
        return ensure_mw_cost_csv()
    return path


def fig_of_cost_values_bitwidth_in_chunk(output_dir: Path, chunk_size: int = CHUNK_SIZE) -> Path:
    os.makedirs(output_dir, exist_ok=True)
    path = _mw_cost_csv()
    df = pd.read_csv(path)
    if 'pack size' not in df.columns or 'cost' not in df.columns:
        raise ValueError(f'{path} missing required columns')
    sub = df.iloc[0:chunk_size]
    sub_sorted = sub.sort_values(by='pack size')
    x = sub_sorted['pack size'].values
    y = sub_sorted['cost'].values
    y1 = sub_sorted['bitwidth_cost'].values
    y2 = sub_sorted['value_cost'].values
    fontsize = FIG_LEGEND_FONTSIZE
    fig = plt.figure(figsize=FIG_FIGSIZE)
    colors = ['#FF0000', '#00FF00', '#0000FF']
    plt.plot(x, y, linestyle='-', marker='o', markersize=3, color=colors[0], label='Total storage cost')
    min_idx = int(np.nanargmin(y))
    min_x = x[min_idx]
    print(
        f'  Minimum total cost at pack size={min_x}, cost={y[min_idx]:.2f}, '
        f'bit-width cost={y1[min_idx]:.2f}, value cost={y2[min_idx]:.2f}'
    )
    plt.plot(x, y1, linestyle='--', marker='x', markersize=3, color=colors[1], label='Bit width cost')
    plt.plot(x, y2, linestyle='--', marker='s', markersize=3, color=colors[2], label='Value cost')
    plt.xlabel('Pack size $s$', fontsize=fontsize)
    plt.ylabel('Cost (bits)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=fontsize,
        labelspacing=0.1,
        handletextpad=0.1,
        columnspacing=0.1,
    )
    out_png = output_dir / f'{MW_COST_STEM}_rows_1_{chunk_size}_value_and_bit_width.png'
    out_eps = output_dir / f'{MW_COST_STEM}_rows_1_{chunk_size}_value_and_bit_width.eps'
    plt.savefig(out_eps, dpi=150, bbox_inches='tight', format='eps')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved chunk plot: {out_png.name}')
    return out_png


def create_chunk_vary_3_plots(output_dir: Path, chunk_size: int = CHUNK_SIZE) -> Path:
    os.makedirs(output_dir, exist_ok=True)
    sequences = [[3, 6, 12, 24, 48, 96, 192, 384, 768]]
    seq_names = ['3*$2^\\beta$ (3,6,...,768)']
    path = _mw_cost_csv()
    df = pd.read_csv(path)
    sub = df.iloc[0:chunk_size]
    colors = ['#FF0000', '#00FF00', '#0000FF']
    fontsize = FIG_LEGEND_FONTSIZE
    fig, ax2 = plt.subplots(figsize=FIG_FIGSIZE)
    fig.subplots_adjust(top=0.88)
    current_packsizes = set(sub['pack size'].unique())
    for seq, seq_name in zip(sequences, seq_names):
        seq_in_data = [ps for ps in seq if ps in current_packsizes]
        if not seq_in_data:
            continue
        seq_data = []
        for ps in seq_in_data:
            row = sub[sub['pack size'] == ps]
            seq_data.append(
                (
                    ps,
                    float(row['cost'].mean()),
                    float(row['bitwidth_cost'].mean()),
                    float(row['value_cost'].mean()),
                )
            )
        seq_data.sort(key=lambda item: item[0])
        seq_x = [item[0] for item in seq_data]
        seq_y = [item[1] for item in seq_data]
        seq_y1 = [item[2] for item in seq_data]
        seq_y2 = [item[3] for item in seq_data]
        seq_label = seq_name.split('(')[0].strip()
        ax2.plot(
            seq_x,
            seq_y,
            linestyle='-',
            marker='o',
            markersize=FIG4_MARKER_SIZE,
            color=colors[0],
            linewidth=2,
            label=f'Total stoarge cost of {seq_label} ',
        )
        ax2.plot(
            seq_x,
            seq_y1,
            linestyle='--',
            marker='x',
            markersize=FIG4_MARKER_SIZE,
            color=colors[1],
            linewidth=1,
            label=f'Bit width cost of {seq_label} ',
        )
        ax2.plot(
            seq_x,
            seq_y2,
            linestyle='--',
            marker='s',
            markersize=FIG4_MARKER_SIZE,
            color=colors[2],
            linewidth=1,
            label=f'Value cost of {seq_label}',
        )
    ax2.set_xlabel('Pack size s', fontsize=fontsize)
    ax2.set_ylabel('Cost (bits)', fontsize=fontsize)
    ax2.tick_params(axis='both', labelsize=fontsize)
    ax2.set_ylim(-1000, 30001)
    fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        fontsize=fontsize,
        labelspacing=0.1,
        handletextpad=0.1,
        columnspacing=0.1,
    )
    out_png = output_dir / f'{MW_COST_STEM}_rows_1_{chunk_size}_grouped_by_3.png'
    out_eps = output_dir / f'{MW_COST_STEM}_rows_1_{chunk_size}_grouped_by_3.eps'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_eps, dpi=300, bbox_inches='tight', format='eps')
    plt.close()
    print(f'  Saved chunk plot: {out_png.name}')
    return out_png


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Using MW cost CSV: {_mw_cost_csv()}')
    create_chunk_vary_3_plots(OUTPUT_DIR, chunk_size=CHUNK_SIZE)
    fig_of_cost_values_bitwidth_in_chunk(OUTPUT_DIR, chunk_size=CHUNK_SIZE)
