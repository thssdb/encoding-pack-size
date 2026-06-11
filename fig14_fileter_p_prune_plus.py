import os
import glob
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig11_improve_compare_ratio import (
    IMPROVEMENT_EXCLUDE_ADD_DATASET_ABBREVS,
    compute_ratio_improvements,
    read_ratio_file,
)
from integer_datasets import ICDE_ROOT, INTEGER_DATASET_MAPPING, RESULTS_DIR, sort_items_by_dataset_label
from proposition_prune_count import collect_proposition_prune_pcts

dataset_mapping = INTEGER_DATASET_MAPPING
FIG_PANEL_W_MIN = 7.0
FIG_PANEL_W_PER_COL = 0.34
FIG_HEIGHT = 6.5
BAR_WIDTH = 0.35  # match improvement_bp_sprintz.png
PRUNE_YLIM_MAX = 105
PRUNE_BAR_LABEL_Y_OFFSET = -1  # data units; slightly below prior v + top*0.01


def improvement_plot_abbrevs(ratio_file: str | None = None) -> list[str]:
    """Dataset abbreviations in the same set/order as improvement_bp_sprintz.png."""
    primary = ratio_file or str(ICDE_ROOT / 'camel_ratio.xlsx')
    df = read_ratio_file(primary)
    if df is None and primary != 'camel_ratio.xlsx':
        df = read_ratio_file('camel_ratio.xlsx')
    if df is None:
        from integer_datasets import sort_dataset_abbrevs

        all_abbr = set(dataset_mapping.values()) - IMPROVEMENT_EXCLUDE_ADD_DATASET_ABBREVS
        return sort_dataset_abbrevs(all_abbr)
    cols, _, _ = compute_ratio_improvements(df)
    excl = IMPROVEMENT_EXCLUDE_ADD_DATASET_ABBREVS
    return [c for c in cols if c not in excl]


def filter_and_order_like_improvement(results: dict[str, float], abbrevs: list[str]) -> dict[str, float]:
    by_abbrev: dict[str, tuple[str, float]] = {}
    for key, value in results.items():
        ab = _dataset_display_label(key)
        by_abbrev[ab] = (key, value)
    return {by_abbrev[ab][0]: by_abbrev[ab][1] for ab in abbrevs if ab in by_abbrev}


def _dataset_display_label(name):
    if name in dataset_mapping:
        return dataset_mapping[name]
    alt = f'{name}.csv'
    if alt in dataset_mapping:
        return dataset_mapping[alt]
    return name


def collect_filter_counts(input_dir, algorithm_filter=None):
    files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    results = {}
    for fp in files:
        try:
            df = pd.read_csv(fp, encoding='utf-8', dtype=str)
        except Exception:
            try:
                df = pd.read_csv(fp, encoding='latin1', dtype=str)
            except Exception:
                continue
        if df.empty:
            continue
        cols = [c.strip() for c in df.columns]
        df.columns = cols
        if 'Filter Count' not in df.columns:
            continue
        if algorithm_filter and 'Encoding Algorithm' in df.columns:
            df = df[df['Encoding Algorithm'] == algorithm_filter]
        try:
            fc = pd.to_numeric(df['Filter Count'], errors='coerce')
            fc = fc.dropna()
            if fc.empty:
                continue
            mean_val = float(fc.mean())
            pct = mean_val / 1024.0 * 100.0
            value = pct
        except Exception:
            vals = []
            for v in df['Filter Count'].astype(str).tolist():
                s = v.strip()
                if not s:
                    continue
                try:
                    vals.append(float(s))
                except Exception:
                    continue
            if not vals:
                continue
            mean_val = sum(vals) / len(vals)
            value = mean_val / 1024.0 * 100.0
        name = os.path.splitext(os.path.basename(fp))[0]
        results[name] = value
    return results

def plot_bar(results, outpath, title='Count of pruned pack sizes in each dataset'):
    if not results:
        print('No data found to plot.')
        return
    items = sort_items_by_dataset_label(results.items(), _dataset_display_label)
    names, values = zip(*items)
    labels = [_dataset_display_label(n) for n in names]
    fontsize = 20
    title_fs = fontsize
    label_fs = fontsize
    tick_fs = fontsize
    annot_fs = fontsize
    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(values)), values, color='C0')
    plt.xticks(range(len(names)), labels, rotation=45, ha='right', fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.xlabel('Dataset', fontsize=label_fs)
    plt.ylabel('Percentage (% of 1024)', fontsize=label_fs, y=0.4)
    plt.title(title, fontsize=title_fs)
    top = max(values) if values else 0
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + top * 0.01, f'{v:.1f}%', ha='center', va='bottom', rotation=30, fontsize=annot_fs)
    if top <= 100:
        plt.ylim(0, max(100, top * 1.05))
    plt.tight_layout()
    outdir = os.path.dirname(outpath) or '.'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(outpath, dpi=400, bbox_inches='tight')
    try:
        eps_path = os.path.splitext(outpath)[0] + '.eps'
        plt.savefig(eps_path, format='eps', dpi=400, bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    print(f'Saved plot: {outpath}')

def sprintz_collect_filter_counts(input_dir, algorithm_filter=None):
    files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    results = {}
    for fp in files:
        try:
            df = pd.read_csv(fp, encoding='utf-8', dtype=str)
        except Exception:
            try:
                df = pd.read_csv(fp, encoding='latin1', dtype=str)
            except Exception:
                continue
        if df.empty:
            continue
        cols = [c.strip() for c in df.columns]
        df.columns = cols
        if 'Filter Count' not in df.columns:
            continue
        if algorithm_filter and 'Encoding Algorithm' in df.columns:
            df = df[df['Encoding Algorithm'] == algorithm_filter]
        try:
            fc = pd.to_numeric(df['Filter Count'], errors='coerce')
            fc = fc.dropna()
            if fc.empty:
                continue
            mean_val = float(fc.mean())
            pct = mean_val / 1024.0 * 100.0
            value = pct
        except Exception:
            vals = []
            for v in df['Filter Count'].astype(str).tolist():
                s = v.strip()
                if not s:
                    continue
                try:
                    vals.append(float(s))
                except Exception:
                    continue
            if not vals:
                continue
            mean_val = sum(vals) / len(vals)
            value = mean_val / 1024.0 * 100.0
        name = os.path.splitext(os.path.basename(fp))[0]
        results[name] = value
    return results

def sprintz_plot_bar(results, outpath, title='Pruning rate of pack sizes on datasets after Sprintz'):
    if not results:
        print('No data found to plot.')
        return
    items = sort_items_by_dataset_label(results.items(), _dataset_display_label)
    names, values = zip(*items)
    labels = [_dataset_display_label(n) for n in names]
    fontsize = 20
    title_fs = fontsize
    label_fs = fontsize
    tick_fs = fontsize
    annot_fs = fontsize
    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(values)), values, color='C0')
    plt.xticks(range(len(names)), labels, rotation=45, ha='right', fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.ylabel('Percentage (% of 1024)', fontsize=label_fs, y=0.4)
    plt.title(title, fontsize=title_fs)
    top = max(values) if values else 0
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + top * 0.01, f'{v:.1f}%', ha='center', va='bottom', rotation=30, fontsize=annot_fs)
    if top <= 100:
        plt.ylim(0, max(100, top * 1.05))
    plt.tight_layout()
    outdir = os.path.dirname(outpath) or '.'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(outpath, dpi=400, bbox_inches='tight')
    try:
        eps_path = os.path.splitext(outpath)[0] + '.eps'
        plt.savefig(eps_path, format='eps', dpi=400, bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    print(f'Saved plot: {outpath}')


def plot_two_bars(
    results1,
    results2,
    outpath,
    title1='(a) Pruning rate of pack sizes on datasets (BP)',
    title2='(b) Pruning rate of pack sizes on datasets (Sprintz)',
    *,
    dataset_abbrevs: list[str] | None = None,
    fig_height: float = FIG_HEIGHT,
):
    if not results1 and (not results2):
        print('No data found to plot.')
        return
    abbrevs = dataset_abbrevs or improvement_plot_abbrevs()
    results1 = filter_and_order_like_improvement(
        {k: v for k, v in results1.items() if k in dataset_mapping or f'{k}.csv' in dataset_mapping},
        abbrevs,
    )
    results2 = filter_and_order_like_improvement(
        {k: v for k, v in results2.items() if k in dataset_mapping or f'{k}.csv' in dataset_mapping},
        abbrevs,
    )
    if not results1 and not results2:
        print('No data left after aligning with improvement_bp_sprintz datasets.')
        return
    fontsize = 20
    title_fs = fontsize
    label_fs = fontsize
    tick_fs = fontsize
    annot_fs = fontsize
    n_cols = max(len(results1), len(results2), 1)
    panel_w = max(FIG_PANEL_W_MIN, n_cols * FIG_PANEL_W_PER_COL)
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, fig_height))

    if results1:
        items1 = (
            list(results1.items())
            if dataset_abbrevs is not None
            else sort_items_by_dataset_label(results1.items(), _dataset_display_label)
        )
        names1, values1 = zip(*items1)
        labels1 = [_dataset_display_label(n) for n in names1]
    else:
        names1, values1, labels1 = ([], [], [])
    x1 = np.arange(len(values1))
    bars = axes[0].bar(x1, values1, BAR_WIDTH, color='C0')
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(labels1, rotation=90, ha='center', va='top', fontsize=tick_fs)
    axes[0].tick_params(axis='y', labelsize=tick_fs)
    axes[0].set_xlabel('Dataset', fontsize=label_fs)
    axes[0].set_ylabel('Percentage (% of 1024)', fontsize=label_fs)
    axes[0].set_title(title1, fontsize=title_fs)
    for b, v in zip(bars, values1):
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            v + PRUNE_BAR_LABEL_Y_OFFSET,
            f'{v:.1f}',
            ha='center',
            va='bottom',
            rotation=0,
            fontsize=annot_fs,
        )
    axes[0].set_ylim(0, PRUNE_YLIM_MAX)
    if results2:
        items2 = (
            list(results2.items())
            if dataset_abbrevs is not None
            else sort_items_by_dataset_label(results2.items(), _dataset_display_label)
        )
        names2, values2 = zip(*items2)
        labels2 = [_dataset_display_label(n) for n in names2]
    else:
        names2, values2, labels2 = ([], [], [])
    x2 = np.arange(len(values2))
    bars2 = axes[1].bar(x2, values2, BAR_WIDTH, color='C1')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels2, rotation=90, ha='center', va='top', fontsize=tick_fs)
    axes[1].tick_params(axis='y', labelsize=tick_fs)
    axes[1].set_xlabel('Dataset', fontsize=label_fs)
    axes[1].set_ylabel('Percentage (% of 1024)', fontsize=label_fs)
    axes[1].set_title(title2, fontsize=title_fs, x=0.45)
    for b, v in zip(bars2, values2):
        axes[1].text(
            b.get_x() + b.get_width() / 2,
            v + PRUNE_BAR_LABEL_Y_OFFSET,
            f'{v:.1f}',
            ha='center',
            va='bottom',
            rotation=0,
            fontsize=annot_fs,
        )
    axes[1].set_ylim(0, PRUNE_YLIM_MAX)
    plt.tight_layout()
    outdir = os.path.dirname(outpath) or '.'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(outpath, dpi=400, bbox_inches='tight')
    try:
        eps_path = os.path.splitext(outpath)[0] + '.eps'
        plt.savefig(eps_path, format='eps', dpi=400, bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    print(f'Saved combined plot: {outpath}')

def main():
    parser = argparse.ArgumentParser(description='Plot Filter Count per dataset for BP and Sprintz on a combined figure')
    parser.add_argument('--testdata', default=str(ICDE_ROOT / 'TestData'), help='TestData directory for proposition prune counts')
    parser.add_argument('--from-results', action='store_true', help='Use Java Filter Count CSVs instead of Python proposition counts')
    parser.add_argument('--input1', default=str(RESULTS_DIR / 'output_BP_filters_plus'), help='Directory containing BP CSV files')
    parser.add_argument('--input2', default=str(RESULTS_DIR / 'output_Sprintz_filters_plus'), help='Directory containing Sprintz CSV files')
    parser.add_argument('--output', '-o', default='figure_for_paper/prune_plus_filters_count_bar_combined.png', help='Output image path')
    parser.add_argument('--algorithm1', default='BP+Prune+Plus+RMQ', help='Encoding Algorithm to filter by for BP')
    parser.add_argument('--algorithm2', default='Sprintz+Prune+Plus+RMQ', help='Encoding Algorithm to filter by for Sprintz')
    parser.add_argument('--ratio-file', default=str(ICDE_ROOT / 'camel_ratio.xlsx'), help='camel_ratio.xlsx for dataset list (same as fig11)')
    parser.add_argument('--fig-height', type=float, default=FIG_HEIGHT, help='Figure height in inches')
    args = parser.parse_args()
    plot_abbrevs = improvement_plot_abbrevs(args.ratio_file)
    allowed = set(INTEGER_DATASET_MAPPING.keys())
    if args.from_results:
        results_bp = collect_filter_counts(args.input1, algorithm_filter=args.algorithm1)
        if not results_bp:
            results_bp = collect_filter_counts(args.input1, algorithm_filter=None)
        results_sp = sprintz_collect_filter_counts(args.input2, algorithm_filter=args.algorithm2)
        if not results_sp:
            results_sp = sprintz_collect_filter_counts(args.input2, algorithm_filter=None)
    else:
        testdata = Path(args.testdata)
        results_bp = collect_proposition_prune_pcts(testdata, sprintz_encode=False)
        results_sp = collect_proposition_prune_pcts(testdata, sprintz_encode=True)
        results_bp = {k: v for k, v in results_bp.items() if f'{k}.csv' in allowed or k in allowed}
        results_sp = {k: v for k, v in results_sp.items() if f'{k}.csv' in allowed or k in allowed}
        print(f'Proposition prune counts from {testdata}')
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    print(f'Plotting {len(plot_abbrevs)} datasets (same as improvement_bp_sprintz): {", ".join(plot_abbrevs)}')
    plot_two_bars(
        results_bp,
        results_sp,
        args.output,
        title1='(a) Pruning rate after BP',
        title2='(b) Pruning rate after Sprintz',
        dataset_abbrevs=plot_abbrevs,
        fig_height=args.fig_height,
    )
if __name__ == '__main__':
    main()
