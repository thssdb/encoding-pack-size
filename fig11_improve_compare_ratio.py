import os
import argparse
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from integer_datasets import INTEGER_DATASET_MAPPING, RESULTS_DIR, sort_dataset_abbrevs

_PCT_1DEC = FormatStrFormatter('%.1f')

def _fmt_bar_text_1dec(v):
    return f'{float(v):.1f}'
dataset_mapping = INTEGER_DATASET_MAPPING
IMPROVEMENT_EXCLUDE_ABBREVS = frozenset()
IMPROVEMENT_EXCLUDE_ADD_DATASET_ABBREVS = frozenset({'SCUC', 'VGSUC', 'RM'})
IMPROVEMENT_HIGHLIGHT_ABBREVS = frozenset({})
IMPROVEMENT_HIGHLIGHT_ADD_DATASET_ABBREVS = frozenset()
C0_DARK = '#1565a8'
C1_DARK = '#cc5500'
PACK_SIZES_VARY_UP_TO_1024 = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

def total_time_vary_pack_sizes(csv_path, pack_sizes=None, encoding_only=False):
    pack_sizes = pack_sizes or PACK_SIZES_VARY_UP_TO_1024
    df = pd.read_csv(csv_path)
    if 'Pack size' not in df.columns:
        return np.nan
    sub = df[df['Pack size'].isin(pack_sizes)]
    if sub.empty:
        return np.nan
    enc = pd.to_numeric(sub['Encoding Time'], errors='coerce').sum()
    if encoding_only:
        return float(enc)
    dec = pd.to_numeric(sub['Decoding Time'], errors='coerce').sum()
    return float(enc + dec)

def total_time_single_run_csv(csv_path, encoding_only=False):
    df = pd.read_csv(csv_path)
    if len(df) < 1:
        return np.nan
    row = df.iloc[0]
    enc = pd.to_numeric(row['Encoding Time'], errors='coerce')
    if pd.isna(enc):
        return np.nan
    if encoding_only:
        return float(enc)
    dec = pd.to_numeric(row['Decoding Time'], errors='coerce')
    if pd.isna(dec):
        return np.nan
    return float(enc + dec)

def build_prune_vs_vary_pct_by_abbrev(vary_dir, single_dir, pack_sizes=None, encoding_only=True):
    pack_sizes = pack_sizes or PACK_SIZES_VARY_UP_TO_1024
    out = {}
    for fname, abbrev in dataset_mapping.items():
        pv = os.path.join(vary_dir, fname)
        ps = os.path.join(single_dir, fname)
        if not (os.path.isfile(pv) and os.path.isfile(ps)):
            continue
        t_vary = total_time_vary_pack_sizes(pv, pack_sizes, encoding_only=encoding_only)
        t_one = total_time_single_run_csv(ps, encoding_only=encoding_only)
        if np.isnan(t_vary) or np.isnan(t_one) or t_vary <= 0:
            continue
        out[abbrev] = (t_vary - t_one) / t_vary * 100.0
    return out

def _fig11_exclude(exclude_abbrevs=None):
    if exclude_abbrevs is None:
        return IMPROVEMENT_EXCLUDE_ABBREVS | IMPROVEMENT_EXCLUDE_ADD_DATASET_ABBREVS
    return exclude_abbrevs


def _filter_abbrevs(abbrevs, exclude_abbrevs=None):
    excl = _fig11_exclude(exclude_abbrevs)
    return sort_dataset_abbrevs([a for a in abbrevs if a not in excl])

def compute_prune_vs_vary_time_pct(bp_vary_dir='output_BP_vary_pack_size', bp_dir='output_BP', sprintz_vary_dir='output_Sprintz_vary_pack_size', sprintz_prune_dir='output_Sprintz_only_Prune_Plus_RMQ_all_no8', pack_sizes=None, exclude_abbrevs=None):
    pack_sizes = pack_sizes or PACK_SIZES_VARY_UP_TO_1024
    bp_pct = build_prune_vs_vary_pct_by_abbrev(bp_vary_dir, bp_dir, pack_sizes, encoding_only=True)
    sp_pct = build_prune_vs_vary_pct_by_abbrev(sprintz_vary_dir, sprintz_prune_dir, pack_sizes, encoding_only=True)
    common = _filter_abbrevs([a for a in bp_pct if a in sp_pct], exclude_abbrevs)
    if not common:
        return (None, None, None)
    vals_bp = [bp_pct[a] for a in common]
    vals_sp = [sp_pct[a] for a in common]
    return (common, vals_bp, vals_sp)

def plot_bp_sprintz_prune_vs_vary_pack_time(bp_vary_dir='output_BP_vary_pack_size', bp_dir='output_BP', sprintz_vary_dir='output_Sprintz_vary_pack_size', sprintz_prune_dir='output_Sprintz_only_Prune_Plus_RMQ_all_no8', outpath='figure_for_paper/bp_sprintz_prune_vs_vary_pack_time.png', pack_sizes=None, exclude_abbrevs=None):
    common, vals_bp, vals_sp = compute_prune_vs_vary_time_pct(
        bp_vary_dir, bp_dir, sprintz_vary_dir, sprintz_prune_dir, pack_sizes, exclude_abbrevs=exclude_abbrevs
    )
    if common is None:
        print('No paired CSVs for BP/Sprintz prune vs vary-pack-size time plot.')
        return False
    # common already in dictionary order
    x = np.arange(len(common))
    width = 0.65
    title_fs = 16
    label_fs = 16
    tick_fs = 16
    annot_fs = 14
    fig_w = max(10, len(common) * 0.28)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 8.3), sharex=True)

    def draw_panel(ax, values, color, title):
        bars = ax.bar(x, values, width, color=color)
        ax.tick_params(axis='y', labelsize=tick_fs)
        ax.yaxis.set_major_formatter(_PCT_1DEC)
        ax.set_ylabel('Time Reduction (%)', fontsize=label_fs)
        ax.set_title(title, fontsize=title_fs)
        top = max(values) if values else 0
        ax.set_ylim(0, max(top * 1.12, 5.0))
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + (top * 0.01 if top > 0 else 0.5), _fmt_bar_text_1dec(v), ha='center', va='bottom', rotation=15, fontsize=annot_fs)
    draw_panel(axes[0], vals_bp, 'C0', '(a) BP (Prune-RMQ) vs. trying pack sizes $2^1–2^{10}$')
    draw_panel(axes[1], vals_sp, 'C1', '(b) Sprintz (Prune-RMQ) vs. trying pack sizes $2^1–2^{10}$')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(common, rotation=45, ha='right', fontsize=tick_fs)
    axes[1].set_xlabel('Dataset', fontsize=label_fs)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    try:
        eps_path = os.path.splitext(outpath)[0] + '.eps'
        plt.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    print(f'Saved plot: {outpath}')
    return True

def aligned_abbrevs_for_combined(cols_impr, common_time):
    cset, tset = (set(cols_impr), set(common_time))
    return sort_dataset_abbrevs([a for a in cset if a in tset])

def compute_ratio_improvements(df):
    cols = sort_dataset_abbrevs([c for c in df.columns if c in dataset_mapping.values()])

    def inv(val):
        try:
            return 1.0 / float(val)
        except Exception:
            return np.nan
    bp_inv = [inv(df.at['BP', c]) if 'BP' in df.index and pd.notna(df.at['BP', c]) else np.nan for c in cols]
    bp_all_inv = [inv(df.at['BP (Prune-RMQ)', c]) if 'BP (Prune-RMQ)' in df.index and pd.notna(df.at['BP (Prune-RMQ)', c]) else np.nan for c in cols]
    sp_inv = [inv(df.at['Sprintz', c]) if 'Sprintz' in df.index and pd.notna(df.at['Sprintz', c]) else np.nan for c in cols]
    sp_all_inv = [inv(df.at['Sprintz (Prune-RMQ)', c]) if 'Sprintz (Prune-RMQ)' in df.index and pd.notna(df.at['Sprintz (Prune-RMQ)', c]) else np.nan for c in cols]
    bp_impr = [(a / b - 1.0) * 100.0 if not np.isnan(a) and (not np.isnan(b)) and (b != 0) else np.nan for a, b in zip(bp_all_inv, bp_inv)]
    sp_impr = [(a / b - 1.0) * 100.0 if not np.isnan(a) and (not np.isnan(b)) and (b != 0) else np.nan for a, b in zip(sp_all_inv, sp_inv)]
    return (cols, bp_impr, sp_impr)

def plot_combined_improvement_and_time(df, bp_vary_dir='output_BP_vary_pack_size', bp_dir='output_BP', sprintz_vary_dir='output_Sprintz_vary_pack_size', sprintz_prune_dir='output_Sprintz_only_Prune_Plus_RMQ_all_no8', outpath='figure_for_paper/improvement_bp_sprintz_combined.png', pack_sizes=None, exclude_abbrevs=None):
    cols, bp_impr, sp_impr = compute_ratio_improvements(df)
    common_t, vals_bp_t, vals_sp_t = compute_prune_vs_vary_time_pct(
        bp_vary_dir, bp_dir, sprintz_vary_dir, sprintz_prune_dir, pack_sizes, exclude_abbrevs=exclude_abbrevs
    )
    if common_t is None or not cols:
        print('Could not build combined BP/Sprintz figure (missing ratio or time data).')
        return False
    bp_impr_d = dict(zip(cols, bp_impr))
    sp_impr_d = dict(zip(cols, sp_impr))
    bp_t_d = dict(zip(common_t, vals_bp_t))
    sp_t_d = dict(zip(common_t, vals_sp_t))
    abbrevs = _filter_abbrevs(aligned_abbrevs_for_combined(cols, common_t), exclude_abbrevs)
    if not abbrevs:
        print('No dataset overlap between camel_ratio and vary-pack timing CSVs.')
        return False
    x = np.arange(len(abbrevs))
    bp_impr_v = [bp_impr_d[a] for a in abbrevs]
    sp_impr_v = [sp_impr_d[a] for a in abbrevs]
    bp_t_v = [bp_t_d[a] for a in abbrevs]
    sp_t_v = [sp_t_d[a] for a in abbrevs]
    title_fs = 16
    label_fs = 16
    tick_fs = 16
    annot_fs = 14
    width_impr = 0.35
    width_time = 0.65
    col_w = max(8, len(abbrevs) * 0.28)
    fig, axes = plt.subplots(2, 2, figsize=(col_w * 1.3, 8.3), sharex=True)

    def annotate_impr(ax, bars, values, top):
        for b, v in zip(bars, values):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + (top * 0.01 if top > 0 else 1.0), _fmt_bar_text_1dec(v), ha='center', va='bottom', rotation=90, fontsize=annot_fs)

    def annotate_time(ax, bars, values, top):
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + (top * 0.01 if top > 0 else 0.5), _fmt_bar_text_1dec(v), ha='center', va='bottom', rotation=90, fontsize=annot_fs)
    ax = axes[0, 0]
    bars = ax.bar(x, np.nan_to_num(bp_impr_v, nan=0.0), width_impr, color='C0')
    ax.set_ylabel('Ratio Improvement (%)', fontsize=label_fs)
    ax.set_title('(a) BP-Prune-RMQ vs Vanilla BP', fontsize=title_fs)
    ax.set_ylim(0, 25)
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.yaxis.set_major_formatter(_PCT_1DEC)
    top0 = max([v for v in bp_impr_v if not np.isnan(v)] or [0])
    annotate_impr(ax, bars, bp_impr_v, top0)
    ax = axes[0, 1]
    bars = ax.bar(x, bp_t_v, width_time, color='C0')
    ax.set_ylabel('Time Reduction (%)', fontsize=label_fs)
    ax.set_title('(b) BP-Prune-RMQ vs Vanilla BP', fontsize=title_fs)
    top_b = max(bp_t_v) if bp_t_v else 0
    ax.set_ylim(0, 150)
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.yaxis.set_major_formatter(_PCT_1DEC)
    annotate_time(ax, bars, bp_t_v, top_b)
    ax = axes[1, 0]
    bars = ax.bar(x, np.nan_to_num(sp_impr_v, nan=0.0), width_impr, color='C1')
    ax.set_ylabel('Ratio Improvement (%)', fontsize=label_fs)
    ax.set_title('(c) Sprintz-Prune-RMQ vs Vanilla Sprintz', fontsize=title_fs, x=0.4)
    ax.set_ylim(0, 25)
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.yaxis.set_major_formatter(_PCT_1DEC)
    top1 = max([v for v in sp_impr_v if not np.isnan(v)] or [0])
    annotate_impr(ax, bars, sp_impr_v, top1)
    ax = axes[1, 1]
    bars = ax.bar(x, sp_t_v, width_time, color='C1')
    ax.set_ylabel('Time Reduction (%)', fontsize=label_fs)
    ax.set_title('(d) Sprintz-Prune-RMQ vs Vanilla Sprintz', fontsize=title_fs, x=0.4)
    top_s = max(sp_t_v) if sp_t_v else 0
    ax.set_ylim(0, 150)
    ax.tick_params(axis='y', labelsize=tick_fs)
    ax.yaxis.set_major_formatter(_PCT_1DEC)
    annotate_time(ax, bars, sp_t_v, top_s)
    for ax in axes[1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(abbrevs, rotation=90, ha='center', va='top', fontsize=tick_fs)
        ax.set_xlabel('Dataset', fontsize=label_fs)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    try:
        eps_path = os.path.splitext(outpath)[0] + '.eps'
        plt.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    print(f'Saved plot: {outpath}')
    return True

def read_ratio_file(primary='camel_ratio.xlsx'):
    if os.path.exists(primary):
        return pd.read_excel(primary, index_col=0)
    return None

def plot_bp_vs_bpall(df, alg1='BP', alg2='BP (Prune-RMQ)', outpath='figure_for_paper/bp_vs_bpall.png'):
    if alg1 not in df.index and alg2 not in df.index:
        print(f'Neither {alg1} nor {alg2} found in input file. Available rows: {list(df.index)}')
        return False
    cols = sort_dataset_abbrevs([c for c in df.columns if c in dataset_mapping.values()])
    vals1 = [float(1 / df.at[alg1, c]) if alg1 in df.index and pd.notna(df.at[alg1, c]) else np.nan for c in cols]
    vals2 = [float(1 / df.at[alg2, c]) if alg2 in df.index and pd.notna(df.at[alg2, c]) else np.nan for c in cols]
    x = np.arange(len(cols))
    width = 0.35
    title_fs = 16
    label_fs = 16
    tick_fs = 16
    legend_fs = 16
    annot_fs = 16
    plt.figure(figsize=(max(10, len(cols) * 0.28), 6))
    bar1 = plt.bar(x - width / 2, np.nan_to_num(vals1, nan=0.0), width, label=alg1, color='C0')
    bar2 = plt.bar(x + width / 2, np.nan_to_num(vals2, nan=0.0), width, label=alg2, color='C1')
    plt.xticks(x, cols, rotation=30, ha='right', fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.ylabel('Compression Ratio', fontsize=label_fs)
    plt.title(f'Improved compression ratio of each dataset', fontsize=title_fs)
    plt.legend(fontsize=legend_fs)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    try:
        eps_path = os.path.splitext(outpath)[0] + '.eps'
        plt.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    print(f'Saved plot: {outpath}')
    return True

def plot_percent_improvements(df, outpath='figure_for_paper/improvement_bp_sprintz.png', highlight_abbrevs=None, ylim_max=20, annot_rotation=90, label_fmt=None, panel_w_per_col=0.28, panel_w_min=6.5, exclude_abbrevs=None):
    hi = IMPROVEMENT_HIGHLIGHT_ABBREVS if highlight_abbrevs is None else highlight_abbrevs
    excl = _fig11_exclude(exclude_abbrevs)
    cols, bp_impr, sp_impr = compute_ratio_improvements(df)
    if not cols:
        print('No ratio columns found for improvement_bp_sprintz plot.')
        return False
    keep_idx = [i for i, c in enumerate(cols) if c not in excl]
    cols = [cols[i] for i in keep_idx]
    bp_impr = [bp_impr[i] for i in keep_idx]
    sp_impr = [sp_impr[i] for i in keep_idx]
    if not cols:
        print('No columns left after excluding improvement_bp_sprintz datasets.')
        return False

    def _display_label(c):
        return label_fmt(c) if label_fmt else c

    x = np.arange(len(cols))
    width = 0.35
    colors0 = [C0_DARK if c in hi else 'C0' for c in cols]
    colors1 = [C1_DARK if c in hi else 'C1' for c in cols]
    title_fs = 22
    label_fs = 22
    tick_fs = 22
    annot_fs = 22
    panel_w = max(panel_w_min, len(cols) * panel_w_per_col)
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, 6.0), sharey=True)
    ax0 = axes[0]
    bars0 = ax0.bar(x, np.nan_to_num(bp_impr, nan=0.0), width, color=colors0)
    ax0.set_xticks(x)
    ax0.set_xticklabels([_display_label(c) for c in cols], rotation=90, ha='center', va='top', fontsize=tick_fs)
    for lab, c in zip(ax0.get_xticklabels(), cols):
        if c in hi:
            lab.set_color('red')
    ax0.tick_params(axis='y', labelsize=tick_fs)
    ax0.yaxis.set_major_formatter(_PCT_1DEC)
    ax0.set_ylim(0, ylim_max)
    ax0.set_xlabel('Dataset', fontsize=label_fs)
    ax0.set_ylabel('Ratio Improvement (%)', fontsize=label_fs)
    ax0.set_title('(a) BP-Prune-RMQ vs vanilla BP', fontsize=title_fs)
    top0 = max([v for v in bp_impr if not np.isnan(v)] or [0])
    for b, v in zip(bars0, bp_impr):
        if np.isnan(v):
            continue
        ax0.text(b.get_x() + b.get_width() / 2, v + (top0 * 0.01 if top0 > 0 else 1.0), _fmt_bar_text_1dec(v), ha='center', va='bottom', rotation=annot_rotation, fontsize=annot_fs)
    ax1 = axes[1]
    bars1 = ax1.bar(x, np.nan_to_num(sp_impr, nan=0.0), width, color=colors1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([_display_label(c) for c in cols], rotation=90, ha='center', va='top', fontsize=tick_fs)
    for lab, c in zip(ax1.get_xticklabels(), cols):
        if c in hi:
            lab.set_color('red')
    ax1.tick_params(axis='y', labelsize=tick_fs)
    ax1.yaxis.set_major_formatter(_PCT_1DEC)
    ax1.set_ylim(0, ylim_max)
    ax1.set_xlabel('Dataset', fontsize=label_fs)
    ax1.tick_params(labelleft=False)
    ax1.set_title('(b) Sprintz-Prune-RMQ vs vanilla Sprintz', fontsize=title_fs, x=0.45)
    top1 = max([v for v in sp_impr if not np.isnan(v)] or [0])
    for b, v in zip(bars1, sp_impr):
        if np.isnan(v):
            continue
        ax1.text(b.get_x() + b.get_width() / 2, v + (top1 * 0.01 if top1 > 0 else 1.0), _fmt_bar_text_1dec(v), ha='center', va='bottom', rotation=annot_rotation, fontsize=annot_fs)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    try:
        eps_path = os.path.splitext(outpath)[0] + '.eps'
        plt.savefig(eps_path, format='eps', dpi=300, bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    print(f'Saved plot: {outpath}')
    return True

def main():
    parser = argparse.ArgumentParser(description='Plot BP vs BP (Prune-RMQ) compression ratios per dataset')
    parser.add_argument('--input', '-i', default='camel_ratio.xlsx', help='Primary input Excel file')
    parser.add_argument('--alg1', default='BP', help='First algorithm row name')
    parser.add_argument('--alg2', default='BP (Prune-RMQ)', help='Second algorithm row name')
    parser.add_argument('--output', '-o', default='figure_for_paper/bp_vs_bpall.png', help='Output image path')
    parser.add_argument('--vary-pack-dir', default=str(RESULTS_DIR / 'output_BP_vary_pack_size'), help='CSV dir from VaryPackSizeTest (sum times over pack sizes)')
    parser.add_argument('--bp-dir', default=str(RESULTS_DIR / 'output_BP'), help='CSV dir for BP (Prune-RMQ) single-run times')
    parser.add_argument('--vary-time-plot', default='figure_for_paper/bp_sprintz_prune_vs_vary_pack_time.png', help='Output path for BP+Sprintz prune vs vary-pack-size time reduction (2 subplots)')
    parser.add_argument('--sprintz-vary-pack-dir', default=str(RESULTS_DIR / 'output_Sprintz_vary_pack_size'), help='CSV dir from VaryPackSizeSprintzTest')
    parser.add_argument('--sprintz-prune-dir', default=str(RESULTS_DIR / 'output_Sprintz_only_Prune_Plus_RMQ_all_no8'), help='CSV dir for Sprintz+Prune+Plus+RMQ single-run times')
    args = parser.parse_args()
    df = read_ratio_file(args.input)
    if df is None:
        print('Could not find input files. Produce camel_ratio.xlsx first (run combine_results.py).')
        return
    if 'avg_ratio' in df.columns:
        df = df.drop(columns=['avg_ratio'])
    plot_percent_improvements(
        df,
        outpath='figure_for_paper/improvement_bp_sprintz.png',
        ylim_max=35,
        annot_rotation=0,
        panel_w_per_col=0.34,
        panel_w_min=7.0,
        exclude_abbrevs=IMPROVEMENT_EXCLUDE_ADD_DATASET_ABBREVS,
    )
    plot_bp_sprintz_prune_vs_vary_pack_time(
        bp_vary_dir=args.vary_pack_dir,
        bp_dir=args.bp_dir,
        sprintz_vary_dir=args.sprintz_vary_pack_dir,
        sprintz_prune_dir=args.sprintz_prune_dir,
        outpath=args.vary_time_plot,
        exclude_abbrevs=IMPROVEMENT_EXCLUDE_ADD_DATASET_ABBREVS,
    )
if __name__ == '__main__':
    main()
