import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from integer_datasets import FIGURE_DIR, INTEGER_DATASET_MAPPING, PAPER_TABLE_DATASET_FILES, RESULTS_DIR
from plot_baselines import populate_fig12_baseline_boxes
from plot_metrics import csv_time_to_ns_per_point

data_dirs = {'BP': str(RESULTS_DIR / 'output_BP_vary_pack_size'), 'Sprintz': str(RESULTS_DIR / 'output_Sprintz_vary_pack_size')}
dataset_mapping = INTEGER_DATASET_MAPPING
FIG12_DATASET_FILES = PAPER_TABLE_DATASET_FILES
print(
    f'fig12: {len(FIG12_DATASET_FILES)} paper-table datasets (BP & Sprintz): '
    + ', '.join(sorted(dataset_mapping[f] for f in FIG12_DATASET_FILES if f in dataset_mapping))
)
vector_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
compression_ratio_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}
encode_time_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}
decode_time_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}
for algorithm, data_dir in data_dirs.items():
    print(f'Processing algorithm: {algorithm}')
    for filename in sorted((f for f in os.listdir(data_dir) if f.endswith('.csv') and f != '.DS_Store' and (f in dataset_mapping) and (f in FIG12_DATASET_FILES))):
        dataset_name = dataset_mapping.get(filename, filename)
        print(f'  Processing dataset: {dataset_name} ({filename})')
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    pack_size = row['Pack size']
                    try:
                        pack_size = int(pack_size)
                    except:
                        continue
                    if pack_size in vector_sizes:
                        compression_ratio = float(row['Compression Ratio'])
                        compression_ratio_data[algorithm][pack_size].append(1 / compression_ratio)
                        encode_time = float(row['Encoding Time'])
                        decode_time = float(row['Decoding Time'])
                        et = csv_time_to_ns_per_point(encode_time, result_dir=data_dir)
                        dt = csv_time_to_ns_per_point(decode_time, result_dir=data_dir)
                        if np.isfinite(et):
                            encode_time_data[algorithm][pack_size].append(et)
                        if np.isfinite(dt):
                            decode_time_data[algorithm][pack_size].append(dt)
            except Exception as e:
                print(f'    Error processing {file_path}: {e}')
                continue
avg_compression_ratio = {}
avg_encode_time = {}
avg_decode_time = {}
std_compression_ratio = {}
std_encode_time = {}
std_decode_time = {}
for algorithm in data_dirs.keys():
    avg_compression_ratio[algorithm] = []
    avg_encode_time[algorithm] = []
    avg_decode_time[algorithm] = []
    std_compression_ratio[algorithm] = []
    std_encode_time[algorithm] = []
    std_decode_time[algorithm] = []
    for size in vector_sizes:
        if compression_ratio_data[algorithm][size]:
            cr_values = np.array(compression_ratio_data[algorithm][size])
            et_values = np.array(encode_time_data[algorithm][size])
            dt_values = np.array(decode_time_data[algorithm][size])
            avg_cr = np.mean(cr_values)
            std_cr = np.std(cr_values)
            avg_compression_ratio[algorithm].append(avg_cr)
            std_compression_ratio[algorithm].append(std_cr)
            avg_et = np.mean(et_values)
            std_et = np.std(et_values)
            avg_encode_time[algorithm].append(avg_et)
            std_encode_time[algorithm].append(std_et)
            avg_dt = np.mean(dt_values)
            std_dt = np.std(dt_values)
            avg_decode_time[algorithm].append(avg_dt)
            std_decode_time[algorithm].append(std_dt)
        else:
            avg_compression_ratio[algorithm].append(0)
            avg_encode_time[algorithm].append(0)
            avg_decode_time[algorithm].append(0)
            std_compression_ratio[algorithm].append(0)
            std_encode_time[algorithm].append(0)
            std_decode_time[algorithm].append(0)
box_cr = {'bp': {}, 'sz': {}}
box_enc = {'bp': {}, 'sz': {}}
box_dec = {'bp': {}, 'sz': {}}
print('\nMean compression ratio:')
for size, ratio in zip(vector_sizes, avg_compression_ratio['BP']):
    print(f'  Pack size {size}: {ratio:.4f}')
populate_fig12_baseline_boxes(box_cr, box_enc, box_dec, dataset_names=FIG12_DATASET_FILES)


TRIM_DROP_LOW = 1
TRIM_DROP_HIGH = 1


def _trimmed_vals(vals, log_scale=False):
    a = np.asarray(vals, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if log_scale:
        a = a[a > 0]
    if a.size <= TRIM_DROP_LOW + TRIM_DROP_HIGH:
        return a
    idx = np.argsort(a)
    return a[idx[TRIM_DROP_LOW:a.size - TRIM_DROP_HIGH]]


def trimmed_mean(vals, log_scale=False):
    a = _trimmed_vals(vals, log_scale)
    if a.size == 0:
        return np.nan
    return float(np.mean(a))


def std_symmetric_errors(vals, mean=None, log_scale=False):
    """Error-bar half-widths: trimmed mean ± std (drop min/max dataset each)."""
    a = _trimmed_vals(vals, log_scale)
    if a.size < 2:
        return (0.0, 0.0)
    s = float(np.std(a))
    if log_scale and mean is not None and np.isfinite(mean) and float(mean) > 0:
        s = min(s, 0.99 * float(mean))
    return (s, s)

def ylim_compression_ab_including_errors(
    vary_algo,
    raw_dict,
    box_store,
    family_key,
    baseline_keys_labels,
    bar_key_order=('all', 'prune', 'prune_rmq'),
    pad_frac=0.05,
    min_top_pad=0.55,
):
    """Y limits for (a)(b): vary-pack curve + optimal-pack baseline bars fully visible."""
    vary_ys = []
    for s in vector_sizes:
        vals = list(raw_dict[vary_algo].get(s, []))
        if not vals:
            vals = [0.0]
        m = trimmed_mean(vals, log_scale=False)
        if not np.isfinite(m):
            continue
        el, eu = std_symmetric_errors(vals, m, log_scale=False)
        vary_ys.extend([m - el, m + eu])
    bar_tops = []
    for key in bar_key_order:
        arr = box_store.get(family_key, {}).get(key)
        if arr is None or np.size(arr) == 0:
            continue
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if not a.size:
            continue
        m = trimmed_mean(a, log_scale=False)
        _, eu = std_symmetric_errors(a, m, log_scale=False)
        bar_tops.append(m + eu)
    if not vary_ys and not bar_tops:
        return None
    lo = min(vary_ys) if vary_ys else min(bar_tops)
    hi = max(vary_ys) if vary_ys else max(bar_tops)
    if bar_tops:
        hi = max(hi, max(bar_tops) + float(min_top_pad))
    span = hi - lo if hi > lo else max(abs(hi), abs(lo), 1.0) * 0.05
    pad = max(float(pad_frac) * span, 0.25)
    return (lo - pad, hi + 0.2 * pad)

def plot_vary_line_baseline_bars_by_packsize(
    ax,
    pack_sizes,
    vary_algo,
    raw_per_pack_dict,
    baseline_store,
    family_key,
    baseline_keys_labels,
    vary_color,
    baseline_colors,
    vary_legend_label,
    exponent_labels,
    fontsize,
    ylabel,
    title,
    ylim=None,
    yscale='linear',
    bar_key_order=('all', 'prune', 'prune_rmq'),
    bar_height_hlines=False,
    show_std_errorbar=False,
    baseline_std_errorbar=False,
    show_baseline_xticks=False,
    baseline_xtick_labels=None,
):
    group_pitch = 0.55
    n = len(pack_sizes)
    xs_line = np.arange(n, dtype=float) * group_pitch
    ys_line = []
    ys_err_lo = []
    ys_err_hi = []
    for size in pack_sizes:
        vals = list(raw_per_pack_dict[vary_algo].get(size, []))
        if len(vals) < 1:
            vals = [0.0]
        m = trimmed_mean(vals, log_scale=yscale == 'log')
        if yscale == 'log' and (not np.isfinite(m) or m <= 0):
            m = np.nan
        ys_line.append(m)
        el, eu = std_symmetric_errors(vals, m, log_scale=yscale == 'log')
        ys_err_lo.append(el)
        ys_err_hi.append(eu)
    if show_std_errorbar:
        yv = np.asarray(ys_line, dtype=float)
        mask = np.isfinite(yv)
        if np.any(mask):
            ax.errorbar(xs_line[mask], yv[mask], yerr=[np.asarray(ys_err_lo)[mask], np.asarray(ys_err_hi)[mask]], fmt='none', ecolor='black', elinewidth=1.4, capsize=3.2, capthick=1.2, zorder=5, clip_on=True, alpha=0.88)
    ax.plot(xs_line, ys_line, color=vary_color, linestyle='-', linewidth=2.0, marker='o', markersize=5, zorder=6, clip_on=True)
    gap_after_last_pack = 0.85
    bar_w = 0.14
    bar_gap = 0.07
    x_bar0 = (n - 1) * group_pitch + gap_after_last_pack
    x_bars = x_bar0 + np.arange(len(bar_key_order), dtype=float) * (bar_w + bar_gap)
    heights = []
    colors_b = []
    bar_err_lo = []
    bar_err_hi = []
    use_bar_err = baseline_std_errorbar
    for key in bar_key_order:
        arr = baseline_store.get(family_key, {}).get(key)
        if arr is not None and np.size(arr) > 0:
            a = np.asarray(arr, dtype=float).ravel()
            a = a[np.isfinite(a)]
            if a.size == 0:
                heights.append(0.0)
                bar_err_lo.append(0.0)
                bar_err_hi.append(0.0)
            else:
                mbar = trimmed_mean(a, log_scale=yscale == 'log')
                if not np.isfinite(mbar):
                    mbar = 0.0
                heights.append(mbar)
                if use_bar_err:
                    elb, eub = std_symmetric_errors(a, mbar, log_scale=yscale == 'log')
                    bar_err_lo.append(elb)
                    bar_err_hi.append(eub)
        else:
            heights.append(0.0)
            if use_bar_err:
                bar_err_lo.append(0.0)
                bar_err_hi.append(0.0)
        colors_b.append(baseline_colors[key])
    if use_bar_err:
        err_kw = dict(elinewidth=1.3, capthick=1.1, alpha=0.88, zorder=5)
        for xi, h, cb, el, eu in zip(x_bars, heights, colors_b, bar_err_lo, bar_err_hi):
            ax.bar([xi], [h], width=bar_w, color=cb, alpha=0.88, edgecolor='none', align='center', zorder=4, yerr=[[el], [eu]], ecolor='black', capsize=3.0, error_kw=err_kw)
    else:
        ax.bar(x_bars, heights, width=bar_w, color=colors_b, alpha=0.88, edgecolor='none', align='center', zorder=4)
    sep_x = (float(xs_line[-1]) + float(x_bar0)) / 2.0
    if show_baseline_xticks:
        bl_labels = baseline_xtick_labels or [lbl for _, lbl in baseline_keys_labels]
        ax.set_xticks(list(xs_line) + list(x_bars))
        tick_labels = list(exponent_labels) + list(bl_labels)
        ax.set_xticklabels(tick_labels, fontsize=fontsize * 0.82)
        for lbl in ax.get_xticklabels()[len(exponent_labels):]:
            lbl.set_rotation(28)
            lbl.set_ha('right')
        ax.axvline(sep_x, color='0.55', linestyle=':', linewidth=1.0, zorder=2, clip_on=True)
    else:
        ax.set_xticks(xs_line)
        ax.set_xticklabels(exponent_labels)
    x_right = float(x_bars[-1] + bar_w / 2 + 0.55)
    ax.set_xlim(xs_line[0] - 0.12 * group_pitch, x_right)
    ax.axvspan(x_bar0 - 0.08, x_right, color='0.96', zorder=0, clip_on=True)
    ax.set_xlabel('Pack Size $s$', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize, x=0.42)
    ax.tick_params(labelsize=fontsize)
    if yscale == 'log':
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)
    if bar_height_hlines:
        for h, c in zip(heights, colors_b):
            if not np.isfinite(h):
                continue
            if yscale == 'log' and h <= 0:
                continue
            ax.axhline(y=h, color=c, linestyle='--', linewidth=1.2, alpha=0.88, zorder=3, clip_on=True)

def _positive_finite(vals):
    return [float(x) for x in vals if np.isfinite(x) and float(x) > 0]

def ylim_compression_tight(vary_algo, raw_dict, box_store, family_key, baseline_keys_labels):
    v = []
    for s in vector_sizes:
        v.extend(_positive_finite(raw_dict[vary_algo].get(s, [])))
    for key, _ in baseline_keys_labels:
        arr = box_store.get(family_key, {}).get(key)
        if arr is not None:
            v.extend(_positive_finite(np.ravel(arr)))
    if len(v) < 2:
        return None
    lo, hi = np.percentile(v, [20, 80])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return (float(max(0.0, lo)), float(hi))

def ylim_encode_log(vary_algo, raw_dict, box_store, family_key, baseline_keys_labels, pad=0.12):
    v = []
    for s in vector_sizes:
        v.extend(_positive_finite(raw_dict[vary_algo].get(s, [])))
    for key, _ in baseline_keys_labels:
        arr = box_store.get(family_key, {}).get(key)
        if arr is not None:
            v.extend(_positive_finite(np.ravel(arr)))
    if not v:
        return (10.0, 2000.0)
    lo, hi = (min(v), max(v))
    if hi <= lo:
        hi = lo * 1.2 + 1.0
    return (max(1e-3, lo * (1 - pad)), hi * (1 + pad))

# Match bp_vary_page_size.png (fig13): same panel font, width, row height; total height ∝ 3/4 rows.
FIG13_PANEL_ROWS = 4
FIG13_FIG_HEIGHT = 24.0  # 20 + 1 inch per subplot row
FIG12_PANEL_ROWS = 3
FIG12_ROW_HEIGHT_IN = FIG13_FIG_HEIGHT / FIG13_PANEL_ROWS - 0.5  # 5.5 inch per row
FIG_PANEL_FONTSIZE = 22
FIG_FIGWIDTH = 14
FIG_FIGHEIGHT = FIG12_ROW_HEIGHT_IN * FIG12_PANEL_ROWS  # 16.5
FIG_WSPACE = 0.065 * 4 * (2 / 3) + 0.05
FIG_HSPACE = 0.38

fig, axs = plt.subplots(3, 2, figsize=(FIG_FIGWIDTH, FIG_FIGHEIGHT))
plt.subplots_adjust(wspace=FIG_WSPACE, hspace=FIG_HSPACE)
fontsize = FIG_PANEL_FONTSIZE
exponents = [int(np.log2(ps)) for ps in vector_sizes]
exponent_labels = [f'$2^{{{exp}}}$' for exp in exponents]
plt.rcParams.update({'font.size': fontsize})
bp_bl_meta = [('prune_rmq', 'BP–Prune–RMQ'), ('prune', 'BP–Prune'), ('all', 'BP–All')]
bp_bl_colors = {'prune_rmq': '#d62728', 'prune': '#2ca02c', 'all': '#ff7f0e'}
sz_bl_meta = [('prune_rmq', 'Sprintz–Prune–RMQ'), ('prune', 'Sprintz–Prune'), ('all', 'Sprintz–All')]
sz_bl_colors = {'prune_rmq': '#17becf', 'prune': '#e377c2', 'all': '#8c564b'}
_ylim_a = (5.0, 8.5)
plot_vary_line_baseline_bars_by_packsize(axs[0, 0], vector_sizes, 'BP', compression_ratio_data, box_cr, 'bp', bp_bl_meta, '#1f77b4', bp_bl_colors, 'BP', exponent_labels, fontsize, 'Compression ratio', '(a) BP: compression ratio', ylim=_ylim_a, bar_height_hlines=True)
_ylim_c = ylim_encode_log('BP', encode_time_data, box_enc, 'bp', bp_bl_meta)
plot_vary_line_baseline_bars_by_packsize(axs[1, 0], vector_sizes, 'BP', encode_time_data, box_enc, 'bp', bp_bl_meta, '#1f77b4', bp_bl_colors, 'BP', exponent_labels, fontsize, 'Time (ns/point)', '(c) BP: compression time', ylim=_ylim_c, yscale='log')
plot_vary_line_baseline_bars_by_packsize(axs[2, 0], vector_sizes, 'BP', decode_time_data, box_dec, 'bp', bp_bl_meta, '#1f77b4', bp_bl_colors, 'BP', exponent_labels, fontsize, 'Time (ns/point)', '(e) BP: decompression time', ylim=(0, 20))
_ylim_b = (10.0, 16.0)
plot_vary_line_baseline_bars_by_packsize(axs[0, 1], vector_sizes, 'Sprintz', compression_ratio_data, box_cr, 'sz', sz_bl_meta, '#9467bd', sz_bl_colors, 'Sprintz', exponent_labels, fontsize, 'Compression ratio', '(b) Sprintz: compression ratio', ylim=_ylim_b, bar_height_hlines=True)
_ylim_d = ylim_encode_log('Sprintz', encode_time_data, box_enc, 'sz', sz_bl_meta)
plot_vary_line_baseline_bars_by_packsize(axs[1, 1], vector_sizes, 'Sprintz', encode_time_data, box_enc, 'sz', sz_bl_meta, '#9467bd', sz_bl_colors, 'Sprintz', exponent_labels, fontsize, 'Time (ns/point)', '(d) Sprintz: compression time', ylim=_ylim_d, yscale='log')
plot_vary_line_baseline_bars_by_packsize(axs[2, 1], vector_sizes, 'Sprintz', decode_time_data, box_dec, 'sz', sz_bl_meta, '#9467bd', sz_bl_colors, 'Sprintz', exponent_labels, fontsize, 'Time (ns/point)', '(f) Sprintz: decompression time', ylim=(0, 20))
legend_handles = [Line2D([0], [0], color='#1f77b4', linestyle='-', linewidth=2.0, marker='o', markersize=5, label='BP'), Patch(facecolor='#ff7f0e', alpha=0.88, edgecolor='0.15', label='BP–All'), Patch(facecolor='#2ca02c', alpha=0.88, edgecolor='0.15', label='BP–Prune'), Patch(facecolor='#d62728', alpha=0.88, edgecolor='0.15', label='BP–Prune–RMQ'), Line2D([0], [0], color='#9467bd', linestyle='-', linewidth=2.0, marker='o', markersize=5, label='Sprintz'), Patch(facecolor='#8c564b', alpha=0.88, edgecolor='0.15', label='Sprintz–All'), Patch(facecolor='#e377c2', alpha=0.88, edgecolor='0.15', label='Sprintz–Prune'), Patch(facecolor='#17becf', alpha=0.88, edgecolor='0.15', label='Sprintz–Prune–RMQ')]
fig.legend(
    legend_handles,
    [h.get_label() for h in legend_handles],
    loc='upper center',
    ncol=4,
    labelspacing=0.15,
    handletextpad=0.35,
    columnspacing=0.9,
    fontsize=fontsize,
    bbox_to_anchor=(0.48, 0.98),
)
output_dir = str(FIGURE_DIR)
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'bp_vary_pack_size.png'), dpi=400, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'bp_vary_pack_size.eps'), format='eps', dpi=400, bbox_inches='tight')
print('\nDetailed statistics:')
print('=' * 70)
print(f"{'Pack Size':>10} {'Compression Ratio':>20} {'Encode Throughput':>20} {'Decode Throughput':>20}")
print('-' * 70)
for i, size in enumerate(vector_sizes):
    ratio = avg_compression_ratio['BP'][i]
    encode_tp = avg_encode_time['BP'][i]
    decode_tp = avg_decode_time['BP'][i]
    print(f'{size:>10} {ratio:>20.4f} {encode_tp:>20.2f} {decode_tp:>20.2f}')
