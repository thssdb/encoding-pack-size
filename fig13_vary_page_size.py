import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
from integer_datasets import ICDE_ROOT, FIGURE_DIR, INTEGER_DATASET_MAPPING, PAPER_TABLE_DATASET_FILES, RESULTS_DIR
from proposition_prune_count import build_vary_page_prune_rate_data

_R = RESULTS_DIR
data_dirs = {'BP': str(_R / 'output_BP_vary_page_size'), 'Sprintz': str(_R / 'output_sprintz_vary_page_size'), 'BP-All': str(_R / 'output_BP_vary_page_size_N2'), 'Sprintz-All': str(_R / 'output_Sprintz_vary_page_size_N2'), 'BP-Prune': str(_R / 'output_BP_only_Prune_vary_page_size'), 'Sprintz-Prune': str(_R / 'output_Sprintz_only_Prune_vary_page_size'), 'BP-Prune-RMQ': str(_R / 'output_BP_Prune_RMQ_vary_page_size'), 'Sprintz-Prune-RMQ': str(_R / 'output_Sprintz_Prune_RMQ_vary_page_size')}
dataset_mapping = INTEGER_DATASET_MAPPING
FIG13_DATASET_FILES = PAPER_TABLE_DATASET_FILES
print(
    f'fig13: {len(FIG13_DATASET_FILES)} paper-table datasets: '
    + ', '.join(sorted(dataset_mapping[f] for f in FIG13_DATASET_FILES if f in dataset_mapping))
)
vector_sizes = [16 * 8, 32 * 8, 64 * 8, 128 * 8, 256 * 8, 512 * 8, 1024 * 8]
compression_ratio_data = {algo: {size: {} for size in vector_sizes} for algo in data_dirs.keys()}
encode_time_data = {algo: {size: {} for size in vector_sizes} for algo in data_dirs.keys()}
decode_time_data = {algo: {size: {} for size in vector_sizes} for algo in data_dirs.keys()}
for algorithm, data_dir in data_dirs.items():
    print(f'Processing algorithm: {algorithm}')
    for filename in os.listdir(data_dir):
        if (
            not filename.endswith('.csv')
            or filename == '.DS_Store'
            or filename not in dataset_mapping
            or filename not in FIG13_DATASET_FILES
        ):
            continue
        dataset_name = dataset_mapping.get(filename, filename)
        print(f'  Processing dataset: {dataset_name} ({filename})')
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    pack_size = row['m']
                    try:
                        pack_size = int(pack_size)
                    except:
                        continue
                    if pack_size in vector_sizes:
                        compression_ratio = float(row['Compression Ratio'])
                        if not (np.isfinite(compression_ratio) and compression_ratio > 0):
                            continue
                        compression_ratio_data[algorithm][pack_size][filename] = 1 / compression_ratio
                        encode_time = float(row['Encoding Time'])
                        if np.isfinite(encode_time) and encode_time > 0:
                            encode_time_data[algorithm][pack_size][filename] = 1 / (encode_time / 8000)
                        decode_time = float(row['Decoding Time'])
                        if np.isfinite(decode_time) and decode_time > 0:
                            decode_time_data[algorithm][pack_size][filename] = 1 / (decode_time / 8000)
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
            cr_values = np.array(list(compression_ratio_data[algorithm][size].values()))
            et_values = np.array(list(encode_time_data[algorithm][size].values()))
            dt_values = np.array(list(decode_time_data[algorithm][size].values()))
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
            avg_compression_ratio[algorithm].append(np.nan)
            avg_encode_time[algorithm].append(np.nan)
            avg_decode_time[algorithm].append(np.nan)
            std_compression_ratio[algorithm].append(np.nan)
            std_encode_time[algorithm].append(np.nan)
            std_decode_time[algorithm].append(np.nan)
print('\nMean compression ratio:')
for size, ratio in zip(vector_sizes, avg_compression_ratio['BP']):
    print(f'  Pack size {size}: {ratio:.4f}')
algorithm_palette = {'BP': '#1f77b4', 'BP-All': '#ff7f0e', 'BP-Prune': '#2ca02c', 'BP-Prune-RMQ': '#d62728', 'Sprintz': '#9467bd', 'Sprintz-All': '#8c564b', 'Sprintz-Prune': '#e377c2', 'Sprintz-Prune-RMQ': '#17becf'}
algorithm_markers = {'BP': 'o', 'BP-All': 's', 'BP-Prune': '^', 'BP-Prune-RMQ': 'D', 'Sprintz': 'P', 'Sprintz-All': 'X', 'Sprintz-Prune': 'v', 'Sprintz-Prune-RMQ': '*'}

def _positive_finite(vals):
    return [float(x) for x in vals if np.isfinite(x) and float(x) > 0]

def _as_value_map(raw_at_pack):
    if isinstance(raw_at_pack, dict):
        return raw_at_pack
    return {i: v for i, v in enumerate(raw_at_pack)}

TRIM_DROP_LOW = 1
TRIM_DROP_HIGH = 1


def _trimmed_vals(vals, log_scale=False):
    a = [float(x) for x in vals if np.isfinite(float(x))]
    if log_scale:
        a = [x for x in a if x > 0]
    if len(a) <= TRIM_DROP_LOW + TRIM_DROP_HIGH:
        return a
    sa = sorted(a)
    return sa[TRIM_DROP_LOW:len(sa) - TRIM_DROP_HIGH]


def trimmed_mean(vals, log_scale=False):
    a = _trimmed_vals(vals, log_scale)
    if not a:
        return np.nan
    return float(np.mean(a))


def yerr_std(vals, mean=None, log_scale=False):
    """Error-bar half-width: trimmed mean ± std (drop min/max dataset each)."""
    a = _trimmed_vals(vals, log_scale)
    if len(a) < 2:
        return 0.0
    s = float(np.std(a))
    if log_scale and mean is not None and np.isfinite(mean) and float(mean) > 0:
        s = min(s, 0.99 * float(mean))
    return s

def ylim_compression_bars_with_errorbar(algorithm_keys, pack_sizes, raw_per_algo_per_pack, show_std_errorbar=True, pad_frac=0.04, pad_min=0.02):
    lo, hi = (np.inf, -np.inf)
    for size in pack_sizes:
        for algo in algorithm_keys:
            vals_map = _as_value_map(raw_per_algo_per_pack[algo].get(size, {}))
            vals = list(vals_map.values())
            if not vals:
                continue
            m = trimmed_mean(vals)
            if not np.isfinite(m):
                continue
            e = float(yerr_std(vals, m)) if show_std_errorbar else 0.0
            if not np.isfinite(e) or e < 0:
                e = 0.0
            lo = min(lo, m - e)
            hi = max(hi, m + e)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None
    span = hi - lo
    pad = max(span * pad_frac, pad_min)
    return (float(lo - pad), float(hi + pad))

def plot_grouped_boxes_multi_algo(ax, pack_sizes, algorithm_keys, raw_per_algo_per_pack, algorithm_palette, exponent_labels, fontsize, ylabel, title, xlabel='Page Size $n$', ylim=None, yscale='linear', show_boxes=True, stagger_algorithms=True, show_std_errorbar=False):
    n_slot = len(algorithm_keys)
    bw = 0.14
    gap = 0.06
    step = bw + gap
    inter_group = 0.48
    if stagger_algorithms:
        spread = (n_slot - 1) * step if n_slot > 1 else 0.0
        offs = np.linspace(-spread / 2, spread / 2, n_slot) if n_slot > 1 else np.array([0.0])
        group_pitch = spread + bw + inter_group
    else:
        offs = np.zeros(n_slot, dtype=float)
        group_pitch = bw + inter_group
    xs_mean = [[] for _ in range(n_slot)]
    ys_mean = [[] for _ in range(n_slot)]
    yerr_per_j = [[] for _ in range(n_slot)]
    for i, size in enumerate(pack_sizes):
        center = i * group_pitch
        for j, algo in enumerate(algorithm_keys):
            vals_map = _as_value_map(raw_per_algo_per_pack[algo].get(size, {}))
            vals = list(vals_map.values())
            if len(vals) < 1:
                vals = [0.0]
                vals_map = {0: 0.0}
            pos = center + offs[j]
            m = trimmed_mean(vals, log_scale=yscale == 'log')
            if yscale == 'log' and (not np.isfinite(m) or m <= 0):
                m = np.nan
            xs_mean[j].append(pos)
            ys_mean[j].append(m)
            if show_std_errorbar:
                yerr_per_j[j].append(yerr_std(vals, m, log_scale=yscale == 'log'))
            else:
                yerr_per_j[j].append(0.0)
            if show_boxes:
                bp = ax.boxplot([vals], positions=[pos], widths=bw, patch_artist=True, manage_ticks=False, showfliers=False)
                color = algorithm_palette[algo]
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.88)
                    patch.set_edgecolor('none')
                    patch.set_linewidth(0)
                for ln in bp['medians']:
                    ln.set_color('0.15')
                    ln.set_linewidth(0.9)
                for ln in bp['whiskers']:
                    ln.set_visible(False)
                for ln in bp['caps']:
                    ln.set_visible(False)
    for j, algo in enumerate(algorithm_keys):
        if len(xs_mean[j]) == 0:
            continue
        mk = algorithm_markers.get(algo, 'o')
        ax.plot(xs_mean[j], ys_mean[j], color=algorithm_palette[algo], linestyle='-', linewidth=2.0, marker=mk, markersize=5, zorder=5, clip_on=True)
        if show_std_errorbar and len(xs_mean[j]) > 0:
            xe, ye, ee = ([], [], [])
            for x, y, e in zip(xs_mean[j], ys_mean[j], yerr_per_j[j]):
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(e) and (e >= 0):
                    xe.append(x)
                    ye.append(y)
                    ee.append(e)
            if xe:
                ax.errorbar(xe, ye, yerr=ee, fmt='none', ecolor='black', elinewidth=1.0, capsize=3, capthick=1.0, zorder=6)
    ax.set_xticks([i * group_pitch for i in range(len(pack_sizes))])
    ax.set_xticklabels(exponent_labels)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize, x=0.42)
    ax.tick_params(labelsize=fontsize)
    if yscale == 'log':
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)

def plot_grouped_bars_multi_algo(ax, pack_sizes, algorithm_keys, raw_per_algo_per_pack, algorithm_palette, exponent_labels, fontsize, ylabel, title, xlabel='Page Size $n$', ylim=None, yscale='linear', show_std_errorbar=False, show_bars=True, line_only_algos=None):
    line_only = set(line_only_algos or ())
    n_slot = len(algorithm_keys)
    bw = 0.14
    gap = 0.06
    step = bw + gap
    spread = (n_slot - 1) * step if n_slot > 1 else 0.0
    offs = np.linspace(-spread / 2, spread / 2, n_slot) if n_slot > 1 else np.array([0.0])
    inter_group = 0.48
    group_pitch = spread + bw + inter_group
    xs_mean = [[] for _ in range(n_slot)]
    ys_mean = [[] for _ in range(n_slot)]
    yerr_per_j = [[] for _ in range(n_slot)]
    for i, size in enumerate(pack_sizes):
        center = i * group_pitch
        for j, algo in enumerate(algorithm_keys):
            vals_map = _as_value_map(raw_per_algo_per_pack[algo].get(size, {}))
            vals = list(vals_map.values())
            if len(vals) < 1:
                vals = [0.0]
                vals_map = {0: 0.0}
            pos = center + offs[j]
            m = trimmed_mean(vals, log_scale=yscale == 'log')
            if yscale == 'log' and (not np.isfinite(m) or m <= 0):
                ys_mean[j].append(np.nan)
            elif not np.isfinite(m):
                ys_mean[j].append(np.nan)
            else:
                ys_mean[j].append(m)
            xs_mean[j].append(pos)
            if show_std_errorbar:
                yerr_per_j[j].append(yerr_std(vals, m if np.isfinite(m) else None, log_scale=yscale == 'log'))
            else:
                yerr_per_j[j].append(0.0)
            draw_bar = show_bars and algo not in line_only and np.isfinite(m) and (yscale != 'log' or m > 0)
            if draw_bar:
                ax.bar(pos, m, width=bw * 0.92, color=algorithm_palette[algo], alpha=0.88, edgecolor='none', align='center', zorder=4)
    for j, algo in enumerate(algorithm_keys):
        if len(xs_mean[j]) == 0:
            continue
        mk = algorithm_markers.get(algo, 'o')
        ax.plot(xs_mean[j], ys_mean[j], color=algorithm_palette[algo], linestyle='-', linewidth=2.0, marker=mk, markersize=5, zorder=6, clip_on=True)
        if show_std_errorbar and len(xs_mean[j]) > 0:
            xe, ye, ee = ([], [], [])
            for x, y, e in zip(xs_mean[j], ys_mean[j], yerr_per_j[j]):
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(e) and (e >= 0):
                    xe.append(x)
                    ye.append(y)
                    ee.append(e)
            if xe:
                ax.errorbar(xe, ye, yerr=ee, fmt='none', ecolor='black', elinewidth=1.0, capsize=3, capthick=1.0, zorder=7)
    ax.set_xticks([i * group_pitch for i in range(len(pack_sizes))])
    ax.set_xticklabels(exponent_labels)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize, x=0.42)
    ax.tick_params(labelsize=fontsize)
    if yscale == 'log':
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)

def ylim_compression_page(algos, raw_dict, pack_sizes, ymax_cap=None):
    v = []
    for algo in algos:
        for s in pack_sizes:
            blob = raw_dict[algo].get(s, {})
            if isinstance(blob, dict):
                v.extend([float(x) for x in blob.values() if np.isfinite(float(x))])
            else:
                v.extend([float(x) for x in blob if np.isfinite(float(x))])
    if len(v) < 2:
        return None
    lo, hi = np.percentile(v, [20, 80])
    lo, hi = (float(max(0.0, lo)), float(hi))
    if ymax_cap is not None:
        hi = min(hi, float(ymax_cap))
    if hi <= lo:
        lo = max(0.0, hi - 0.5)
    return (lo, hi)

def ylim_encode_log_page(algos, raw_dict, pack_sizes, pad=0.12):
    v = []
    for algo in algos:
        for s in pack_sizes:
            blob = raw_dict[algo].get(s, {})
            seq = blob.values() if isinstance(blob, dict) else blob
            v.extend(_positive_finite(seq))
    if not v:
        return (10.0, 2000.0)
    lo, hi = (min(v), max(v))
    return (lo * (1 - pad), hi * (1 + pad))
def _algo_has_vary_page_data(algo: str, raw: dict, sizes: list[int], min_datasets: int = 1) -> bool:
    for s in sizes:
        if len(raw[algo].get(s, {})) < min_datasets:
            return False
    return True


bp_group_all = ['BP', 'BP-All', 'BP-Prune', 'BP-Prune-RMQ']
sprintz_group_all = ['Sprintz', 'Sprintz-All', 'Sprintz-Prune', 'Sprintz-Prune-RMQ']
bp_group = [a for a in bp_group_all if _algo_has_vary_page_data(a, compression_ratio_data, vector_sizes)]
sprintz_group = [a for a in sprintz_group_all if _algo_has_vary_page_data(a, compression_ratio_data, vector_sizes)]
if not bp_group:
    bp_group = bp_group_all
if not sprintz_group:
    sprintz_group = sprintz_group_all
fig, axs = plt.subplots(4, 2, figsize=(14, 24))
plt.subplots_adjust(wspace=0.065 * 4 * (2 / 3) + 0.05, hspace=0.38)
fontsize = 22
exponents = [int(np.log2(ps)) for ps in vector_sizes]
exponent_labels = [f'$2^{{{exp}}}$' for exp in exponents]
plt.rcParams.update({'font.size': fontsize})
_ylim_a = ylim_compression_bars_with_errorbar(bp_group, vector_sizes, compression_ratio_data, show_std_errorbar=False) or (4.8, 5.4)
plot_grouped_bars_multi_algo(axs[0, 0], vector_sizes, bp_group, compression_ratio_data, algorithm_palette, exponent_labels, fontsize, 'Compression ratio', '(a) BP: Compression Ratio', ylim=_ylim_a, show_bars=False)
_ylim_c = ylim_encode_log_page(bp_group, encode_time_data, vector_sizes)
plot_grouped_boxes_multi_algo(axs[1, 0], vector_sizes, bp_group, encode_time_data, algorithm_palette, exponent_labels, fontsize, 'Time (ns/point)', '(c) BP: Compression Time', ylim=_ylim_c, yscale='log', show_boxes=False, stagger_algorithms=False)
plot_grouped_boxes_multi_algo(axs[2, 0], vector_sizes, bp_group, decode_time_data, algorithm_palette, exponent_labels, fontsize, 'Time (ns/point)', '(e) BP: Decompression Time', ylim=(0, 20), show_boxes=False, stagger_algorithms=False)
_ylim_b = ylim_compression_bars_with_errorbar(sprintz_group, vector_sizes, compression_ratio_data, show_std_errorbar=False) or (6.2, 6.8)
plot_grouped_bars_multi_algo(axs[0, 1], vector_sizes, sprintz_group, compression_ratio_data, algorithm_palette, exponent_labels, fontsize, 'Compression ratio', '(b) Sprintz: Compression Ratio', ylim=_ylim_b, show_bars=False)
_ylim_d = ylim_encode_log_page(sprintz_group, encode_time_data, vector_sizes)
plot_grouped_boxes_multi_algo(axs[1, 1], vector_sizes, sprintz_group, encode_time_data, algorithm_palette, exponent_labels, fontsize, 'Time (ns/point)', '(d) Sprintz: Compression Time', ylim=_ylim_d, yscale='log', show_boxes=False, stagger_algorithms=False)
plot_grouped_boxes_multi_algo(axs[2, 1], vector_sizes, sprintz_group, decode_time_data, algorithm_palette, exponent_labels, fontsize, 'Time (ns/point)', '(f) Sprintz: Decompression Time', ylim=(0, 20), show_boxes=False, stagger_algorithms=False)
print(
    f'Proposition prune rates from {ICDE_ROOT / "TestData"} '
    f'({len(FIG13_DATASET_FILES)} paper-table datasets)'
)
pruning_rate_data = build_vary_page_prune_rate_data(
    ICDE_ROOT / 'TestData',
    vector_sizes,
    dataset_names=FIG13_DATASET_FILES,
)
plot_grouped_boxes_multi_algo(axs[3, 0], vector_sizes, ['BP-Prune'], pruning_rate_data, algorithm_palette, exponent_labels, fontsize, 'Percentage (% of page size)', '(g) BP: Pruning Rate', ylim=(0, 105), show_boxes=False)
plot_grouped_boxes_multi_algo(axs[3, 1], vector_sizes, ['Sprintz-Prune'], pruning_rate_data, algorithm_palette, exponent_labels, fontsize, 'Percentage (% of page size)', '(h) Sprintz: Pruning Rate', ylim=(0, 105), show_boxes=False)
_legend_algos = bp_group + [a for a in sprintz_group if a not in bp_group]
legend_handles = [Line2D([0], [0], color=algorithm_palette[a], linestyle='-', linewidth=2.0, marker=algorithm_markers[a], markersize=5, label=a.replace('-', '–')) for a in _legend_algos]
fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc='upper center', ncol=4, labelspacing=0.15, handletextpad=0.35, columnspacing=0.9, fontsize=fontsize, bbox_to_anchor=(0.5, 0.95))
output_dir = str(FIGURE_DIR)
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'bp_vary_page_size.png'), dpi=400, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'bp_vary_page_size.eps'), format='eps', dpi=400, bbox_inches='tight')
