import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
data_dirs = {'BP': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_BP_vary_page_size', 'Sprintz': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_sprintz_vary_page_size', 'BP-All': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_BP_vary_page_size_N2', 'Sprintz-All': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_Sprintz_vary_page_size_N2', 'BP-Prune': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_BP_only_Prune_vary_page_size', 'Sprintz-Prune': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_Sprintz_only_Prune_vary_page_size', 'BP-Prune-RMQ': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_BP_Prune_RMQ_vary_page_size', 'Sprintz-Prune-RMQ': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_Sprintz_Prune_RMQ_vary_page_size'}
dataset_mapping = {'City-temp.csv': 'CT', 'Wind-Speed.csv': 'WS', 'IR-bio-temp.csv': 'IR', 'PM10-dust.csv': 'PM10', 'Air-pressure.csv': 'AP', 'Dew-point-temp.csv': 'DT', 'Stocks-UK.csv': 'SUK', 'Stocks-USA.csv': 'SUA', 'Stocks-DE.csv': 'SDE', 'Bird-migration.csv': 'BM', 'Food-price.csv': 'FP', 'Blockchain-tr.csv': 'BTR', 'City-lat.csv': 'CLT', 'City-lon.csv': 'CLN'}
vector_sizes = [16 * 8, 32 * 8, 64 * 8, 128 * 8, 256 * 8, 512 * 8, 1024 * 8]
compression_ratio_data = {algo: {size: {} for size in vector_sizes} for algo in data_dirs.keys()}
encode_time_data = {algo: {size: {} for size in vector_sizes} for algo in data_dirs.keys()}
decode_time_data = {algo: {size: {} for size in vector_sizes} for algo in data_dirs.keys()}
for algorithm, data_dir in data_dirs.items():
    print(f'Processing algorithm: {algorithm}')
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv') or filename == '.DS_Store' or (not filename in dataset_mapping):
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
                        compression_ratio_data[algorithm][pack_size][filename] = 1 / compression_ratio
                        encode_time = float(row['Encoding Time'])
                        encode_time_data[algorithm][pack_size][filename] = 1 / (encode_time / 8000)
                        decode_time = float(row['Decoding Time'])
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
            avg_compression_ratio[algorithm].append(0)
            avg_encode_time[algorithm].append(0)
            avg_decode_time[algorithm].append(0)
            std_compression_ratio[algorithm].append(0)
            std_encode_time[algorithm].append(0)
            std_decode_time[algorithm].append(0)
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

def yerr_closest_cr(cr_by_ds, y_by_ds, k=3):
    cr_map = _as_value_map(cr_by_ds)
    y_map = _as_value_map(y_by_ds)
    common = [d for d in cr_map if d in y_map]
    if len(common) < 2:
        return 0.0
    cr_sub = {d: float(cr_map[d]) for d in common if np.isfinite(float(cr_map[d]))}
    if len(cr_sub) < 2:
        return 0.0
    mean_cr = float(np.mean(list(cr_sub.values())))
    ds_sorted = sorted(cr_sub.keys(), key=lambda d: abs(cr_sub[d] - mean_cr))
    take = min(k, len(ds_sorted))
    chosen = ds_sorted[:take]
    ys = [float(y_map[d]) for d in chosen if d in y_map and np.isfinite(float(y_map[d]))]
    if len(ys) < 2:
        return 0.0
    return float(np.std(ys, ddof=1))

def ylim_compression_bars_with_errorbar(algorithm_keys, pack_sizes, raw_per_algo_per_pack, compression_cr_raw, pad_frac=0.04, pad_min=0.02):
    lo, hi = (np.inf, -np.inf)
    for size in pack_sizes:
        for algo in algorithm_keys:
            vals_map = _as_value_map(raw_per_algo_per_pack[algo].get(size, {}))
            vals = list(vals_map.values())
            if not vals:
                continue
            m = float(np.mean(vals))
            if not np.isfinite(m):
                continue
            cr_map = compression_cr_raw[algo].get(size, {}) if compression_cr_raw else {}
            e = float(yerr_closest_cr(cr_map, vals_map)) if compression_cr_raw else 0.0
            if not np.isfinite(e) or e < 0:
                e = 0.0
            lo = min(lo, m - e)
            hi = max(hi, m + e)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None
    span = hi - lo
    pad = max(span * pad_frac, pad_min)
    return (float(lo - pad), float(hi + pad))

def plot_grouped_boxes_multi_algo(ax, pack_sizes, algorithm_keys, raw_per_algo_per_pack, algorithm_palette, exponent_labels, fontsize, ylabel, title, xlabel='Page Size $n$', ylim=None, yscale='linear', show_boxes=True, stagger_algorithms=True, compression_cr_raw=None):
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
            m = float(np.mean(vals))
            if yscale == 'log' and (not np.isfinite(m) or m <= 0):
                m = np.nan
            xs_mean[j].append(pos)
            ys_mean[j].append(m)
            if compression_cr_raw is not None:
                cr_map = compression_cr_raw[algo].get(size, {})
                yerr_per_j[j].append(yerr_closest_cr(cr_map, vals_map))
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
        if compression_cr_raw is not None and len(xs_mean[j]) > 0:
            if yscale == 'log':
                yerr_plot = []
                for y, e in zip(ys_mean[j], yerr_per_j[j]):
                    if not (np.isfinite(y) and y > 0):
                        yerr_plot.append(0.0)
                    elif not np.isfinite(e):
                        yerr_plot.append(0.0)
                    else:
                        yerr_plot.append(min(float(e), 0.99 * float(y)))
            else:
                yerr_plot = list(yerr_per_j[j])
            xe, ye, ee = ([], [], [])
            for x, y, e in zip(xs_mean[j], ys_mean[j], yerr_plot):
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

def plot_grouped_bars_multi_algo(ax, pack_sizes, algorithm_keys, raw_per_algo_per_pack, algorithm_palette, exponent_labels, fontsize, ylabel, title, xlabel='Page Size $n$', ylim=None, yscale='linear', compression_cr_raw=None):
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
            m = float(np.mean(vals))
            if yscale == 'log' and (not np.isfinite(m) or m <= 0):
                ys_mean[j].append(np.nan)
            elif not np.isfinite(m):
                ys_mean[j].append(np.nan)
            else:
                ys_mean[j].append(m)
            xs_mean[j].append(pos)
            if compression_cr_raw is not None:
                cr_map = compression_cr_raw[algo].get(size, {})
                yerr_per_j[j].append(yerr_closest_cr(cr_map, vals_map))
            else:
                yerr_per_j[j].append(0.0)
            draw_bar = np.isfinite(m) and (yscale != 'log' or m > 0)
            if draw_bar:
                ax.bar(pos, m, width=bw * 0.92, color=algorithm_palette[algo], alpha=0.88, edgecolor='none', align='center', zorder=4)
    for j, algo in enumerate(algorithm_keys):
        if len(xs_mean[j]) == 0:
            continue
        mk = algorithm_markers.get(algo, 'o')
        ax.plot(xs_mean[j], ys_mean[j], color=algorithm_palette[algo], linestyle='-', linewidth=2.0, marker=mk, markersize=5, zorder=6, clip_on=True)
        if compression_cr_raw is not None and len(xs_mean[j]) > 0:
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
bp_group = ['BP', 'BP-All', 'BP-Prune', 'BP-Prune-RMQ']
sprintz_group = ['Sprintz', 'Sprintz-All', 'Sprintz-Prune', 'Sprintz-Prune-RMQ']
fig, axs = plt.subplots(4, 2, figsize=(14, 20))
plt.subplots_adjust(wspace=0.065 * 4 * (2 / 3) + 0.05, hspace=0.38)
fontsize = 22
exponents = [int(np.log2(ps)) for ps in vector_sizes]
exponent_labels = [f'$2^{{{exp}}}$' for exp in exponents]
plt.rcParams.update({'font.size': fontsize})
_ylim_a = ylim_compression_bars_with_errorbar(bp_group, vector_sizes, compression_ratio_data, compression_ratio_data) or (4.8, 5.4)
plot_grouped_bars_multi_algo(axs[0, 0], vector_sizes, bp_group, compression_ratio_data, algorithm_palette, exponent_labels, fontsize, 'Compression ratio', '(a) BP: Compression Ratio', ylim=_ylim_a, compression_cr_raw=compression_ratio_data)
_ylim_c = ylim_encode_log_page(bp_group, encode_time_data, vector_sizes)
plot_grouped_boxes_multi_algo(axs[1, 0], vector_sizes, bp_group, encode_time_data, algorithm_palette, exponent_labels, fontsize, 'Time (ns/point)', '(c) BP: Compression Time', ylim=_ylim_c, yscale='log', show_boxes=False, stagger_algorithms=False, compression_cr_raw=compression_ratio_data)
plot_grouped_boxes_multi_algo(axs[2, 0], vector_sizes, bp_group, decode_time_data, algorithm_palette, exponent_labels, fontsize, 'Time (ns/point)', '(e) BP: Decompression Time', ylim=(0, 30), show_boxes=False, stagger_algorithms=False, compression_cr_raw=compression_ratio_data)
_ylim_b = ylim_compression_bars_with_errorbar(sprintz_group, vector_sizes, compression_ratio_data, compression_ratio_data) or (6.2, 6.8)
plot_grouped_bars_multi_algo(axs[0, 1], vector_sizes, sprintz_group, compression_ratio_data, algorithm_palette, exponent_labels, fontsize, 'Compression ratio', '(b) Sprintz: Compression Ratio', ylim=_ylim_b, compression_cr_raw=compression_ratio_data)
_ylim_d = ylim_encode_log_page(sprintz_group, encode_time_data, vector_sizes)
plot_grouped_boxes_multi_algo(axs[1, 1], vector_sizes, sprintz_group, encode_time_data, algorithm_palette, exponent_labels, fontsize, 'Time (ns/point)', '(d) Sprintz: Compression Time', ylim=_ylim_d, yscale='log', show_boxes=False, stagger_algorithms=False, compression_cr_raw=compression_ratio_data)
plot_grouped_boxes_multi_algo(axs[2, 1], vector_sizes, sprintz_group, decode_time_data, algorithm_palette, exponent_labels, fontsize, 'Time (ns/point)', '(f) Sprintz: Decompression Time', ylim=(0, 30), show_boxes=False, stagger_algorithms=False, compression_cr_raw=compression_ratio_data)
prune_data_dirs = {'BP-Prune': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_BP_filters_plus_vary_page_size', 'Sprintz-Prune': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_Sprintz_filters_plus_vary_page_size'}
pruning_rate_data = {algo: {size: {} for size in vector_sizes} for algo in prune_data_dirs.keys()}
for algorithm, data_dir in prune_data_dirs.items():
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv') or filename == '.DS_Store' or (not filename in dataset_mapping):
            continue
        dataset_name = dataset_mapping.get(filename, filename)
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    pack_size = row['Page size']
                    try:
                        pack_size = int(pack_size)
                    except:
                        continue
                    if pack_size in vector_sizes:
                        filter_count = float(row['Filter Count'])
                        page_size = float(row['Page size'])
                        pruning_rate_data[algorithm][pack_size][filename] = filter_count / page_size * 100
            except Exception as e:
                print(f'    Error processing {file_path}: {e}')
                continue
plot_grouped_boxes_multi_algo(axs[3, 0], vector_sizes, ['BP-Prune'], pruning_rate_data, algorithm_palette, exponent_labels, fontsize, 'Percentage (% of page size)', '(g) BP: Pruning Rate', ylim=(0, 105), show_boxes=False, compression_cr_raw=compression_ratio_data)
plot_grouped_boxes_multi_algo(axs[3, 1], vector_sizes, ['Sprintz-Prune'], pruning_rate_data, algorithm_palette, exponent_labels, fontsize, 'Percentage (% of page size)', '(h) Sprintz: Pruning Rate', ylim=(0, 105), show_boxes=False, compression_cr_raw=compression_ratio_data)
legend_handles = [Line2D([0], [0], color=algorithm_palette['BP'], linestyle='-', linewidth=2.0, marker=algorithm_markers['BP'], markersize=5, label='BP'), Line2D([0], [0], color=algorithm_palette['BP-All'], linestyle='-', linewidth=2.0, marker=algorithm_markers['BP-All'], markersize=5, label='BP–All'), Line2D([0], [0], color=algorithm_palette['BP-Prune'], linestyle='-', linewidth=2.0, marker=algorithm_markers['BP-Prune'], markersize=5, label='BP–Prune'), Line2D([0], [0], color=algorithm_palette['BP-Prune-RMQ'], linestyle='-', linewidth=2.0, marker=algorithm_markers['BP-Prune-RMQ'], markersize=5, label='BP–Prune–RMQ'), Line2D([0], [0], color=algorithm_palette['Sprintz'], linestyle='-', linewidth=2.0, marker=algorithm_markers['Sprintz'], markersize=5, label='Sprintz'), Line2D([0], [0], color=algorithm_palette['Sprintz-All'], linestyle='-', linewidth=2.0, marker=algorithm_markers['Sprintz-All'], markersize=5, label='Sprintz–All'), Line2D([0], [0], color=algorithm_palette['Sprintz-Prune'], linestyle='-', linewidth=2.0, marker=algorithm_markers['Sprintz-Prune'], markersize=5, label='Sprintz–Prune'), Line2D([0], [0], color=algorithm_palette['Sprintz-Prune-RMQ'], linestyle='-', linewidth=2.0, marker=algorithm_markers['Sprintz-Prune-RMQ'], markersize=5, label='Sprintz–Prune–RMQ')]
fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc='upper center', ncol=4, labelspacing=0.15, handletextpad=0.35, columnspacing=0.9, fontsize=fontsize, bbox_to_anchor=(0.5, 0.95))
output_dir = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/figure_for_paper'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'bp_vary_page_size.png'), dpi=400, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'bp_vary_page_size.eps'), format='eps', dpi=400, bbox_inches='tight')
