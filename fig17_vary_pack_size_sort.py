import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size'
OUT_DIR = os.path.join(BASE, 'figure_for_paper')
NO_SORT_RATIO_PATH = os.path.join(BASE, 'camel_ratio.xlsx')
NO_SORT_COMPRESSION_TIME_PATH = os.path.join(BASE, 'compression_time.xlsx')
NO_SORT_DECOMPRESSION_TIME_PATH = os.path.join(BASE, 'decompression_time.xlsx')
SORT_RATIO_PATH = os.path.join(BASE, 'sort_camel_ratio.xlsx')
SORT_COMPRESSION_TIME_PATH = os.path.join(BASE, 'sort_compression_time.xlsx')
SORT_DECOMPRESSION_TIME_PATH = os.path.join(BASE, 'sort_decompression_time.xlsx')
ALGO_PAIRS = [('BP (All)', 'Sort-BP (All)'), ('BP (Prune-RMQ)', 'Sort-BP (Prune-RMQ)')]
ALGO_LEGEND_LABELS = [('BP-All', 'BP-All (Sort)'), ('BP-Prune-RMQ', 'BP-Prune-RMQ (Sort)')]
GROUP_LABELS = [a[0] for a in ALGO_LEGEND_LABELS]
GROUP_COLORS = ['#9467bd', '#d62728']
ERRORBAR_ECOLOR = 'k'
ERRORBAR_CAPSIZE = 2.8
ERR_KW_BAR = dict(elinewidth=1.1, capthick=1.0, alpha=1.0, zorder=5)
RATIO_Y_MARGIN_FRAC = 0.05

def _bar_with_err(ax, x0, h, w, yerr, **kwargs):
    kw = dict(x=x0, height=h, width=w, zorder=3, **kwargs)
    if yerr is not None:
        kw['yerr'] = yerr
        kw['ecolor'] = ERRORBAR_ECOLOR
        kw['capsize'] = ERRORBAR_CAPSIZE
        kw['error_kw'] = ERR_KW_BAR
    ax.bar(**kw)

def _yerr_for_bar(v: float, std: float, log_scale: bool):
    if std <= 0 or not np.isfinite(v) or (not np.isfinite(std)):
        return None
    if not log_scale:
        return std
    if v <= 0:
        return None
    el = float(min(std, v - 1e-12))
    if el <= 0:
        return None
    return np.array([[el], [std]])
dataset_mapping = {'Food-price.csv': 'FP', 'electric_vehicle_charging.csv': 'VC', 'Blockchain-tr.csv': 'BTR', 'SSD-bench.csv': 'SB', 'City-lat.csv': 'CLT', 'City-lon.csv': 'CLN'}
VALID_DATASETS = set(dataset_mapping.values())

def load_ratio_stats_per_algo(path: str, algo_list: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    mean_out: dict[str, float] = {}
    std_out: dict[str, float] = {}
    if not os.path.exists(path):
        return (mean_out, std_out)
    df = pd.read_excel(path)
    valid_cols = [c for c in df.columns[1:] if c in VALID_DATASETS]
    for algo in algo_list:
        row = df[df.iloc[:, 0] == algo]
        if row.empty:
            mean_out[algo] = np.nan
            std_out[algo] = 0.0
            continue
        vals = []
        for col in valid_cols:
            v = row[col].iloc[0]
            if pd.notna(v):
                try:
                    vals.append(1.0 / float(v))
                except Exception:
                    pass
        if vals:
            mean_out[algo] = float(np.mean(vals))
            std_out[algo] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        else:
            mean_out[algo] = np.nan
            std_out[algo] = 0.0
    return (mean_out, std_out)

def load_time_stats_per_algo(path: str, algo_list: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    mean_out: dict[str, float] = {}
    std_out: dict[str, float] = {}
    if not os.path.exists(path):
        return (mean_out, std_out)
    df = pd.read_excel(path)
    valid_cols = [c for c in df.columns[1:] if c in VALID_DATASETS]
    for algo in algo_list:
        row = df[df.iloc[:, 0] == algo]
        if row.empty:
            mean_out[algo] = np.nan
            std_out[algo] = 0.0
            continue
        vals = []
        for col in valid_cols:
            v = row[col].iloc[0]
            if pd.notna(v):
                try:
                    vals.append(1.0 / (float(v) / 8000.0))
                except Exception:
                    pass
        if vals:
            mean_out[algo] = float(np.mean(vals))
            std_out[algo] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        else:
            mean_out[algo] = 0.0
            std_out[algo] = 0.0
    return (mean_out, std_out)

def plot_compare_sort():
    all_algos = [a for pair in ALGO_PAIRS for a in pair]
    ratio_no, ratio_no_sd = load_ratio_stats_per_algo(NO_SORT_RATIO_PATH, all_algos)
    ratio_sort, ratio_sort_sd = load_ratio_stats_per_algo(SORT_RATIO_PATH, all_algos)
    enc_no, enc_no_sd = load_time_stats_per_algo(NO_SORT_COMPRESSION_TIME_PATH, all_algos)
    enc_sort, enc_sort_sd = load_time_stats_per_algo(SORT_COMPRESSION_TIME_PATH, all_algos)
    dec_no, dec_no_sd = load_time_stats_per_algo(NO_SORT_DECOMPRESSION_TIME_PATH, all_algos)
    dec_sort, dec_sort_sd = load_time_stats_per_algo(SORT_DECOMPRESSION_TIME_PATH, all_algos)
    print('\nCompression time mean (ns/point) across datasets:')
    for (no_name, s_name), (leg_no, leg_s) in zip(ALGO_PAIRS, ALGO_LEGEND_LABELS):
        v_no = enc_no.get(no_name, np.nan)
        v_s = enc_sort.get(s_name, np.nan)
        print(f'  {leg_no}: {v_no:.6g}')
        print(f'  {leg_s}: {v_s:.6g}')
    n_groups = len(ALGO_PAIRS)
    x = np.arange(n_groups)
    width = 0.35
    fontsize = 22
    plt.rcParams.update({'font.size': fontsize})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.35)
    ratio_ylim_hi = 3.0
    for ax_idx, (ax, vals_no, vals_s, sd_no, sd_s) in enumerate([(axes[0], [ratio_no.get(no_name, np.nan) for no_name, _ in ALGO_PAIRS], [ratio_sort.get(s_name, np.nan) for _, s_name in ALGO_PAIRS], [ratio_no_sd.get(no_name, 0.0) for no_name, _ in ALGO_PAIRS], [ratio_sort_sd.get(s_name, 0.0) for _, s_name in ALGO_PAIRS]), (axes[1], [enc_no.get(no_name, np.nan) for no_name, _ in ALGO_PAIRS], [enc_sort.get(s_name, np.nan) for _, s_name in ALGO_PAIRS], [enc_no_sd.get(no_name, 0.0) for no_name, _ in ALGO_PAIRS], [enc_sort_sd.get(s_name, 0.0) for _, s_name in ALGO_PAIRS]), (axes[2], [dec_no.get(no_name, np.nan) for no_name, _ in ALGO_PAIRS], [dec_sort.get(s_name, np.nan) for _, s_name in ALGO_PAIRS], [dec_no_sd.get(no_name, 0.0) for no_name, _ in ALGO_PAIRS], [dec_sort_sd.get(s_name, 0.0) for _, s_name in ALGO_PAIRS])]):
        log_scale = ax_idx == 1
        for i in range(n_groups):
            no_name, s_name = (ALGO_PAIRS[i][0], ALGO_PAIRS[i][1])
            leg_no, leg_s = ALGO_LEGEND_LABELS[i]
            c = GROUP_COLORS[i]
            v_no_raw = vals_no[i]
            v_s_raw = vals_s[i]
            v_no = float(np.nan_to_num(v_no_raw, nan=0.0))
            v_s = float(np.nan_to_num(v_s_raw, nan=0.0))
            std_no = float(sd_no[i])
            std_s = float(sd_s[i])
            if ax_idx == 1 and (v_no <= 0.0 or v_s <= 0.0):
                v_no = max(v_no, 1e-10)
                v_s = max(v_s, 1e-10)
            yerr_no = _yerr_for_bar(v_no, std_no, log_scale)
            yerr_s = _yerr_for_bar(v_s, std_s, log_scale)
            if ax_idx == 0:
                if np.isfinite(v_no_raw):
                    ratio_ylim_hi = max(ratio_ylim_hi, float(v_no_raw) + std_no)
                if np.isfinite(v_s_raw):
                    ratio_ylim_hi = max(ratio_ylim_hi, float(v_s_raw) + std_s)
            _bar_with_err(ax, x[i] - width / 2, v_no, width, yerr_no, color=c, edgecolor='none', linewidth=0, label=leg_no if ax_idx == 0 else None)
            _bar_with_err(ax, x[i] + width / 2, v_s, width, yerr_s, color=c, edgecolor='white', linewidth=0.8, hatch='///', label=leg_s if ax_idx == 0 else None)
    axes[0].set_ylabel('Compression Ratio', fontsize=fontsize)
    axes[0].set_title('(a) Compression Ratio', fontsize=fontsize)
    axes[0].set_xlabel('')
    y0_lo = 3.0
    span = max(ratio_ylim_hi - y0_lo, 1e-06)
    margin = max(span * RATIO_Y_MARGIN_FRAC, 0.02 * max(abs(ratio_ylim_hi), 1.0))
    axes[0].set_ylim(y0_lo, ratio_ylim_hi + margin)
    axes[1].set_ylabel('Time (ns/point)', fontsize=fontsize)
    axes[1].set_title('(b) Compression Time', fontsize=fontsize)
    axes[1].set_xlabel('')
    axes[1].set_yscale('log')
    axes[1].set_ylim(50, 2000)
    axes[2].set_ylabel('Time (ns/point)', fontsize=fontsize)
    axes[2].set_title('(c) Decompression Time', fontsize=fontsize)
    axes[2].set_xlabel('')
    axes[2].set_ylim(bottom=10)
    for ax in axes:
        ax.set_xticks([])
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, labelspacing=0.1, handletextpad=0.1, columnspacing=0.1, fontsize=fontsize - 1, bbox_to_anchor=(0.5, 0.98))
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, 'sort_vs_no_sort_compare.png')
    out_eps = os.path.join(OUT_DIR, 'sort_vs_no_sort_compare.eps')
    plt.savefig(out_png, dpi=400, bbox_inches='tight')
    plt.savefig(out_eps, format='eps', dpi=400, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_png}')
if __name__ == '__main__':
    plot_compare_sort()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.axes import Axes
import matplotlib.path as mpath
from collections import OrderedDict
data_dirs = {'BP': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_BP_vary_pack_size', 'Sort-BP': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_BP_vary_pack_size_sort', 'Sprintz': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_Sprintz_vary_pack_size', 'Sort-Sprintz': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_Sprintz_vary_pack_size_sort'}
dataset_mapping = {'Food-price.csv': 'FP', 'electric_vehicle_charging.csv': 'VC', 'Blockchain-tr.csv': 'BTR', 'SSD-bench.csv': 'SB', 'City-lat.csv': 'CLT', 'City-lon.csv': 'CLN'}
vector_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
compression_ratio_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}
encode_time_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}
decode_time_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}
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
                    pack_size = row['Pack size']
                    try:
                        pack_size = int(pack_size)
                    except:
                        continue
                    if pack_size in vector_sizes:
                        compression_ratio = float(row['Compression Ratio'])
                        compression_ratio_data[algorithm][pack_size].append(1 / compression_ratio)
                        encode_time = float(row['Encoding Time'])
                        encode_time_data[algorithm][pack_size].append(1 / (encode_time / 8000))
                        decode_time = float(row['Decoding Time'])
                        decode_time_data[algorithm][pack_size].append(1 / (decode_time / 8000))
            except Exception as e:
                print(f'    Error processing {file_path}: {e}')
                continue
avg_compression_ratio = {}
avg_encode_time = {}
avg_decode_time = {}
for algorithm in data_dirs.keys():
    avg_compression_ratio[algorithm] = []
    avg_encode_time[algorithm] = []
    avg_decode_time[algorithm] = []
    for size in vector_sizes:
        if compression_ratio_data[algorithm][size]:
            avg_cr = np.mean(compression_ratio_data[algorithm][size])
            avg_compression_ratio[algorithm].append(avg_cr)
            avg_et = np.mean(encode_time_data[algorithm][size])
            avg_encode_time[algorithm].append(avg_et)
            avg_dt = np.mean(decode_time_data[algorithm][size])
            avg_decode_time[algorithm].append(avg_dt)
        else:
            avg_compression_ratio[algorithm].append(0)
            avg_encode_time[algorithm].append(0)
            avg_decode_time[algorithm].append(0)
print('\nMean compression ratio:')
for size, ratio in zip(vector_sizes, avg_compression_ratio['BP']):
    print(f'  Pack size {size}: {ratio:.4f}')
bp_rmq_mean = None
bp_all_mean = None
sprintz_rmq_mean = None
sprintz_all_mean = None
camel_ratio_path = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/sort_camel_ratio.xlsx'
if os.path.exists(camel_ratio_path):
    print(f'\nReading camel_ratio.xlsx: {camel_ratio_path}')
    camel_df = pd.read_excel(camel_ratio_path)
    valid_datasets = set(dataset_mapping.values())
    print(f'Valid dataset short names: {valid_datasets}')
    all_columns = camel_df.columns.tolist()
    print(f'Excel column names: {all_columns}')
    valid_columns = [col for col in all_columns[1:] if col in valid_datasets]
    print(f'Columns used in aggregation: {valid_columns}')
    bp_rmq_row = camel_df[camel_df.iloc[:, 0] == 'BP (Prune-RMQ)']
    if not bp_rmq_row.empty:
        bp_rmq_values = []
        for col in valid_columns:
            if col in bp_rmq_row.columns:
                val = bp_rmq_row[col].iloc[0]
                if pd.notna(val):
                    bp_rmq_values.append(float(val))
        if bp_rmq_values:
            bp_rmq_values = 1 / np.array(bp_rmq_values)
            bp_rmq_mean = np.mean(bp_rmq_values)
            print(f'BP-RMQ mean (over {len(bp_rmq_values)} datasets): {bp_rmq_mean:.4f}')
            print(f'Values: {bp_rmq_values}')
        else:
            print('No valid data in BP-RMQ row')
            bp_rmq_mean = None
    else:
        print('BP-RMQ row not found')
        bp_rmq_mean = None
    bp_learn_row = camel_df[camel_df.iloc[:, 0] == 'Sort-BP (Prune-RMQ)']
    if not bp_learn_row.empty:
        bp_learn_values = []
        for col in valid_columns:
            if col in bp_learn_row.columns:
                val = bp_learn_row[col].iloc[0]
                if pd.notna(val):
                    bp_learn_values.append(float(val))
        if bp_learn_values:
            bp_learn_values = 1 / np.array(bp_learn_values)
            bp_learn_mean = np.mean(bp_learn_values)
            print(f'BP-Learn mean (over {len(bp_learn_values)} datasets): {bp_learn_mean:.4f}')
            print(f'Values: {bp_learn_values}')
        else:
            print('No valid data in BP-Learn row')
            bp_learn_mean = None
    else:
        print('BP-Learn row not found')
        bp_learn_mean = None
    sprintz_rmq_row = camel_df[camel_df.iloc[:, 0] == 'Sprintz (RMQ)']
    if not sprintz_rmq_row.empty:
        sprintz_rmq_values = []
        for col in valid_columns:
            if col in sprintz_rmq_row.columns:
                val = sprintz_rmq_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_rmq_values.append(float(val))
        if sprintz_rmq_values:
            sprintz_rmq_values = 1 / np.array(sprintz_rmq_values)
            sprintz_rmq_mean = np.mean(sprintz_rmq_values)
            print(f'Sprintz-RMQ mean (over {len(sprintz_rmq_values)} datasets): {sprintz_rmq_mean:.4f}')
            print(f'Values: {sprintz_rmq_values}')
        else:
            print('No valid data in Sprintz-RMQ row')
            sprintz_rmq_mean = None
    else:
        print('Sprintz-RMQ row not found')
        sprintz_rmq_mean = None
    sprintz_all_row = camel_df[camel_df.iloc[:, 0] == 'Sort-Sprintz (Prune-RMQ)']
    if not sprintz_all_row.empty:
        sprintz_all_values = []
        for col in valid_columns:
            if col in sprintz_all_row.columns:
                val = sprintz_all_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_all_values.append(float(val))
        if sprintz_all_values:
            sprintz_all_values = 1 / np.array(sprintz_all_values)
            sprintz_all_mean = np.mean(sprintz_all_values)
            print(f'Sprintz-All mean (over {len(sprintz_all_values)} datasets): {sprintz_all_mean:.4f}')
            print(f'Values: {sprintz_all_values}')
        else:
            print('No valid data in Sprintz-All row')
            sprintz_all_mean = None
    else:
        print('Sprintz-All row not found')
        sprintz_all_mean = None
else:
    print(f'\ncamel_ratio.xlsx not found: {camel_ratio_path}')
    bp_rmq_mean = None
    bp_all_mean = None
    sprintz_rmq_mean = None
    sprintz_all_mean = None
camel_encode_path = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/sort_compression_time.xlsx'
camel_decode_path = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/sort_decompression_time.xlsx'
bp_rmq_encode_mean = None
bp_rmq_decode_mean = None
sprintz_rmq_encode_mean = None
sprintz_rmq_decode_mean = None
bp_all_encode_mean = None
bp_all_decode_mean = None
sprintz_all_encode_mean = None
sprintz_all_decode_mean = None
if os.path.exists(camel_encode_path):
    print(f'\nReading camel_encode.xlsx: {camel_encode_path}')
    encode_df = pd.read_excel(camel_encode_path)
    valid_datasets = set(dataset_mapping.values())
    all_columns = encode_df.columns.tolist()
    valid_columns = [col for col in all_columns[1:] if col in valid_datasets]
    bp_rmq_encode_row = encode_df[encode_df.iloc[:, 0] == 'BP (Prune-RMQ)']
    if not bp_rmq_encode_row.empty:
        bp_rmq_encode_values = []
        for col in valid_columns:
            if col in bp_rmq_encode_row.columns:
                val = bp_rmq_encode_row[col].iloc[0]
                if pd.notna(val):
                    bp_rmq_encode_values.append(1 / (float(val) / 8000))
        if bp_rmq_encode_values:
            bp_rmq_encode_mean = np.mean(bp_rmq_encode_values)
            print(f'BP-RMQ mean encode throughput (over {len(bp_rmq_encode_values)} datasets): {bp_rmq_encode_mean:.2f} MB/s')
        else:
            print('No valid encode throughput in BP-RMQ row')
    else:
        print('BP-RMQ encode throughput row not found')
    bp_all_encode_row = encode_df[encode_df.iloc[:, 0] == 'Sort-BP (Prune-RMQ)']
    if not bp_all_encode_row.empty:
        bp_all_encode_values = []
        for col in valid_columns:
            if col in bp_all_encode_row.columns:
                val = bp_all_encode_row[col].iloc[0]
                if pd.notna(val):
                    bp_all_encode_values.append(1 / (float(val) / 8000))
        if bp_all_encode_values:
            bp_all_encode_mean = np.mean(bp_all_encode_values)
            print(f'BP-All mean encode throughput (over {len(bp_all_encode_values)} datasets): {bp_all_encode_mean:.2f} MB/s')
        else:
            print('No valid encode throughput in BP-All row')
    else:
        print('BP-All encode throughput row not found')
    sprintz_rmq_encode_row = encode_df[encode_df.iloc[:, 0] == 'Sprintz (RMQ)']
    if not sprintz_rmq_encode_row.empty:
        sprintz_rmq_encode_values = []
        for col in valid_columns:
            if col in sprintz_rmq_encode_row.columns:
                val = sprintz_rmq_encode_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_rmq_encode_values.append(1 / (float(val) / 8000))
        if sprintz_rmq_encode_values:
            sprintz_rmq_encode_mean = np.mean(sprintz_rmq_encode_values)
            print(f'Sprintz-RMQ mean encode throughput (over {len(sprintz_rmq_encode_values)} datasets): {sprintz_rmq_encode_mean:.2f} MB/s')
        else:
            print('No valid encode throughput in Sprintz-RMQ row')
    else:
        print('Sprintz-RMQ encode throughput row not found')
    sprintz_all_encode_row = encode_df[encode_df.iloc[:, 0] == 'Sort-Sprintz (Prune-RMQ)']
    if not sprintz_all_encode_row.empty:
        sprintz_all_encode_values = []
        for col in valid_columns:
            if col in sprintz_all_encode_row.columns:
                val = sprintz_all_encode_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_all_encode_values.append(1 / (float(val) / 8000))
        if sprintz_all_encode_values:
            sprintz_all_encode_mean = np.mean(sprintz_all_encode_values)
            print(f'Sprintz-All mean encode throughput (over {len(sprintz_all_encode_values)} datasets): {sprintz_all_encode_mean:.2f} MB/s')
        else:
            print('No valid encode throughput in Sprintz-All row')
    else:
        print('Sprintz-All encode throughput row not found')
else:
    print(f'\ncamel_encode.xlsx not found: {camel_encode_path}')
if os.path.exists(camel_decode_path):
    print(f'\nReading camel_decode.xlsx: {camel_decode_path}')
    decode_df = pd.read_excel(camel_decode_path)
    valid_datasets = set(dataset_mapping.values())
    all_columns = decode_df.columns.tolist()
    valid_columns = [col for col in all_columns[1:] if col in valid_datasets]
    bp_rmq_decode_row = decode_df[decode_df.iloc[:, 0] == 'BP (Prune-RMQ)']
    if not bp_rmq_decode_row.empty:
        bp_rmq_decode_values = []
        for col in valid_columns:
            if col in bp_rmq_decode_row.columns:
                val = bp_rmq_decode_row[col].iloc[0]
                if pd.notna(val):
                    bp_rmq_decode_values.append(1 / (float(val) / 8000))
        if bp_rmq_decode_values:
            bp_rmq_decode_mean = np.mean(bp_rmq_decode_values)
            print(f'BP-RMQ mean decode throughput (over {len(bp_rmq_decode_values)} datasets): {bp_rmq_decode_mean:.2f} MB/s')
        else:
            print('No valid decode throughput in BP-RMQ row')
    else:
        print('BP-RMQ decode throughput row not found')
    bp_all_decode_row = decode_df[decode_df.iloc[:, 0] == 'Sort-BP (Prune-RMQ)']
    if not bp_all_decode_row.empty:
        bp_all_decode_values = []
        for col in valid_columns:
            if col in bp_all_decode_row.columns:
                val = bp_all_decode_row[col].iloc[0]
                if pd.notna(val):
                    bp_all_decode_values.append(1 / (float(val) / 8000))
        if bp_all_decode_values:
            bp_all_decode_mean = np.mean(bp_all_decode_values)
            print(f'BP-All mean decode throughput (over {len(bp_all_decode_values)} datasets): {bp_all_decode_mean:.2f} MB/s')
        else:
            print('No valid decode throughput in BP-All row')
    else:
        print('BP-All decode throughput row not found')
    sprintz_rmq_decode_row = decode_df[decode_df.iloc[:, 0] == 'Sprintz (RMQ)']
    if not sprintz_rmq_decode_row.empty:
        sprintz_rmq_decode_values = []
        for col in valid_columns:
            if col in sprintz_rmq_decode_row.columns:
                val = sprintz_rmq_decode_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_rmq_decode_values.append(1 / (float(val) / 8000))
        if sprintz_rmq_decode_values:
            sprintz_rmq_decode_mean = np.mean(sprintz_rmq_decode_values)
            print(f'Sprintz-RMQ mean decode throughput (over {len(sprintz_rmq_decode_values)} datasets): {sprintz_rmq_decode_mean:.2f} MB/s')
        else:
            print('No valid decode throughput in Sprintz-RMQ row')
    else:
        print('Sprintz-RMQ decode throughput row not found')
    sprintz_all_decode_row = decode_df[decode_df.iloc[:, 0] == 'Sort-Sprintz (Prune-RMQ)']
    if not sprintz_all_decode_row.empty:
        sprintz_all_decode_values = []
        for col in valid_columns:
            if col in sprintz_all_decode_row.columns:
                val = sprintz_all_decode_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_all_decode_values.append(1 / (float(val) / 8000))
        if sprintz_all_decode_values:
            sprintz_all_decode_mean = np.mean(sprintz_all_decode_values)
            print(f'Sprintz-All mean decode throughput (over {len(sprintz_all_decode_values)} datasets): {sprintz_all_decode_mean:.2f} MB/s')
        else:
            print('No valid decode throughput in Sprintz-All row')
    else:
        print('Sprintz-All decode throughput row not found')
else:
    print(f'\ncamel_decode.xlsx not found: {camel_decode_path}')
df_compression_ratio = pd.DataFrame(avg_compression_ratio, index=vector_sizes)
df_compression_ratio.index.name = 'Pack Size'
df_encode_time = pd.DataFrame(avg_encode_time, index=vector_sizes)
df_encode_time.index.name = 'Pack Size'
df_decode_time = pd.DataFrame(avg_decode_time, index=vector_sizes)
df_decode_time.index.name = 'Pack Size'
df_compression_ratio_reset = df_compression_ratio.reset_index().melt(id_vars='Pack Size', var_name='Algorithm', value_name='Compression Ratio')
df_encode_time_reset = df_encode_time.reset_index().melt(id_vars='Pack Size', var_name='Algorithm', value_name='Encoding Time (MB/s)')
df_decode_time_reset = df_decode_time.reset_index().melt(id_vars='Pack Size', var_name='Algorithm', value_name='Decoding Time (MB/s)')
heart_vertices = [(0, 0), (0.5, 0.5), (1, 0), (0.5, -0.5), (0, 0), (-0.5, -0.5), (-1, 0), (-0.5, 0.5), (0, 0)]
heart = mpath.Path(heart_vertices)
t = np.linspace(0, 2 * np.pi, 100)
x = 16 * np.sin(t) ** 3
y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
x = x / np.max(np.abs(x))
y = y / np.max(np.abs(y))
heart_parametric = mpath.Path(np.column_stack([x, y]))
trapezoid_vertices = [(-0.8, -1), (0.8, -1), (0.4, 1), (-0.4, 1), (-0.8, -1)]
trapezoid = mpath.Path(trapezoid_vertices)
parallelogram_vertices = [(-1, -0.6), (0.5, -0.6), (1, 0.6), (-0.5, 0.6), (-1, -0.6)]
parallelogram = mpath.Path(parallelogram_vertices)
markers = ['o', '^', parallelogram, heart, 's', 'v', trapezoid, heart_parametric]
algorithm_order = ['BP', 'BP (Prune-RMQ)', 'Sort-BP', 'Sort-BP (Prune-RMQ)', 'Sprintz', 'Sort-Sprintz', 'Sprintz (RMQ)', 'Sort-Sprintz (Prune-RMQ)']
algorithm_palette = {'BP': '#1f77b4', 'Sort-BP': '#9467bd', 'Sprintz': '#9467bd', 'Sort-Sprintz': '#8c564b'}
fig, axs = plt.subplots(3, 2, figsize=(9, 13))
plt.subplots_adjust(wspace=0.45, hspace=0.35)
fontsize = 18
exponents = [int(np.log2(ps)) for ps in vector_sizes]
exponent_labels = [f'$2^{{{exp}}}$' for exp in exponents]
plt.rcParams.update({'font.size': fontsize})
bp_group = ['BP', 'Sort-BP']
sprintz_group = ['Sprintz', 'Sort-Sprintz']
ax1 = axs[0, 0]
for i, algorithm in enumerate(bp_group):
    data = df_compression_ratio_reset[df_compression_ratio_reset['Algorithm'] == algorithm]
    ax1.plot(data['Pack Size'], data['Compression Ratio'], color=algorithm_palette[algorithm], linestyle='-', marker=markers[i], markersize=7, linewidth=2.2, label=algorithm)
if bp_rmq_mean is not None:
    ax1.axhline(y=bp_rmq_mean, color='#d62728', linestyle='--', linewidth=1.8, label='BP (Prune-RMQ)')
if bp_learn_mean is not None:
    ax1.axhline(y=bp_learn_mean, color='#2ca02c', linestyle='--', linewidth=1.8, label='Sort-BP (Prune-RMQ)')
ax1.set_xscale('log', base=2)
ax1.set_ylabel('Compression Ratio', fontsize=fontsize)
ax1.set_xlabel('Pack Size $s$', fontsize=fontsize)
ax1.set_title('(a) BP: Compression Ratio', fontsize=fontsize, x=0.4)
ax1.set_xticks(vector_sizes)
ax1.set_xticklabels(exponent_labels)
ax1.tick_params(labelsize=fontsize)
ax2 = axs[1, 0]
for i, algorithm in enumerate(bp_group):
    data = df_encode_time_reset[df_encode_time_reset['Algorithm'] == algorithm]
    ax2.plot(data['Pack Size'], data['Encoding Time (MB/s)'], color=algorithm_palette[algorithm], linestyle='-', marker=markers[i], markersize=7, linewidth=2.2, label=algorithm)
if bp_rmq_encode_mean is not None:
    ax2.axhline(y=bp_rmq_encode_mean, color='#d62728', linestyle='--', linewidth=1.8, label='BP (Prune-RMQ)')
if bp_all_encode_mean is not None:
    ax2.axhline(y=bp_all_encode_mean, color='#2ca02c', linestyle='--', linewidth=1.8, label='Sort-BP (Prune-RMQ)')
ax2.set_xscale('log', base=2)
ax2.set_ylim(0, 120)
ax2.set_ylabel('Time (ns/point)', fontsize=fontsize)
ax2.set_xlabel('Pack Size $s$', fontsize=fontsize)
ax2.set_title('(c) BP: Compression Time', fontsize=fontsize, x=0.4)
ax2.set_xticks(vector_sizes)
ax2.set_xticklabels(exponent_labels)
ax2.tick_params(labelsize=fontsize)
ax3 = axs[2, 0]
for i, algorithm in enumerate(bp_group):
    data = df_decode_time_reset[df_decode_time_reset['Algorithm'] == algorithm]
    ax3.plot(data['Pack Size'], data['Decoding Time (MB/s)'], color=algorithm_palette[algorithm], linestyle='-', marker=markers[i], markersize=7, linewidth=2.2, label=algorithm)
if bp_rmq_decode_mean is not None:
    ax3.axhline(y=bp_rmq_decode_mean, color='#d62728', linestyle='--', linewidth=1.8, label='BP (Prune-RMQ)')
if bp_all_decode_mean is not None:
    ax3.axhline(y=bp_all_decode_mean, color='#2ca02c', linestyle='--', linewidth=1.8, label='Sort-BP (Prune-RMQ)')
ax3.set_xscale('log', base=2)
ax3.set_ylim(0, 120)
ax3.set_ylabel('Time (ns/point)', fontsize=fontsize)
ax3.set_xlabel('Pack Size $s$', fontsize=fontsize)
ax3.set_title('(e) BP: Decompression Time', fontsize=fontsize, x=0.4)
ax3.set_xticks(vector_sizes)
ax3.set_xticklabels(exponent_labels)
ax3.tick_params(labelsize=fontsize)
ax4 = axs[0, 1]
for i, algorithm in enumerate(sprintz_group):
    data = df_compression_ratio_reset[df_compression_ratio_reset['Algorithm'] == algorithm]
    ax4.plot(data['Pack Size'], data['Compression Ratio'], color=algorithm_palette[algorithm], linestyle='-', marker=markers[i + 4], markersize=7, linewidth=2.2, label=algorithm)
if sprintz_rmq_mean is not None:
    ax4.axhline(y=sprintz_rmq_mean, color='#17becf', linestyle='--', linewidth=1.8, label='Sprintz (Prune-RMQ)')
if sprintz_all_mean is not None:
    ax4.axhline(y=sprintz_all_mean, color='#8c564b', linestyle='--', linewidth=1.8, label='Sort-Sprintz (Prune-RMQ)')
ax4.set_xscale('log', base=2)
ax4.set_ylabel('Compression Ratio', fontsize=fontsize)
ax4.set_xlabel('Pack Size $s$', fontsize=fontsize)
ax4.set_title('(b) Sprintz: Compression Ratio', fontsize=fontsize, x=0.34)
ax4.set_xticks(vector_sizes)
ax4.set_xticklabels(exponent_labels)
ax4.tick_params(labelsize=fontsize)
ax5 = axs[1, 1]
for i, algorithm in enumerate(sprintz_group):
    data = df_encode_time_reset[df_encode_time_reset['Algorithm'] == algorithm]
    ax5.plot(data['Pack Size'], data['Encoding Time (MB/s)'], color=algorithm_palette[algorithm], linestyle='-', marker=markers[i + 4], markersize=7, linewidth=2.2, label=algorithm)
if sprintz_rmq_encode_mean is not None:
    ax5.axhline(y=sprintz_rmq_encode_mean, color='#17becf', linestyle='--', linewidth=1.8, label='Sprintz (Prune-RMQ)')
if sprintz_all_encode_mean is not None:
    ax5.axhline(y=sprintz_all_encode_mean, color='#8c564b', linestyle='--', linewidth=1.8, label='Sort-Sprintz (Prune-RMQ)')
ax5.set_xscale('log', base=2)
ax5.set_ylim(0, 120)
ax5.set_ylabel('Time (ns/point)', fontsize=fontsize)
ax5.set_xlabel('Pack Size $s$', fontsize=fontsize)
ax5.set_title('(d) Sprintz: Compression Time', fontsize=fontsize, x=0.34)
ax5.set_xticks(vector_sizes)
ax5.set_xticklabels(exponent_labels)
ax5.tick_params(labelsize=fontsize)
ax6 = axs[2, 1]
for i, algorithm in enumerate(sprintz_group):
    data = df_decode_time_reset[df_decode_time_reset['Algorithm'] == algorithm]
    ax6.plot(data['Pack Size'], data['Decoding Time (MB/s)'], color=algorithm_palette[algorithm], linestyle='-', marker=markers[i + 4], markersize=7, linewidth=2.2, label=algorithm)
if sprintz_rmq_decode_mean is not None:
    ax6.axhline(y=sprintz_rmq_decode_mean, color='#17becf', linestyle='--', linewidth=1.8, label='Sprintz (Prune-RMQ)')
if sprintz_all_decode_mean is not None:
    ax6.axhline(y=sprintz_all_decode_mean, color='#8c564b', linestyle='--', linewidth=1.8, label='Sort-Sprintz (Prune-RMQ)')
ax6.set_xscale('log', base=2)
ax6.set_ylim(0, 120)
ax6.set_ylabel('Time (ns/point)', fontsize=fontsize)
ax6.set_xlabel('Pack Size $s$', fontsize=fontsize)
ax6.set_title('(f) Sprintz: Decompression Time', fontsize=fontsize, x=0.33)
ax6.set_xticks(vector_sizes)
ax6.set_xticklabels(exponent_labels)
ax6.tick_params(labelsize=fontsize)
all_handles = []
all_labels = []
for ax_row in axs:
    for ax in ax_row:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
by_label = OrderedDict(zip(all_labels, all_handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=2, labelspacing=0.1, handletextpad=0.1, columnspacing=0.1, fontsize=fontsize - 1, bbox_to_anchor=(0.5, 1.02))
output_dir = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/figure_for_paper'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'sort_bp_vary_pack_size.png'), dpi=400, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'sort_bp_vary_pack_size.eps'), format='eps', dpi=400, bbox_inches='tight')
