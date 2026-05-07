import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
BASE = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size'
SIMPLE8B_DIR = os.path.join(BASE, 'output_Simple8b')
ELFSTAR_DIR = os.path.join(BASE, 'output_ElfStar')
BITWEAVING_DIR = os.path.join(BASE, 'output_Bitweaving')
BP_JAVA_DIR = os.path.join(BASE, 'output_BP')
CAMEL_RATIO_PATH = os.path.join(BASE, 'camel_ratio.xlsx')
COMPRESSION_TIME_PATH = os.path.join(BASE, 'compression_time.xlsx')
DECOMPRESSION_TIME_PATH = os.path.join(BASE, 'decompression_time.xlsx')
dataset_mapping = {'City-temp.csv': 'CT', 'Wind-Speed.csv': 'WS', 'IR-bio-temp.csv': 'IR', 'PM10-dust.csv': 'PM10', 'Air-pressure.csv': 'AP', 'Dew-point-temp.csv': 'DT', 'Stocks-UK.csv': 'SUK', 'Stocks-USA.csv': 'SUA', 'Stocks-DE.csv': 'SDE', 'Bitcoin-price.csv': 'BP', 'Bird-migration.csv': 'BM', 'Cpu-usage_right.csv': 'CPU', 'Disk-usage.csv': 'DISK', 'Mem-usage.csv': 'MEM', 'Food-price.csv': 'FP', 'electric_vehicle_charging.csv': 'VC', 'Blockchain-tr.csv': 'BTR', 'SSD-bench.csv': 'SB', 'City-lat.csv': 'CLT', 'City-lon.csv': 'CLN'}
ALGOS = ['BP-Prune-RMQ', 'BP-Pack8', 'Bitweaving', 'Simple8b', 'ElfStar']
ERR_CAPSIZE = 2.0
ERR_ECOL = '#333333'

def _std_across_datasets(merged, algo: str, key: str) -> float:
    vals = [d[algo][key] for _, d in merged if algo in d and (not np.isnan(d[algo][key]))]
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1))

def _sem_across_datasets(merged, algo: str, key: str) -> float:
    vals = [d[algo][key] for _, d in merged if algo in d and (not np.isnan(d[algo][key]))]
    n = len(vals)
    if n < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(n))

def legend_label(algo: str) -> str:
    if algo == 'ElfStar':
        return 'Elf*'
    if algo == 'BP-Pack8':
        return 'BP'
    return algo

def load_simple8b():
    data = {abbr: {} for abbr in dataset_mapping.values()}
    for fname, abbr in dataset_mapping.items():
        path = os.path.join(SIMPLE8B_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            row = df.iloc[0]
            cr = float(row['Compression Ratio'])
            expansion = 1.0 / cr if cr > 0 else np.nan
            enc_mbs = float(row['Encoding Time'])
            dec_mbs = float(row['Decoding Time'])
            enc_ns = 8000.0 / enc_mbs if enc_mbs > 0 else np.nan
            dec_ns = 8000.0 / dec_mbs if dec_mbs > 0 else np.nan
            data[abbr]['Simple8b'] = {'ratio': expansion, 'encode_ns': enc_ns, 'decode_ns': dec_ns}
        except Exception as e:
            print(f'  Simple8b skip {fname}: {e}')
    return data

def load_bp_java():
    data = {abbr: {} for abbr in dataset_mapping.values()}
    algo_col = 'Encoding Algorithm'
    for fname, abbr in dataset_mapping.items():
        path = os.path.join(BP_JAVA_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            row = df.iloc[0]
            if str(row.get(algo_col, '')).strip() != 'BP':
                continue
            cr = float(row['Compression Ratio'])
            expansion = 1.0 / cr if cr > 0 else np.nan
            enc_mbs = float(row['Encoding Time'])
            dec_mbs = float(row['Decoding Time'])
            enc_ns = 8000.0 / enc_mbs if enc_mbs > 0 else np.nan
            dec_ns = 8000.0 / dec_mbs if dec_mbs > 0 else np.nan
            data[abbr]['BP-Pack8'] = {'ratio': expansion, 'encode_ns': enc_ns, 'decode_ns': dec_ns}
        except Exception as e:
            print(f'  BP (Java) skip {fname}: {e}')
    return data

def load_elfstar():
    data = {abbr: {} for abbr in dataset_mapping.values()}
    algo_col = 'Encoding Algorithm'
    name_map = {'ElfStarElfHuffXORCompressor': 'ElfStar'}
    for fname, abbr in dataset_mapping.items():
        path = os.path.join(ELFSTAR_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                algo_raw = row.get(algo_col, '')
                algo = name_map.get(algo_raw)
                if algo is None:
                    continue
                cr = float(row['Compression Ratio'])
                expansion = 1.0 / cr if cr > 0 else np.nan
                enc_mbs = float(row['Encoding Time'])
                dec_mbs = float(row['Decoding Time'])
                points = int(row['Points'])
                enc_ns = 8000.0 / enc_mbs if enc_mbs > 0 else np.nan
                dec_ns = 8000.0 / dec_mbs if dec_mbs > 0 else np.nan
                data[abbr][algo] = {'ratio': expansion, 'encode_ns': enc_ns, 'decode_ns': dec_ns}
        except Exception as e:
            print(f'  ElfStar skip {fname}: {e}')
    return data

def load_bitweaving():
    data = {abbr: {} for abbr in dataset_mapping.values()}
    for fname, abbr in dataset_mapping.items():
        path = os.path.join(BITWEAVING_DIR, fname)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            row = df.iloc[0]
            cr = float(row['Compression Ratio'])
            expansion = 1.0 / cr if cr > 0 else np.nan
            enc_mbs = float(row['Encoding Throughput (MB/s)'])
            dec_mbs = float(row['Decoding Throughput (MB/s)'])
            enc_ns = 8000.0 / enc_mbs if enc_mbs > 0 else np.nan
            dec_ns = 8000.0 / dec_mbs if dec_mbs > 0 else np.nan
            data[abbr]['Bitweaving'] = {'ratio': expansion, 'encode_ns': enc_ns, 'decode_ns': dec_ns}
        except Exception as e:
            print(f'  Bitweaving skip {fname}: {e}')
    return data

def load_bp_prune_rmq():
    data = {abbr: {} for abbr in dataset_mapping.values()}
    valid_abbrs = set(dataset_mapping.values())
    if os.path.exists(CAMEL_RATIO_PATH):
        df_ratio = pd.read_excel(CAMEL_RATIO_PATH)
        row = df_ratio[df_ratio.iloc[:, 0] == 'BP (Prune-RMQ)']
        if not row.empty:
            for abbr in valid_abbrs:
                if abbr not in row.columns:
                    continue
                val = row[abbr].iloc[0]
                if pd.notna(val):
                    try:
                        data[abbr]['BP-Prune-RMQ'] = {'ratio': 1.0 / float(val), 'encode_ns': np.nan, 'decode_ns': np.nan}
                    except Exception:
                        pass
    if os.path.exists(COMPRESSION_TIME_PATH):
        df_enc = pd.read_excel(COMPRESSION_TIME_PATH)
        enc_row = df_enc[df_enc.iloc[:, 0] == 'BP (Prune-RMQ)']
        if not enc_row.empty:
            for abbr in valid_abbrs:
                if abbr not in enc_row.columns:
                    continue
                val = enc_row[abbr].iloc[0]
                if pd.notna(val):
                    try:
                        t_ns = 8000.0 / float(val)
                        if abbr not in data:
                            data[abbr]['BP-Prune-RMQ'] = {'ratio': np.nan, 'encode_ns': np.nan, 'decode_ns': np.nan}
                        data[abbr]['BP-Prune-RMQ']['encode_ns'] = t_ns
                    except Exception:
                        pass
    if os.path.exists(DECOMPRESSION_TIME_PATH):
        df_dec = pd.read_excel(DECOMPRESSION_TIME_PATH)
        dec_row = df_dec[df_dec.iloc[:, 0] == 'BP (Prune-RMQ)']
        if not dec_row.empty:
            for abbr in valid_abbrs:
                if abbr not in dec_row.columns:
                    continue
                val = dec_row[abbr].iloc[0]
                if pd.notna(val):
                    try:
                        t_ns = 8000.0 / float(val)
                        if abbr not in data:
                            data[abbr]['BP-Prune-RMQ'] = {'ratio': np.nan, 'encode_ns': np.nan, 'decode_ns': np.nan}
                        data[abbr]['BP-Prune-RMQ']['decode_ns'] = t_ns
                    except Exception:
                        pass
    return data

def merge_datasets(s8, elfstar, bitweaving, bp, bp_java):
    abbrs = sorted(dataset_mapping.values())
    out = []
    for abbr in abbrs:
        d = {}
        for algo in ALGOS:
            if algo == 'Simple8b' and abbr in s8 and ('Simple8b' in s8[abbr]):
                d[algo] = s8[abbr]['Simple8b']
            elif algo == 'ElfStar' and abbr in elfstar and ('ElfStar' in elfstar[abbr]):
                d[algo] = elfstar[abbr]['ElfStar']
            elif algo == 'Bitweaving' and abbr in bitweaving and ('Bitweaving' in bitweaving[abbr]):
                d[algo] = bitweaving[abbr]['Bitweaving']
            elif algo == 'BP-Prune-RMQ' and abbr in bp and ('BP-Prune-RMQ' in bp[abbr]):
                d[algo] = bp[abbr]['BP-Prune-RMQ']
            elif algo == 'BP-Pack8' and abbr in bp_java and ('BP-Pack8' in bp_java[abbr]):
                d[algo] = bp_java[abbr]['BP-Pack8']
        if d:
            out.append((abbr, d))
    return out

def plot_compare_baseline():
    s8 = load_simple8b()
    elfstar = load_elfstar()
    bitweaving = load_bitweaving()
    bp = load_bp_prune_rmq()
    bp_java = load_bp_java()
    merged = merge_datasets(s8, elfstar, bitweaving, bp, bp_java)
    if not merged:
        print('No common datasets found.')
        return
    datasets = [m[0] for m in merged]
    n = len(datasets)
    x = np.arange(n)
    n_algos = len(ALGOS)
    width = 0.8 / n_algos * 0.95

    def bar_offset(i):
        return (i - (n_algos - 1) / 2) * width
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fontsize = 22
    colors = {'BP-Prune-RMQ': '#d62728', 'BP-Pack8': '#ff6b6b', 'Bitweaving': '#8c564b', 'Simple8b': '#ff7f0e', 'ElfStar': '#17becf'}

    def hatch_by_index(i: int) -> str:
        return '\\' if i % 2 == 0 else '/'
    std_ratio_by_algo = {a: _std_across_datasets(merged, a, 'ratio') for a in ALGOS}
    std_enc_by_algo = {a: _std_across_datasets(merged, a, 'encode_ns') for a in ALGOS}
    std_dec_by_algo = {a: _std_across_datasets(merged, a, 'decode_ns') for a in ALGOS}
    err_kw = dict(elinewidth=0.8, capthick=0.8, ecolor=ERR_ECOL)
    ax1 = axes[0]
    for i, algo in enumerate(ALGOS):
        vals = []
        errs = []
        for abbr, d in merged:
            if algo in d and (not np.isnan(d[algo]['ratio'])):
                vals.append(d[algo]['ratio'])
                errs.append(std_ratio_by_algo[algo])
            else:
                vals.append(0)
                errs.append(0.0)
        ax1.bar(x + bar_offset(i), vals, width, yerr=errs, capsize=ERR_CAPSIZE, error_kw=err_kw, label=legend_label(algo), color=colors[algo], hatch=hatch_by_index(i), edgecolor='white', linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha='right', fontsize=fontsize)
    ax1.set_ylabel('Compression Ratio', fontsize=fontsize)
    ax1.set_title('(a) Compression Ratio', fontsize=fontsize)
    ax1.legend(fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    ax2 = axes[1]
    for i, algo in enumerate(ALGOS):
        vals = []
        errs = []
        for abbr, d in merged:
            if algo in d and (not np.isnan(d[algo]['encode_ns'])):
                vals.append(d[algo]['encode_ns'])
                errs.append(std_enc_by_algo[algo])
            else:
                vals.append(0)
                errs.append(0.0)
        ax2.bar(x + bar_offset(i), vals, width, yerr=errs, capsize=ERR_CAPSIZE, error_kw=err_kw, label=legend_label(algo), color=colors[algo], hatch=hatch_by_index(i), edgecolor='white', linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45, ha='right', fontsize=fontsize)
    ax2.set_ylabel('Time (ns/point)', fontsize=fontsize)
    ax2.set_title('(b) Compression Time', fontsize=fontsize)
    ax2.legend(fontsize=fontsize)
    ax2.tick_params(labelsize=fontsize)
    ax3 = axes[2]
    for i, algo in enumerate(ALGOS):
        vals = []
        errs = []
        for abbr, d in merged:
            if algo in d and (not np.isnan(d[algo]['decode_ns'])):
                vals.append(d[algo]['decode_ns'])
                errs.append(std_dec_by_algo[algo])
            else:
                vals.append(0)
                errs.append(0.0)
        ax3.bar(x + bar_offset(i), vals, width, yerr=errs, capsize=ERR_CAPSIZE, error_kw=err_kw, label=legend_label(algo), color=colors[algo], hatch=hatch_by_index(i), edgecolor='white', linewidth=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets, rotation=45, ha='right', fontsize=fontsize)
    ax3.set_ylabel('Time (ns/point)', fontsize=fontsize)
    ax3.set_title('(c) Decompression Time', fontsize=fontsize)
    ax3.legend(fontsize=fontsize)
    ax3.tick_params(labelsize=fontsize)
    plt.tight_layout()
    out_dir = os.path.join(BASE, 'figure_for_paper')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'compare_baseline.png')
    out_path_eps = os.path.join(out_dir, 'compare_baseline.eps')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_eps, format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')
    avg_ratio = []
    avg_encode_ns = []
    avg_decode_ns = []
    err_avg_ratio = []
    err_avg_encode_ns = []
    err_avg_decode_ns = []
    for algo in ALGOS:
        ratios = [d[algo]['ratio'] for abbr, d in merged if algo in d and (not np.isnan(d[algo]['ratio']))]
        encodes = [d[algo]['encode_ns'] for abbr, d in merged if algo in d and (not np.isnan(d[algo]['encode_ns']))]
        decodes = [d[algo]['decode_ns'] for abbr, d in merged if algo in d and (not np.isnan(d[algo]['decode_ns']))]
        avg_ratio.append(np.mean(ratios) if ratios else np.nan)
        avg_encode_ns.append(np.mean(encodes) if encodes else np.nan)
        avg_decode_ns.append(np.mean(decodes) if decodes else np.nan)
        err_avg_ratio.append(_sem_across_datasets(merged, algo, 'ratio'))
        err_avg_encode_ns.append(_sem_across_datasets(merged, algo, 'encode_ns'))
        err_avg_decode_ns.append(_sem_across_datasets(merged, algo, 'decode_ns'))
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3))
    x_algos = np.arange(n_algos)
    width_single = 0.5
    ax1 = axes2[0]
    for i, algo in enumerate(ALGOS):
        v = 0 if np.isnan(avg_ratio[i]) else avg_ratio[i]
        e = 0 if np.isnan(avg_ratio[i]) else err_avg_ratio[i]
        ax1.bar(x_algos[i], v, width_single, yerr=e, capsize=ERR_CAPSIZE, error_kw=err_kw, color=colors[algo], edgecolor='white', linewidth=0.6, hatch=hatch_by_index(i), label=legend_label(algo))
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_ylabel('Compression Ratio', fontsize=fontsize, y=0.6)
    ax1.set_title('(a) Compression Ratio', fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    ax1.set_ylim(bottom=1)
    ax2 = axes2[1]
    for i, algo in enumerate(ALGOS):
        v = 0 if np.isnan(avg_encode_ns[i]) else avg_encode_ns[i]
        e = 0 if np.isnan(avg_encode_ns[i]) else err_avg_encode_ns[i]
        ax2.bar(x_algos[i], v, width_single, yerr=e, capsize=ERR_CAPSIZE, error_kw=err_kw, color=colors[algo], edgecolor='white', linewidth=0.6, hatch=hatch_by_index(i), label=legend_label(algo))
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    ax2.set_ylabel('Time (ns/point)', fontsize=fontsize)
    ax2.set_title('(b) Compression Time', fontsize=fontsize)
    ax2.tick_params(labelsize=fontsize)
    ax3 = axes2[2]
    for i, algo in enumerate(ALGOS):
        v = 0 if np.isnan(avg_decode_ns[i]) else avg_decode_ns[i]
        e = 0 if np.isnan(avg_decode_ns[i]) else err_avg_decode_ns[i]
        ax3.bar(x_algos[i], v, width_single, yerr=e, capsize=ERR_CAPSIZE, error_kw=err_kw, color=colors[algo], edgecolor='white', linewidth=0.6, hatch=hatch_by_index(i), label=legend_label(algo))
    ax3.set_xticks([])
    ax3.set_xticklabels([])
    ax3.set_ylabel('Time (ns/point)', fontsize=fontsize)
    ax3.set_title('(c) Decompression Time', fontsize=fontsize, x=0.4)
    ax3.tick_params(labelsize=fontsize)
    avg_handles = [Patch(facecolor=colors[algo], edgecolor='white', linewidth=0.6, hatch=hatch_by_index(i), label=legend_label(algo)) for i, algo in enumerate(ALGOS)]
    fig2.legend(handles=avg_handles, fontsize=fontsize, labels=[legend_label(a) for a in ALGOS], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, columnspacing=0.5, handletextpad=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path_avg = os.path.join(out_dir, 'compare_baseline_avg.png')
    plt.savefig(out_path_avg, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.splitext(out_path_avg)[0] + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path_avg}')
if __name__ == '__main__':
    plot_compare_baseline()
