import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
BASE = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size'
TSFILE_OUT_DIR = os.path.join(BASE, 'output_tsfile_packsize_comparison_cpp')
TSFILE_CSV_PATH = os.path.join(TSFILE_OUT_DIR, 'tsfile_comparison_cpp.csv')
dataset_mapping = {'City-temp.csv': 'CT', 'Wind-Speed.csv': 'WS', 'IR-bio-temp.csv': 'IR', 'PM10-dust.csv': 'PM10', 'Air-pressure.csv': 'AP', 'Dew-point-temp.csv': 'DT', 'Stocks-UK.csv': 'SUK', 'Stocks-USA.csv': 'SUA', 'Stocks-DE.csv': 'SDE', 'Bitcoin-price.csv': 'BP', 'Bird-migration.csv': 'BM', 'Food-price.csv': 'FP', 'electric_vehicle_charging.csv': 'VC', 'Blockchain-tr.csv': 'BTR', 'City-lat.csv': 'CLT', 'City-lon.csv': 'CLN'}
ALLOWED_DATASETS = frozenset(dataset_mapping.keys())
MODE_TO_ALGO = {'PackSize8': 'Sprintz', 'OptimalPackSize': 'Sprintz-Prune-RMQ'}
ALGOS = ['Sprintz', 'Sprintz-Prune-RMQ']
COLORS = ['#2ca02c', '#17becf']
IO_HATCH = '///'
LEGEND_CPU_IO_FILL = 'white'
EXAMPLE_BY_ALGO = {'Sprintz': {'compression_ratio': 6.5, 'compression_ratio_std': 0.4, 'write_enc_pp': 95.0, 'write_enc_std': 12.0, 'write_io_pp': 25.0, 'write_io_std': 4.0, 'read_dec_pp': 55.0, 'read_dec_std': 7.0, 'read_io_pp': 25.0, 'read_io_std': 4.0, 'write_total_std': 14.0, 'read_total_std': 9.0}, ALGOS[1]: {'compression_ratio': 7.0, 'compression_ratio_std': 0.35, 'write_enc_pp': 170.0, 'write_enc_std': 18.0, 'write_io_pp': 30.0, 'write_io_std': 5.0, 'read_dec_pp': 50.0, 'read_dec_std': 6.0, 'read_io_pp': 25.0, 'read_io_std': 4.0, 'write_total_std': 22.0, 'read_total_std': 8.0}}

def _std_across_rows(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors='coerce').dropna()
    return float(s.std(ddof=1)) if len(s) > 1 else 0.0

def load_tsfile_comparison():
    if not os.path.exists(TSFILE_CSV_PATH):
        return None
    df = pd.read_csv(TSFILE_CSV_PATH)
    if df.empty or 'Mode' not in df.columns:
        return None
    if 'Dataset' not in df.columns:
        return None
    req = ['Write Time (ns)', 'Read Time (ns)', 'Points', 'TsFile Size (bytes)']
    if not all((c in df.columns for c in req)):
        return None
    has_split = all((c in df.columns for c in ('Write Encode (ns)', 'Write IO (ns)', 'Read IO (ns)', 'Read Decode (ns)')))
    df = df[df['Dataset'].isin(ALLOWED_DATASETS) & df['Mode'].isin(MODE_TO_ALGO)].copy()
    if df.empty:
        return None
    df['_algo'] = df['Mode'].map(MODE_TO_ALGO)
    out = {}
    for algo in ALGOS:
        sub = df[df['_algo'] == algo]
        if sub.empty:
            return None
        pts = sub['Points'].astype(float)
        raw_bytes = pts * 8.0
        comp_bytes = sub['TsFile Size (bytes)'].astype(float)
        ratio_pre_over_post = raw_bytes / comp_bytes.replace(0, np.nan)
        comp_mean = float(ratio_pre_over_post.mean())
        if not np.isfinite(comp_mean):
            return None
        comp_std = _std_across_rows(ratio_pre_over_post)
        if has_split:
            we = sub['Write Encode (ns)'] / pts
            wi = sub['Write IO (ns)'] / pts
            rd = sub['Read Decode (ns)'] / pts
            ri = sub['Read IO (ns)'] / pts
            write_enc_pp = float(we.mean())
            write_io_pp = float(wi.mean())
            read_dec_pp = float(rd.mean())
            read_io_pp = float(ri.mean())
            write_enc_std = _std_across_rows(we)
            write_io_std = _std_across_rows(wi)
            read_dec_std = _std_across_rows(rd)
            read_io_std = _std_across_rows(ri)
            tw = we + wi
            tr = rd + ri
            write_total_std = _std_across_rows(tw)
            read_total_std = _std_across_rows(tr)
        else:
            wt = sub['Write Time (ns)'] / pts
            rt = sub['Read Time (ns)'] / pts
            write_enc_pp = float(wt.mean())
            write_io_pp = 0.0
            read_dec_pp = float(rt.mean())
            read_io_pp = 0.0
            write_enc_std = _std_across_rows(wt)
            write_io_std = 0.0
            read_dec_std = _std_across_rows(rt)
            read_io_std = 0.0
            write_total_std = write_enc_std
            read_total_std = read_dec_std
        out[algo] = {'compression_ratio': comp_mean, 'compression_ratio_std': comp_std, 'write_enc_pp': write_enc_pp, 'write_enc_std': write_enc_std, 'write_io_pp': write_io_pp, 'write_io_std': write_io_std, 'read_dec_pp': read_dec_pp, 'read_dec_std': read_dec_std, 'read_io_pp': read_io_pp, 'read_io_std': read_io_std, 'write_total_std': write_total_std, 'read_total_std': read_total_std, 'has_split': has_split}
    return out

def plot_system():
    data = load_tsfile_comparison()
    if data is None:
        print(f'Data not found or incomplete: {TSFILE_CSV_PATH}; using example data.')
        data = {a: {**EXAMPLE_BY_ALGO[a], 'has_split': True} for a in ALGOS}
    val_compression = [data[a]['compression_ratio'] for a in ALGOS]
    e_compression = [data[a]['compression_ratio_std'] for a in ALGOS]
    w_enc = [data[a]['write_enc_pp'] for a in ALGOS]
    w_io = [data[a]['write_io_pp'] for a in ALGOS]
    r_dec = [data[a]['read_dec_pp'] for a in ALGOS]
    r_io = [data[a]['read_io_pp'] for a in ALGOS]
    e_w_tot = [data[a]['write_total_std'] for a in ALGOS]
    e_r_tot = [data[a]['read_total_std'] for a in ALGOS]
    has_split = data[ALGOS[0]].get('has_split', False)
    x = np.arange(len(ALGOS))
    width = 0.5
    capsize = 5
    err_kw = {'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': 'k'}
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fontsize_title = 22
    fontsize_label = 22
    fontsize_tick = 22
    ax1 = axes[0]
    ax1.bar(x, val_compression, width, color=COLORS, edgecolor='k', linewidth=0.5, yerr=e_compression, capsize=capsize, error_kw=err_kw)
    ax1.set_ylabel('Compression ratio', fontsize=fontsize_label)
    ax1.set_title('(a) Compression Ratio', fontsize=fontsize_title)
    _a_hi = max((v + e for v, e in zip(val_compression, e_compression)))
    _a_lo = min((v - e for v, e in zip(val_compression, e_compression)))
    _a_margin = max(e_compression) * 0.15 + 0.05 * _a_hi
    ax1.set_ylim(max(0.0, _a_lo - _a_margin), _a_hi + _a_margin)
    ax1.tick_params(labelsize=fontsize_tick)
    ax2 = axes[1]
    for i in range(len(ALGOS)):
        if has_split and w_io[i] > 0:
            ax2.bar(x[i], w_io[i], width, color=COLORS[i], edgecolor='k', linewidth=0.5, hatch=IO_HATCH)
            ax2.bar(x[i], w_enc[i], width, bottom=w_io[i], color=COLORS[i], edgecolor='k', linewidth=0.5)
        else:
            ax2.bar(x[i], w_enc[i], width, color=COLORS[i], edgecolor='k', linewidth=0.5)
    tot_w_y = [w_io[i] + w_enc[i] if has_split and w_io[i] > 0 else w_enc[i] for i in range(len(ALGOS))]
    ax2.errorbar(x, tot_w_y, yerr=e_w_tot, fmt='none', capsize=capsize, **err_kw)
    ax2.set_ylabel('Time (ns/point)', fontsize=fontsize_label)
    ax2.set_title('(b) Write Time', fontsize=fontsize_title)
    ax2.tick_params(labelsize=fontsize_tick)
    ax3 = axes[2]
    for i in range(len(ALGOS)):
        if has_split and r_io[i] > 0:
            ax3.bar(x[i], r_io[i], width, color=COLORS[i], edgecolor='k', linewidth=0.5, hatch=IO_HATCH)
            ax3.bar(x[i], r_dec[i], width, bottom=r_io[i], color=COLORS[i], edgecolor='k', linewidth=0.5)
        else:
            ax3.bar(x[i], r_dec[i], width, color=COLORS[i], edgecolor='k', linewidth=0.5)
    tot_r_y = [r_io[i] + r_dec[i] if has_split and r_io[i] > 0 else r_dec[i] for i in range(len(ALGOS))]
    ax3.errorbar(x, tot_r_y, yerr=e_r_tot, fmt='none', capsize=capsize, **err_kw)
    ax3.set_ylabel('Time (ns/point)', fontsize=fontsize_label)
    ax3.set_title('(c) Read Time', fontsize=fontsize_title)
    ax3.tick_params(labelsize=fontsize_tick)
    wr_hi = [tot_w_y[i] + e_w_tot[i] for i in range(len(ALGOS))]
    rd_hi = [tot_r_y[i] + e_r_tot[i] for i in range(len(ALGOS))]
    _pad = 0.08 * max(max(wr_hi), max(rd_hi), 1e-09)
    bc_ymax = max(max(wr_hi), max(rd_hi), 1e-09) + _pad
    ax2.set_ylim(0, bc_ymax)
    ax3.set_ylim(0, bc_ymax)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ALGOS, fontsize=fontsize_tick, rotation=10, ha='right')
        ax.tick_params(axis='x', labelbottom=True, bottom=True)
    legend_handles = [Patch(facecolor=LEGEND_CPU_IO_FILL, edgecolor='k', linewidth=0.5, hatch='', label='CPU Time')]
    if has_split:
        legend_handles.append(Patch(facecolor=LEGEND_CPU_IO_FILL, edgecolor='k', linewidth=0.5, hatch=IO_HATCH, label='I/O Time'))
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2 if has_split else 1, frameon=True, fontsize=fontsize_tick)
    plt.tight_layout(rect=[0, 0, 1, 0.88], w_pad=0.35)
    out_dir = os.path.join(BASE, 'figure_for_paper')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'system_compare.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.splitext(out_path)[0] + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')
if __name__ == '__main__':
    plot_system()
