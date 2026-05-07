import os
import sys
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'Noto Sans CJK SC'] + list(matplotlib.rcParams.get('font.sans-serif', []))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ALP_DIR = os.path.join(BASE, 'output_ALP')
OUTPUT_ALP_V5_DIR = os.path.join(BASE, 'output_ALP_optimal_v5')
OUTPUT_CUSZP_DIR = os.path.join(BASE, 'output_cuszp_cpu')
OUTPUT_CUSZP_V5_DIR = os.path.join(BASE, 'output_cuszp_cpu_optimal_v5')
ALP_CSV = [(['ALP'], 'ALP'), (['ALP+V5pack', 'ALP+optimal-V5', 'ALP optimal V5'], 'ALP（Prune-RMQ）')]
ALP_ALGOS = [x[1] for x in ALP_CSV]
CUSZP_CSV = [(['cuSZp-cpu-plain-simplified', 'cuSZp-cpu', 'CuSZp-cpu'], 'CuSZp2'), (['cuSZp-cpu+V5pack', 'cuSZp+V5pack', 'CuSZp+V5pack'], 'CuSZp2 (Prune-RMQ)')]
CUSZP_ALGOS = [x[1] for x in CUSZP_CSV]
GROUP_COLOR_ALP = '#9467bd'
GROUP_COLOR_CUSZP = '#5d8499'
ALGO_FACE_COLOR = {'ALP': GROUP_COLOR_ALP, 'ALP（Prune-RMQ）': GROUP_COLOR_ALP, 'CuSZp2': GROUP_COLOR_CUSZP, 'CuSZp2 (Prune-RMQ)': GROUP_COLOR_CUSZP}
ALGO_USE_HATCH = {'ALP': False, 'ALP（Prune-RMQ）': True, 'CuSZp2': False, 'CuSZp2 (Prune-RMQ)': True}
ALGO_ORDER = ('ALP', 'ALP（Prune-RMQ）', 'CuSZp2', 'CuSZp2 (Prune-RMQ)')
BAR_HATCH_PATTERN = '///'
ERRORBAR_NEAREST_K = 5
ERRORBAR_ECOLOR = '0.2'
ERRORBAR_CAPSIZE = 2.8

def _bar_kwargs(algo, fc):
    kw = {'color': fc}
    if ALGO_USE_HATCH.get(algo, False):
        kw['edgecolor'] = 'white'
        kw['linewidth'] = 1.0
        kw['hatch'] = BAR_HATCH_PATTERN
    else:
        kw['edgecolor'] = 'none'
        kw['linewidth'] = 0
    return kw

def nearest_k_minmax_errors(vals, m, k, log_scale=False):
    if k is None or k < 1:
        return (0.0, 0.0)
    a = np.asarray(vals, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if log_scale:
        a = a[a > 0]
    if a.size == 0 or not np.isfinite(m):
        return (0.0, 0.0)
    k_eff = min(int(k), a.size)
    dist = np.abs(a - float(m))
    idx = np.argpartition(dist, k_eff - 1)[:k_eff]
    subset = a[idx]
    lo, hi = (float(np.min(subset)), float(np.max(subset)))
    el = max(0.0, float(m) - lo)
    eu = max(0.0, hi - float(m))
    return (el, eu)

def _legend_patch_kwargs(name):
    pkw = {'facecolor': ALGO_FACE_COLOR[name], 'label': name}
    if ALGO_USE_HATCH.get(name, False):
        pkw['edgecolor'] = 'white'
        pkw['linewidth'] = 1.0
        pkw['hatch'] = BAR_HATCH_PATTERN
    else:
        pkw['edgecolor'] = 'none'
        pkw['linewidth'] = 0
    return pkw
YLABELS_METRICS = ('Compression Ratio', 'Time (ns/point)', 'Time (ns/point)')
dataset_mapping = {'City-temp.csv': 'CT', 'Wind-Speed.csv': 'WS', 'IR-bio-temp.csv': 'IR', 'PM10-dust.csv': 'PM10', 'Air-pressure.csv': 'AP', 'Dew-point-temp.csv': 'DT', 'Stocks-UK.csv': 'SUK', 'Stocks-USA.csv': 'SUA', 'Stocks-DE.csv': 'SDE', 'Bitcoin-price.csv': 'BP', 'Bird-migration.csv': 'BM', 'Cpu-usage_right.csv': 'CPU', 'Disk-usage.csv': 'DISK', 'Mem-usage.csv': 'MEM', 'Food-price.csv': 'FP', 'electric_vehicle_charging.csv': 'VC', 'Blockchain-tr.csv': 'BTR', 'SSD-bench.csv': 'SB', 'City-lat.csv': 'CLT', 'City-lon.csv': 'CLN'}

def _pick_row(df, csv_names):
    algo_col = df['Encoding Algorithm'].astype(str).str.strip().str.strip('"')
    for csv_name in csv_names:
        m = algo_col == csv_name
        if m.any():
            return df.loc[m].iloc[0]
    return None

def _row_to_metrics_alp(row):
    try:
        cr = float(row['Compression Ratio'])
        expansion = 1.0 / cr if cr > 0 else np.nan
        enc_mbs = float(row['Encoding Time'])
        dec_mbs = float(row['Decoding Time'])
        enc_ns = 4000.0 / enc_mbs if enc_mbs > 0 else np.nan
        dec_ns = 4000.0 / dec_mbs if dec_mbs > 0 else np.nan
        return (expansion, enc_ns, dec_ns)
    except Exception:
        return (np.nan, np.nan, np.nan)

def _row_to_metrics_cuszp(row):
    try:
        pts = float(row['Points'])
        cb = float(row['Compressed Size (bits)'])
        cr = cb / (pts * 64.0) if pts > 0 else np.nan
        expansion = 1.0 / cr if cr > 0 else np.nan
        enc_mbs = float(row['Encoding Time'])
        dec_mbs = float(row['Decoding Time'])
        enc_ns = 4000.0 / enc_mbs if enc_mbs > 0 else np.nan
        dec_ns = 4000.0 / dec_mbs if dec_mbs > 0 else np.nan
        return (expansion, enc_ns, dec_ns)
    except Exception:
        return (np.nan, np.nan, np.nan)

def load_merged_alp():
    data = {abbr: {} for abbr in dataset_mapping.values()}
    for fname, abbr in dataset_mapping.items():
        p_alp = os.path.join(OUTPUT_ALP_DIR, fname)
        p_v5 = os.path.join(OUTPUT_ALP_V5_DIR, fname)
        if not os.path.exists(p_alp) or not os.path.exists(p_v5):
            continue
        try:
            df8 = pd.read_csv(p_alp)
            dfv = pd.read_csv(p_v5)
            if 'Encoding Algorithm' not in df8.columns or 'Encoding Algorithm' not in dfv.columns:
                continue
            for path_df, candidates in ((df8, ALP_CSV[0]), (dfv, ALP_CSV[1])):
                csv_names, disp = candidates
                row = _pick_row(path_df, csv_names)
                if row is None:
                    continue
                ratio, enc_ns, dec_ns = _row_to_metrics_alp(row)
                data[abbr][disp] = {'ratio': ratio, 'encode_ns': enc_ns, 'decode_ns': dec_ns}
        except Exception as e:
            print(f'  ALP skip {fname}: {e}', file=sys.stderr)
    return data

def load_merged_cuszp():
    data = {abbr: {} for abbr in dataset_mapping.values()}
    for fname, abbr in dataset_mapping.items():
        p8 = os.path.join(OUTPUT_CUSZP_DIR, fname)
        pv = os.path.join(OUTPUT_CUSZP_V5_DIR, fname)
        if not os.path.exists(p8) or not os.path.exists(pv):
            continue
        try:
            df8 = pd.read_csv(p8)
            dfv = pd.read_csv(pv)
            if 'Encoding Algorithm' not in df8.columns or 'Encoding Algorithm' not in dfv.columns:
                continue
            for path_df, candidates in ((df8, CUSZP_CSV[0]), (dfv, CUSZP_CSV[1])):
                csv_names, disp = candidates
                row = _pick_row(path_df, csv_names)
                if row is None:
                    continue
                ratio, enc_ns, dec_ns = _row_to_metrics_cuszp(row)
                data[abbr][disp] = {'ratio': ratio, 'encode_ns': enc_ns, 'decode_ns': dec_ns}
        except Exception as e:
            print(f'  CuSZp skip {fname}: {e}', file=sys.stderr)
    return data

def merge_ordered_alp(simd_data):
    abbrs = sorted(dataset_mapping.values())
    out = []
    for abbr in abbrs:
        if abbr not in simd_data or len(simd_data[abbr]) < 2:
            continue
        d = {algo: simd_data[abbr][algo] for algo in ALP_ALGOS if algo in simd_data[abbr]}
        if len(d) == 2:
            out.append((abbr, d))
    return out

def merge_ordered_cuszp(simd_data):
    abbrs = sorted(dataset_mapping.values())
    out = []
    for abbr in abbrs:
        if abbr not in simd_data or len(simd_data[abbr]) < 2:
            continue
        d = {algo: simd_data[abbr][algo] for algo in CUSZP_ALGOS if algo in simd_data[abbr]}
        if len(d) == 2:
            out.append((abbr, d))
    return out

def intersection_merged(merged_alp, merged_cuszp):
    da = {a: d for a, d in merged_alp}
    db = {a: d for a, d in merged_cuszp}
    order = sorted(dataset_mapping.values())
    common = [a for a in order if a in da and a in db]
    return ([(a, da[a]) for a in common], [(a, db[a]) for a in common])

def _grouped_bar_xcenters(xc, w, gap_in, gap_between_groups):
    span = 4 * w + 2 * gap_in + gap_between_groups
    c0 = xc - span / 2 + w / 2
    c1 = c0 + w + gap_in
    c2 = c1 + w + gap_between_groups
    c3 = c2 + w + gap_in
    return (c0, c1, c2, c3, span)

def _plot_grouped_metrics(axes, merged_alp_i, merged_cuszp_i, fontsize, y_labelsize, letters=('a', 'b', 'c'), *, bar_w=0.14, gap_in=0.012, gap_between_groups=0.07, gap_clusters=0.38):
    n = len(merged_alp_i)
    if n == 0:
        return (1.0, 0.14)
    _, _, _, _, span = _grouped_bar_xcenters(0.0, bar_w, gap_in, gap_between_groups)
    pitch = span + gap_clusters
    keys = ('ratio', 'encode_ns', 'decode_ns')
    titles = (f'({letters[0]}) Compression Ratio', f'({letters[1]}) Compression Time', f'({letters[2]}) Decompression Time')
    err_kw_bar = dict(elinewidth=1.1, capthick=1.0, alpha=0.88, zorder=5)
    ratio_y_low = []
    for ax, title, key, yl in zip(axes, titles, keys, YLABELS_METRICS):
        for k in range(n):
            _abbr_a, d_a = merged_alp_i[k]
            _abbr_c, d_c = merged_cuszp_i[k]
            xc = k * pitch
            c0, c1, c2, c3, _ = _grouped_bar_xcenters(xc, bar_w, gap_in, gap_between_groups)
            vals = []
            for algo in ALGO_ORDER:
                src = d_a if algo in ALP_ALGOS else d_c
                v = src[algo][key]
                vals.append(0.0 if np.isnan(v) else float(v))
            for xi, algo, v in zip((c0, c1, c2, c3), ALGO_ORDER, vals):
                fc = ALGO_FACE_COLOR.get(algo, '#7f7f7f')
                pool = []
                for j in range(n):
                    if j == k:
                        continue
                    d_aj, d_cj = (merged_alp_i[j][1], merged_cuszp_i[j][1])
                    src_j = d_aj if algo in ALP_ALGOS else d_cj
                    if algo in src_j:
                        vv = float(src_j[algo][key])
                        if np.isfinite(vv):
                            pool.append(vv)
                if np.isfinite(v) and len(pool) > 0 and (ERRORBAR_NEAREST_K is not None) and (int(ERRORBAR_NEAREST_K) >= 1):
                    el, eu = nearest_k_minmax_errors(pool, v, ERRORBAR_NEAREST_K)
                else:
                    el, eu = (0.0, 0.0)
                if key == 'ratio' and np.isfinite(v) and np.isfinite(el):
                    ratio_y_low.append(v - el)
                bar_kw = _bar_kwargs(algo, fc)
                if el > 0 or eu > 0:
                    ax.bar(xi, v, bar_w, yerr=[[el], [eu]], ecolor=ERRORBAR_ECOLOR, capsize=ERRORBAR_CAPSIZE, error_kw=err_kw_bar, **bar_kw)
                else:
                    ax.bar(xi, v, bar_w, **bar_kw)
        ax.set_xlim(-0.5 * pitch, (n - 1) * pitch + 0.5 * pitch)
        ax.set_xticks([])
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_ylabel(yl, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=y_labelsize)
    ratio_bottom = 3.0
    if ratio_y_low:
        ratio_bottom = min(ratio_bottom, min(ratio_y_low) * 0.97)
    axes[0].set_ylim(bottom=ratio_bottom)
    return (pitch, span)

def _default_cluster_pitch():
    _, _, _, _, span = _grouped_bar_xcenters(0.0, 0.14, 0.012, 0.07)
    return span + 0.38

def plot_combined():
    raw_a = load_merged_alp()
    raw_c = load_merged_cuszp()
    merged_alp = merge_ordered_alp(raw_a)
    merged_cuszp = merge_ordered_cuszp(raw_c)
    merged_alp_i, merged_cuszp_i = intersection_merged(merged_alp, merged_cuszp)
    if not merged_alp_i:
        print('No datasets with both ALP and CuSZp paired outputs; run both test harnesses.', file=sys.stderr)
        return
    n = len(merged_alp_i)
    fontsize = 14
    y_labelsize = fontsize
    pitch = _default_cluster_pitch()
    fig_w = min(18, max(9.5, n * pitch + 1.8))
    fig_h = 2.1
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), sharex=False)
    _plot_grouped_metrics(axes, merged_alp_i, merged_cuszp_i, fontsize, y_labelsize, ('a', 'b', 'c'))
    legend_elements = [Patch(**_legend_patch_kwargs(name)) for name in ALGO_ORDER]
    plt.tight_layout(rect=(0.06, 0.1, 1.0, 0.74))
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.07), bbox_transform=fig.transFigure, ncol=4, fontsize=fontsize, frameon=True, fancybox=False, edgecolor='0.8', borderaxespad=0.0)
    out_dir = os.path.join(BASE, 'figure_for_paper')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'alp_cuszp_pack8_vs_v5_compare_combined.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.splitext(out_path)[0] + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')
    bar_w = 0.16
    gap_in = 0.014
    gap_between_groups = 0.08
    c0, c1, c2, c3, span = _grouped_bar_xcenters(0.0, bar_w, gap_in, gap_between_groups)
    pad = 0.12
    xlim = (c0 - bar_w / 2 - pad, c3 + bar_w / 2 + pad)
    fig2_w = 10.0
    fig2_h = 2.0
    fig2, axes2 = plt.subplots(1, 3, figsize=(fig2_w, fig2_h), sharex=False)
    fig2.subplots_adjust(wspace=0.35)
    keys = ('ratio', 'encode_ns', 'decode_ns')
    titles = ('(a) Compression Ratio', '(b) Compression Time', '(c) Decompression Time')
    err_kw_avg = dict(elinewidth=1.1, capthick=1.0, alpha=0.88, zorder=5)
    ratio_avg_low = []
    for ax, title, key, yl in zip(axes2, titles, keys, YLABELS_METRICS):
        for xi, algo in zip((c0, c1, c2, c3), ALGO_ORDER):
            if algo in ALP_ALGOS:
                vals = [float(d[algo][key]) for _abbr, d in merged_alp_i if algo in d and np.isfinite(d[algo][key])]
            else:
                vals = [float(d[algo][key]) for _abbr, d in merged_cuszp_i if algo in d and np.isfinite(d[algo][key])]
            v = float(np.mean(vals)) if vals else 0.0
            yerr = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if key == 'ratio' and np.isfinite(v) and np.isfinite(yerr):
                ratio_avg_low.append(v - yerr)
            fc = ALGO_FACE_COLOR.get(algo, '#7f7f7f')
            bar_kw = _bar_kwargs(algo, fc)
            if yerr > 0:
                ax.bar(xi, v, bar_w, yerr=yerr, ecolor=ERRORBAR_ECOLOR, capsize=ERRORBAR_CAPSIZE, error_kw=err_kw_avg, **bar_kw)
            else:
                ax.bar(xi, v, bar_w, **bar_kw)
        ax.set_xlim(xlim)
        ax.set_xticks([])
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_ylabel(yl, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize, x=0.45)
        ax.tick_params(axis='y', labelsize=y_labelsize)
    avg_ratio_bottom = 3.0
    if ratio_avg_low:
        avg_ratio_bottom = min(avg_ratio_bottom, min(ratio_avg_low) * 0.97)
    axes2[0].set_ylim(bottom=avg_ratio_bottom)
    fig2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.23), bbox_transform=fig2.transFigure, ncol=4, fontsize=fontsize, frameon=True, fancybox=False, edgecolor='0.8', borderaxespad=0.0, columnspacing=0.5)
    out_avg = os.path.join(out_dir, 'alp_cuszp_pack8_vs_v5_compare_combined_avg.png')
    plt.savefig(out_avg, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.splitext(out_avg)[0] + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_avg}')
if __name__ == '__main__':
    plot_combined()
