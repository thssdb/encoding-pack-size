#!/usr/bin/env python3.11
"""
Compare eight pipelines from optimal_pack_prune (CSV outputs).

Layout: four groups (Scalar / SIMD / FastLanes / SIMT); plain BP solid, Prune-RMQ same color + white hatch ///.
Data: encode/decode from output_simd except SIMT (output_simd2); compression ratio (expansion) from output_simd2 when present.
Error bars: black — per-dataset = nearest-k min–max on other datasets; average = sample std across datasets.
Subplot (a) y-axis is set from all bars ± errors so caps stay visible.

Benchmark C++: ENC_TIME_REPEATS=500 / DEC_TIME_REPEATS=1000 in optimal_pack_prune_main.cpp.

Usage: python3.11 fig_simd.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.abspath(__file__))

GROUP_COLORS = ["#9467bd", "#d62728", "#8c564b", "#17becf"]
OUTPUT_SIMD_DIR = os.path.join(BASE, "output_simd")
OUTPUT_SIMD2_DIR = os.path.join(BASE, "output_simd2")
SIMT_LEGEND_LABELS = frozenset({"BP (SIMT)", "BP-Prune-RMQ (SIMT)"})

ALGO_CSV_CANDIDATES = [
    (["BP-16", "BP-Scalar"], "BP (Scalar)"),
    (["BP-SIMD-16", "BP-SIMDComp"], "BP (SIMD)"),
    (["BP-FastLanes-16", "BP-FastLanes"], "BP (FastLanes)"),
    (["BP-SIMT-16"], "BP (SIMT)"),
    (["BP+Prune", "BP+Prune-Scalar"], "BP-Prune-RMQ (Scalar)"),
    (["BP+Prune-SIMD", "BP+Prune-SIMDComp"], "BP-Prune-RMQ (SIMD)"),
    (["BP+Prune-FastLanes"], "BP-Prune-RMQ (FastLanes)"),
    (["BP+Prune-SIMT"], "BP-Prune-RMQ (SIMT)"),
]
ALGOS = [x[1] for x in ALGO_CSV_CANDIDATES]
N_IMPL = 4

_BAR_W = 0.10
_GROUP_CENTER_OFF = np.array([-0.33, -0.11, 0.11, 0.33], dtype=float)
_BAR_PAIR_HALF = 0.055

ERRORBAR_NEAREST_K = 5
ERRORBAR_ECOLOR = "k"
ERRORBAR_CAPSIZE = 2.8
ERR_KW_BAR = dict(elinewidth=1.1, capthick=1.0, alpha=1.0, zorder=5)
ERR_KW_AVG = dict(elinewidth=1.1, capthick=1.0, alpha=1.0, zorder=5)

# Fallback ylim for ratio panel when range cannot be computed
RATIO_YLIM_FALLBACK = (4.75, 12.0)
# Extra vertical margin as fraction of span so error caps are not flush to the axis
RATIO_Y_MARGIN_FRAC = 0.05

# All figure text (ticks, labels, titles, legend) use this size.
FONTSIZE = 22

EXAMPLE_DATA = [
    (
        "Example",
        {
            "BP (Scalar)": {"ratio": 10.0, "encode_ns": 190.0, "decode_ns": 70.0},
            "BP (SIMD)": {"ratio": 10.0, "encode_ns": 110.0, "decode_ns": 36.0},
            "BP (FastLanes)": {"ratio": 10.0, "encode_ns": 120.0, "decode_ns": 38.0},
            "BP (SIMT)": {"ratio": 10.0, "encode_ns": 90.0, "decode_ns": 30.0},
            "BP-Prune-RMQ (Scalar)": {"ratio": 8.0, "encode_ns": 160.0, "decode_ns": 44.0},
            "BP-Prune-RMQ (SIMD)": {"ratio": 8.0, "encode_ns": 100.0, "decode_ns": 40.0},
            "BP-Prune-RMQ (FastLanes)": {"ratio": 8.0, "encode_ns": 200.0, "decode_ns": 48.0},
            "BP-Prune-RMQ (SIMT)": {"ratio": 8.0, "encode_ns": 84.0, "decode_ns": 28.0},
        },
    ),
]

dataset_mapping = {
    "City-temp.csv": "CT",
    "Wind-Speed.csv": "WS",
    "IR-bio-temp.csv": "IR",
    "PM10-dust.csv": "PM10",
    "Air-pressure.csv": "AP",
    "Dew-point-temp.csv": "DT",
    "Stocks-UK.csv": "SUK",
    "Stocks-USA.csv": "SUA",
    "Stocks-DE.csv": "SDE",
    "Bitcoin-price.csv": "BP",
    "Bird-migration.csv": "BM",
    "Cpu-usage_right.csv": "CPU",
    "Disk-usage.csv": "DISK",
    "Mem-usage.csv": "MEM",
    "Food-price.csv": "FP",
    "electric_vehicle_charging.csv": "VC",
    "Blockchain-tr.csv": "BTR",
    "SSD-bench.csv": "SB",
    "City-lat.csv": "CLT",
    "City-lon.csv": "CLN",
}


def nearest_k_minmax_errors(vals, m, k, log_scale=False):
    if k is None or k < 1:
        return 0.0, 0.0
    a = np.asarray(vals, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if log_scale:
        a = a[a > 0]
    if a.size == 0 or not np.isfinite(m):
        return 0.0, 0.0
    k_eff = min(int(k), a.size)
    dist = np.abs(a - float(m))
    idx = np.argpartition(dist, k_eff - 1)[:k_eff]
    subset = a[idx]
    lo, hi = float(np.min(subset)), float(np.max(subset))
    el = max(0.0, float(m) - lo)
    eu = max(0.0, hi - float(m))
    return el, eu


def _vals_and_yerr_arrays(merged, key, algo):
    n = len(merged)
    vals = []
    el = []
    eu = []
    for j in range(n):
        d = merged[j][1]
        if algo in d and np.isfinite(d[algo][key]):
            v_raw = float(d[algo][key])
            v = v_raw
        else:
            v_raw = np.nan
            v = 0.0
        pool = []
        for jj in range(n):
            if jj == j:
                continue
            dj = merged[jj][1]
            if algo in dj:
                vv = float(dj[algo][key])
                if np.isfinite(vv):
                    pool.append(vv)
        if np.isfinite(v_raw) and len(pool) > 0 and ERRORBAR_NEAREST_K >= 1:
            e_lo, e_hi = nearest_k_minmax_errors(pool, v_raw, ERRORBAR_NEAREST_K)
        else:
            e_lo, e_hi = 0.0, 0.0
        vals.append(v)
        el.append(e_lo)
        eu.append(e_hi)
    return vals, el, eu


def _ratio_bar_y_low_high_per_dataset(merged, n, err_cache, algos, n_impl):
    """Min/max y across all (a) bars including asymmetric error bars."""
    key = "ratio"
    lows, highs = [], []
    for j in range(n):
        d = merged[j][1]
        for i in range(n_impl):
            plain, prune = algos[i], algos[i + n_impl]
            _, elp, eup = err_cache[(key, plain)]
            _, elr, eur = err_cache[(key, prune)]
            if plain in d and np.isfinite(d[plain][key]):
                vp = float(d[plain][key])
                lows.append(vp - elp[j])
                highs.append(vp + eup[j])
            if prune in d and np.isfinite(d[prune][key]):
                vr = float(d[prune][key])
                lows.append(vr - elr[j])
                highs.append(vr + eur[j])
    if not lows:
        return None, None
    return min(lows), max(highs)


def _ratio_bar_y_low_high_avg(avg_ratio, std_ratio, n_impl):
    lows, highs = [], []
    for i in range(n_impl):
        for idx in (i, i + n_impl):
            v = avg_ratio[idx]
            if not np.isfinite(v):
                continue
            v = float(v)
            ye = float(std_ratio[idx])
            if ye > 0:
                lows.append(v - ye)
                highs.append(v + ye)
            else:
                lows.append(v)
                highs.append(v)
    if not lows:
        return None, None
    return min(lows), max(highs)


def _apply_ratio_panel_ylim(ax, y_lo, y_hi):
    """Set (a) ylim so error bars (and caps) stay inside the panel."""
    if (
        y_lo is None
        or y_hi is None
        or not np.isfinite(y_lo)
        or not np.isfinite(y_hi)
        or y_hi < y_lo
    ):
        ax.set_ylim(*RATIO_YLIM_FALLBACK)
        return
    span = y_hi - y_lo
    margin = max(span * RATIO_Y_MARGIN_FRAC, 0.02 * max(abs(y_hi), abs(y_lo), 1.0))
    ax.set_ylim(y_lo - margin, y_hi + margin)


def _row_to_metrics(row):
    try:
        cr = float(row["Compression Ratio"])
        expansion = 1.0 / cr if cr > 0 else np.nan
        enc_mbs = float(row["Encoding Throughput (MB/s)"])
        dec_mbs = float(row["Decoding Throughput (MB/s)"])
        enc_ns = 8000.0 / enc_mbs if enc_mbs > 0 else np.nan
        dec_ns = 8000.0 / dec_mbs if dec_mbs > 0 else np.nan
        return expansion, enc_ns, dec_ns
    except Exception:
        return np.nan, np.nan, np.nan


def load_output_simd():
    data = {abbr: {} for abbr in dataset_mapping.values()}

    def _pick_row(df, csv_names):
        if df is None or "Encoding Algorithm" not in df.columns:
            return None
        algo_col = df["Encoding Algorithm"].astype(str).str.strip().str.strip('"')
        for csv_name in csv_names:
            m = algo_col == csv_name
            if m.any():
                return df.loc[m].iloc[0]
        return None

    for fname, abbr in dataset_mapping.items():
        path = os.path.join(OUTPUT_SIMD_DIR, fname)
        path2 = os.path.join(OUTPUT_SIMD2_DIR, fname)
        df_main = pd.read_csv(path) if os.path.exists(path) else None
        df_simt = pd.read_csv(path2) if os.path.exists(path2) else None
        if df_main is None and df_simt is None:
            continue
        try:
            if df_main is not None and "Encoding Algorithm" not in df_main.columns:
                print(f"  skip {fname}: no Encoding Algorithm in output_simd", file=sys.stderr)
                continue
            for csv_names, disp in ALGO_CSV_CANDIDATES:
                source = df_simt if disp in SIMT_LEGEND_LABELS else df_main
                row_time = _pick_row(source, csv_names)
                if row_time is None:
                    continue
                row_ratio = _pick_row(df_simt, csv_names)
                if row_ratio is not None:
                    ratio, _, _ = _row_to_metrics(row_ratio)
                else:
                    ratio, _, _ = _row_to_metrics(row_time)
                _, enc_ns, dec_ns = _row_to_metrics(row_time)
                data[abbr][disp] = {"ratio": ratio, "encode_ns": enc_ns, "decode_ns": dec_ns}
        except Exception as e:
            print(f"  skip {fname}: {e}", file=sys.stderr)
    return data


def merge_from_simd(simd_data):
    abbrs = sorted(dataset_mapping.values())
    out = []
    for abbr in abbrs:
        if abbr not in simd_data or not simd_data[abbr]:
            continue
        d = {}
        for algo in ALGOS:
            if algo in simd_data[abbr]:
                d[algo] = simd_data[abbr][algo]
        if d:
            out.append((abbr, d))
    return out


def plot_simd():
    simd_data = load_output_simd()
    merged = merge_from_simd(simd_data)
    if not merged:
        print("No CSV data; using example data.", file=sys.stderr)
        merged = EXAMPLE_DATA

    n_algos = len(ALGOS)
    if merged:
        n_found = max(len(d[1]) for d in merged)
        if n_found < n_algos:
            print(
                f"  Warning: expected {n_algos} algorithms; found at most {n_found}.",
                file=sys.stderr,
            )

    datasets = [m[0] for m in merged]
    n = len(datasets)
    x = np.arange(n, dtype=float)
    bar_w = _BAR_W
    fig_w = 1.2 * min(22, 10 + max(n, 1) * 0.45)
    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "axes.titlesize": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "legend.fontsize": FONTSIZE,
        }
    )
    fontsize = FONTSIZE
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 4))
    plt.subplots_adjust(wspace=0.35)

    def _metric(d, algo, key):
        if algo not in d or np.isnan(d[algo][key]):
            return 0.0
        return float(d[algo][key])

    def _asymmetric_yerr(el_j, eu_j):
        if el_j <= 0 and eu_j <= 0:
            return None
        return np.array([[el_j], [eu_j]])

    err_cache = {}
    for _key in ("ratio", "encode_ns", "decode_ns"):
        for _algo in ALGOS:
            err_cache[(_key, _algo)] = _vals_and_yerr_arrays(merged, _key, _algo)

    def _bar_with_err(ax, x0, h, w, yerr, **kwargs):
        kw = dict(x=x0, height=h, width=w, zorder=3, **kwargs)
        if yerr is not None:
            kw["yerr"] = yerr
            kw["ecolor"] = ERRORBAR_ECOLOR
            kw["capsize"] = ERRORBAR_CAPSIZE
            kw["error_kw"] = ERR_KW_BAR
        ax.bar(**kw)

    def _draw_grouped_bars(ax, key):
        for j in range(n):
            xc = float(x[j])
            d = merged[j][1]
            for i in range(N_IMPL):
                c = GROUP_COLORS[i]
                plain, prune = ALGOS[i], ALGOS[i + N_IMPL]
                x_left = xc + _GROUP_CENTER_OFF[i] - _BAR_PAIR_HALF
                x_right = xc + _GROUP_CENTER_OFF[i] + _BAR_PAIR_HALF
                vp = _metric(d, plain, key)
                vr = _metric(d, prune, key)
                _, elp, eup = err_cache[(key, plain)]
                _, elr, eur = err_cache[(key, prune)]
                yerr_p = _asymmetric_yerr(elp[j], eup[j])
                yerr_r = _asymmetric_yerr(elr[j], eur[j])
                _bar_with_err(
                    ax,
                    x_left,
                    vp,
                    bar_w,
                    yerr_p,
                    color=c,
                    edgecolor="none",
                    linewidth=0,
                )
                _bar_with_err(
                    ax,
                    x_right,
                    vr,
                    bar_w,
                    yerr_r,
                    color=c,
                    edgecolor="white",
                    linewidth=1.0,
                    hatch="///",
                )

    ax1 = axes[0]
    _draw_grouped_bars(ax1, "ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels([])
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax1.set_xlim(-0.55, max(n - 1, 0) + 0.55)
    ax1.set_ylabel("Compression Ratio", fontsize=fontsize)
    ax1.set_title("(a) Compression Ratio", fontsize=fontsize)
    ax1.tick_params(axis="y", labelsize=fontsize)
    _rl, _rh = _ratio_bar_y_low_high_per_dataset(merged, n, err_cache, ALGOS, N_IMPL)
    _apply_ratio_panel_ylim(ax1, _rl, _rh)

    ax2 = axes[1]
    _draw_grouped_bars(ax2, "encode_ns")
    ax2.set_xticks(x)
    ax2.set_xticklabels([])
    ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax2.set_xlim(-0.55, max(n - 1, 0) + 0.55)
    ax2.set_ylabel("Time (ns/point)", fontsize=fontsize)
    ax2.set_title("(b) Compression Time", fontsize=fontsize)
    ax2.tick_params(axis="y", labelsize=fontsize)

    ax3 = axes[2]
    _draw_grouped_bars(ax3, "decode_ns")
    ax3.set_xticks(x)
    ax3.set_xticklabels([])
    ax3.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax3.set_xlim(-0.55, max(n - 1, 0) + 0.55)
    ax3.set_ylabel("Time (ns/point)", fontsize=fontsize)
    ax3.set_title("(c) Decompression Time", fontsize=fontsize)
    ax3.tick_params(axis="y", labelsize=fontsize)

    # Legend: row-major, left-to-right — row1 = BP (Scalar→SIMT) solid; row2 = Prune-RMQ hatched.
    legend_handles = []
    for i in range(N_IMPL):
        c = GROUP_COLORS[i]
        legend_handles.append(Patch(facecolor=c, edgecolor="none", linewidth=0, label=ALGOS[i]))
    for i in range(N_IMPL):
        c = GROUP_COLORS[i]
        legend_handles.append(
            Patch(facecolor=c, edgecolor="white", linewidth=1.0, hatch="///", label=ALGOS[i + N_IMPL])
        )
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=fontsize,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        labelspacing=0.1,
        handletextpad=0.1,
        columnspacing=0.1,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    out_dir = os.path.join(BASE, "figure_for_paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "simd_compare.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(os.path.splitext(out_path)[0] + ".eps", format="eps", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    avg_ratio = []
    avg_encode_ns = []
    avg_decode_ns = []
    std_ratio = []
    std_encode_ns = []
    std_decode_ns = []
    for algo in ALGOS:
        ratios = [d[algo]["ratio"] for _, d in merged if algo in d and not np.isnan(d[algo]["ratio"])]
        encodes = [d[algo]["encode_ns"] for _, d in merged if algo in d and not np.isnan(d[algo]["encode_ns"])]
        decodes = [d[algo]["decode_ns"] for _, d in merged if algo in d and not np.isnan(d[algo]["decode_ns"])]
        avg_ratio.append(np.mean(ratios) if ratios else np.nan)
        avg_encode_ns.append(np.mean(encodes) if encodes else np.nan)
        avg_decode_ns.append(np.mean(decodes) if decodes else np.nan)
        std_ratio.append(float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0)
        std_encode_ns.append(float(np.std(encodes, ddof=1)) if len(encodes) > 1 else 0.0)
        std_decode_ns.append(float(np.std(decodes, ddof=1)) if len(decodes) > 1 else 0.0)

    fig2_w = 14
    fig2, axes2 = plt.subplots(1, 3, figsize=(fig2_w, 4))
    plt.subplots_adjust(wspace=0.35)
    x_avg = np.arange(N_IMPL, dtype=float)
    width_pair = 0.35

    def _draw_avg_grouped(ax, vals, stds):
        for i in range(N_IMPL):
            c = GROUP_COLORS[i]
            vp = float(np.nan_to_num(vals[i], nan=0.0))
            vr = float(np.nan_to_num(vals[i + N_IMPL], nan=0.0))
            ye_p = float(stds[i])
            ye_r = float(stds[i + N_IMPL])
            _bar_with_err(
                ax,
                x_avg[i] - width_pair / 2,
                vp,
                width_pair,
                ye_p if ye_p > 0 else None,
                color=c,
                edgecolor="none",
                linewidth=0,
            )
            _bar_with_err(
                ax,
                x_avg[i] + width_pair / 2,
                vr,
                width_pair,
                ye_r if ye_r > 0 else None,
                color=c,
                edgecolor="white",
                linewidth=1.0,
                hatch="///",
            )

    ax1 = axes2[0]
    _draw_avg_grouped(ax1, avg_ratio, std_ratio)
    ax1.set_xticks([])
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax1.set_xlim(-0.5, N_IMPL - 0.5)
    ax1.set_ylabel("Compression Ratio", fontsize=fontsize)
    ax1.set_title("(a) Compression Ratio", fontsize=fontsize)
    ax1.tick_params(axis="y", labelsize=fontsize)
    _rl2, _rh2 = _ratio_bar_y_low_high_avg(avg_ratio, std_ratio, N_IMPL)
    _apply_ratio_panel_ylim(ax1, _rl2, _rh2)

    ax2 = axes2[1]
    _draw_avg_grouped(ax2, avg_encode_ns, std_encode_ns)
    ax2.set_xticks([])
    ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax2.set_xlim(-0.5, N_IMPL - 0.5)
    ax2.set_ylabel("Time (ns/point)", fontsize=fontsize)
    ax2.set_title("(b) Compression Time", fontsize=fontsize)
    ax2.tick_params(axis="y", labelsize=fontsize)

    ax3 = axes2[2]
    _draw_avg_grouped(ax3, avg_decode_ns, std_decode_ns)
    ax3.set_xticks([])
    ax3.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax3.set_xlim(-0.5, N_IMPL - 0.5)
    ax3.set_ylabel("Time (ns/point)", fontsize=fontsize)
    ax3.set_title("(c) Decompression Time", fontsize=fontsize)
    ax3.tick_params(axis="y", labelsize=fontsize)

    fig2.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=fontsize,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        labelspacing=0.1,
        handletextpad=0.1,
        columnspacing=0.1,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    out_path_avg = os.path.join(out_dir, "simd_compare_avg.png")
    plt.savefig(out_path_avg, dpi=300, bbox_inches="tight")
    plt.savefig(os.path.splitext(out_path_avg)[0] + ".eps", format="eps", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path_avg}")


if __name__ == "__main__":
    plot_simd()
