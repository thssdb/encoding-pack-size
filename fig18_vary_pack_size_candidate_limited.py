"""
Plot pack size candidate limited (1~16, 1~32, ..., 1~1024) vs full search (OptimalPackSizePruneRMQTest).

- 折线: OptimalPackSizePruneRMQLimitedTest 结果 (output_BP_Prune_RMQ_limited)
- 横线: OptimalPackSizePruneRMQTest 结果 (output_BP_Prune_all_no8)

Timing in Java (AllNo8PacksizeOptimal): per-chunk encode averaged over BENCH_ENCODE_REPEATS (1000),
decode over BENCH_DECODE_REPEATS (2000); CSV stores throughput (MB/s), converted here to ns/point.

Each subplot: one line for pre-defined max pack-size candidates (median across datasets) and a dashed
horizontal line for the all-pack-sizes baseline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = "/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size"
OUT_DIR = os.path.join(BASE, "figure_for_paper")

LIMITED_DIR = os.path.join(BASE, "output_BP_Prune_RMQ_limited")
FULL_DIR = os.path.join(BASE, "output_BP_Prune_all_no8")

# Datasets to include (same as other figs)
DATASET_FILES = [
    'City-temp.csv',
    'Wind-Speed.csv',
    'IR-bio-temp.csv',
    'PM10-dust.csv',
    'Air-pressure.csv',
    'Dew-point-temp.csv',
    'Stocks-UK.csv',
    'Stocks-USA.csv',
    'Stocks-DE.csv',
    'Bitcoin-price.csv',
    'Bird-migration.csv',
    'Food-price.csv',
    # 'electric_vehicle_charging.csv',
    'Blockchain-tr.csv',
    # 'SSD-bench.csv',
    # 'City-lat.csv',
    # 'City-lon.csv',
    # 'Cyber-Vehicle.csv',
    'TY-Fuel.csv',
    'TY-Transport.csv',
]

MAX_PACK_SIZES = [16, 32, 64, 128, 256, 512, 1024]

# fmt= is not valid for Line2D kwargs; use marker + linestyle.
LINE_KW = dict(marker="o", linestyle="-", color="#1f77b4", linewidth=2, markersize=8)
HLINE_KW = dict(color="#d62728", linestyle="--", linewidth=2)


def load_limited_data():
    """Load limited pack size results: expansion and enc/dec ns/pt (median across datasets per max pack)."""
    ratio_by_max = {m: [] for m in MAX_PACK_SIZES}
    enc_time_by_max = {m: [] for m in MAX_PACK_SIZES}
    dec_time_by_max = {m: [] for m in MAX_PACK_SIZES}

    for fname in DATASET_FILES:
        path = os.path.join(LIMITED_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            try:
                mp = int(row["Max Pack Size"])
                if mp not in ratio_by_max:
                    continue
                ratio = float(row["Compression Ratio"])
                expansion = 1.0 / ratio if ratio > 0 else np.nan  # > 1
                enc_mbps = float(row["Encoding Throughput (MB/s)"])
                dec_mbps = float(row["Decoding Throughput (MB/s)"])
                # time ns/point = 8000 / throughput_MB_per_s
                enc_ns = 8000.0 / enc_mbps if enc_mbps > 0 else np.nan
                dec_ns = 8000.0 / dec_mbps if dec_mbps > 0 else np.nan

                ratio_by_max[mp].append(expansion)
                enc_time_by_max[mp].append(enc_ns)
                dec_time_by_max[mp].append(dec_ns)
            except (KeyError, ValueError, TypeError):
                continue

    ratio_med = [np.nanmedian(ratio_by_max[m]) if ratio_by_max[m] else np.nan for m in MAX_PACK_SIZES]
    enc_med = [np.nanmedian(enc_time_by_max[m]) if enc_time_by_max[m] else np.nan for m in MAX_PACK_SIZES]
    dec_med = [np.nanmedian(dec_time_by_max[m]) if dec_time_by_max[m] else np.nan for m in MAX_PACK_SIZES]

    return ratio_med, enc_med, dec_med


def load_full_data():
    """Load full search results (OptimalPackSizePruneRMQTest): single mean per metric."""
    expansions = []
    enc_times = []
    dec_times = []

    for fname in DATASET_FILES:
        path = os.path.join(FULL_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0]
        # Columns: Encoding Time, Decoding Time - but values are actually throughput (MB/s)
        enc_mbps = float(row["Encoding Time"])
        dec_mbps = float(row["Decoding Time"])
        ratio = float(row["Compression Ratio"])
        expansion = 1.0 / ratio if ratio > 0 else np.nan
        enc_ns = 8000.0 / enc_mbps if enc_mbps > 0 else np.nan
        dec_ns = 8000.0 / dec_mbps if dec_mbps > 0 else np.nan

        expansions.append(expansion)
        enc_times.append(enc_ns)
        dec_times.append(dec_ns)

    r_med = float(np.nanmedian(expansions)) if expansions else np.nan
    e_med = float(np.nanmedian(enc_times)) if enc_times else np.nan
    d_med = float(np.nanmedian(dec_times)) if dec_times else np.nan
    return r_med, e_med, d_med


def plot_pack_size_limited_vs_full():
    ratio_limited, enc_limited, dec_limited = load_limited_data()
    ratio_full, enc_full, dec_full = load_full_data()

    fontsize = 22
    plt.rcParams.update({"font.size": fontsize})

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.3)

    x = np.array(MAX_PACK_SIZES)
    xtick_labels = [f"$2^{{{int(np.log2(m))}}}$" for m in x]

    # Last x aligns with full-search baseline (horizontal reference)
    ratio_plot = np.array(ratio_limited, dtype=float)
    enc_plot = np.array(enc_limited, dtype=float)
    dec_plot = np.array(dec_limited, dtype=float)
    if ratio_plot.size > 0 and not np.isnan(ratio_full):
        ratio_plot[-1] = ratio_full
    if enc_plot.size > 0 and not np.isnan(enc_full):
        enc_plot[-1] = enc_full
    if dec_plot.size > 0 and not np.isnan(dec_full):
        dec_plot[-1] = dec_full

    limited_label = "Pre-defined Pack Sizes"
    full_label = "BP-Prune-RMQ"

    # (a) Compression ratio
    ax = axes[0]
    ax.plot(x, ratio_plot, **LINE_KW, label=limited_label)
    ax.axhline(y=ratio_full, **HLINE_KW, label=full_label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel(r"Threshold $\lambda$", fontsize=fontsize)
    ax.set_ylabel("Compression Ratio", fontsize=fontsize)
    ax.set_title("(a) Compression Ratio", fontsize=fontsize,x=0.3)
    ax.tick_params(axis="both", labelsize=fontsize)

    # (b) Compression time (ns/point)
    ax = axes[1]
    ax.plot(x, enc_plot, **LINE_KW, label=limited_label)
    ax.axhline(y=enc_full, **HLINE_KW, label=full_label, zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel(r"Threshold $\lambda$", fontsize=fontsize)
    ax.set_ylabel("Time (ns/point)", fontsize=fontsize,y=0.4)
    ax.set_title("(b) Compression Time", fontsize=fontsize,x=0.3)
    ax.tick_params(axis="both", labelsize=fontsize)

    # (c) Decompression time (ns/point)
    ax = axes[2]
    ax.plot(x, dec_plot, **LINE_KW, label=limited_label)
    ax.axhline(y=dec_full, **HLINE_KW, label=full_label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel(r"Threshold $\lambda$", fontsize=fontsize)
    ax.set_ylabel("Time (ns/point)", fontsize=fontsize,y=0.4)
    ax.set_title("(c) Decompression Time", fontsize=fontsize,x=0.3)
    ax.tick_params(axis="both", labelsize=fontsize)

    handles = [
        Line2D(
            [0],
            [0],
            color=LINE_KW["color"],
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=8,
            label=limited_label,
        ),
        Line2D([0], [0], color=HLINE_KW["color"], linestyle="--", linewidth=2, label=full_label),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        fontsize=fontsize,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "pack_size_limited_vs_full.png")
    out_eps = os.path.join(OUT_DIR, "pack_size_limited_vs_full.eps")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_eps, format="eps", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_png}")
    print(f"Limited ratio (median): {ratio_limited}")
    print(f"Full ratio: {ratio_full:.6f}")
    print(f"Limited enc time (ns/pt, median): {enc_limited}")
    print(f"Full enc time: {enc_full:.2f} ns/pt")
    print(f"Limited dec time (ns/pt, median): {dec_limited}")
    print(f"Full dec time: {dec_full:.2f} ns/pt")


if __name__ == "__main__":
    plot_pack_size_limited_vs_full()
