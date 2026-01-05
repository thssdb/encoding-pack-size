import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_mapping = {
    # 时间序列数据集
    'City-temp.csv': 'CT',
    'Wind-Speed.csv': 'WS',
    'IR-bio-temp.csv': 'IR',
    'PM10-dust.csv': 'PM10',
    'Air-pressure.csv': 'AP',
    'Dew-point-temp.csv': 'DT',
    'Stocks-UK.csv': 'SUK',
    'Stocks-USA.csv': 'SUA',
    'Stocks-DE.csv': 'SDE',
    'Bitcoin-price.csv': 'BP',
    'Bird-migration.csv': 'BM',
    # 'Cpu-usage_right.csv': 'CPU',
    # 'Disk-usage.csv': 'DISK',
    # 'Mem-usage.csv': 'MEM',
    
    # 非时间序列数据集
    'Food-price.csv': 'FP',
    'electric_vehicle_charging.csv': 'VC',
    'Blockchain-tr.csv': 'BTR',
    'SSD-bench.csv': 'SB',
    'City-lat.csv': 'CLT',
    'City-lon.csv': 'CLN',
}
def read_ratio_file(primary='camel_ratio.xlsx'):
    if os.path.exists(primary):
        return pd.read_excel(primary, index_col=0)
    return None


def plot_bp_vs_bpall(df, alg1='BP', alg2='BP (Prune-RMQ)', outpath='figure_for_paper/bp_vs_bpall.png'):
    if alg1 not in df.index and alg2 not in df.index:
        print(f'Neither {alg1} nor {alg2} found in input file. Available rows: {list(df.index)}')
        return False

    cols = list(df.columns)
    # gather values, use NaN for missing so bars can be skipped/zeroed
    vals1 = [float(1/df.at[alg1, c]) if (alg1 in df.index and pd.notna(df.at[alg1, c])) else np.nan for c in cols]
    vals2 = [float(1/df.at[alg2, c]) if (alg2 in df.index and pd.notna(df.at[alg2, c])) else np.nan for c in cols]

    x = np.arange(len(cols))
    width = 0.35

    # font sizes
    title_fs = 16
    label_fs = 16
    tick_fs = 16
    legend_fs = 16
    annot_fs = 16

    plt.figure(figsize=(max(10, len(cols) * 0.28), 6))
    bar1 = plt.bar(x - width/2, np.nan_to_num(vals1, nan=0.0), width, label=alg1, color='C0')
    bar2 = plt.bar(x + width/2, np.nan_to_num(vals2, nan=0.0), width, label=alg2, color='C1')

    # xticks: use column names (dataset abbreviations)
    plt.xticks(x, cols, rotation=30, ha='right', fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.ylabel('Compression Ratio', fontsize=label_fs)
    plt.title(f'Improved compression ratio of each dataset', fontsize=title_fs)
    plt.legend(fontsize=legend_fs)

    # # annotate values on bars (skip zeros where NaN was used)
    # def annotate(bars, values):
    #     for b, v in zip(bars, values):
    #         if np.isnan(v):
    #             continue
    #         plt.text(b.get_x() + b.get_width()/2, v + 0.005 * max(np.nanmax(vals1), np.nanmax(vals2)), f'{v:.3f}',
    #                  ha='center', va='bottom', rotation=30, fontsize=10)

    # annotate(bar1, vals1)
    # annotate(bar2, vals2)

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


def plot_percent_improvements(df, outpath='figure_for_paper/improvement_bp_sprintz.png'):
    # compute inverse ratios (higher is better) then percent improvements
    cols = list(df.columns)
    # cols中不在dataset_mapping中value的列名将被过滤掉
    cols = [c for c in cols if c in dataset_mapping.values()]

    def inv(val):
        try:
            return 1.0 / float(val)
        except Exception:
            return np.nan

    df 

    bp_inv = [inv(df.at['BP', c]) if ('BP' in df.index and pd.notna(df.at['BP', c])) else np.nan for c in cols]
    bp_all_inv = [inv(df.at['BP (Prune-RMQ)', c]) if ('BP (Prune-RMQ)' in df.index and pd.notna(df.at['BP (Prune-RMQ)', c])) else np.nan for c in cols]

    sp_inv = [inv(df.at['Sprintz', c]) if ('Sprintz' in df.index and pd.notna(df.at['Sprintz', c])) else np.nan for c in cols]
    sp_all_inv = [inv(df.at['Sprintz (Prune-RMQ)', c]) if ('Sprintz (Prune-RMQ)' in df.index and pd.notna(df.at['Sprintz (Prune-RMQ)', c])) else np.nan for c in cols]

    # percent improvement: (inv_all / inv_base - 1) * 100
    bp_impr = [((a / b - 1.0) * 100.0) if (not np.isnan(a) and not np.isnan(b) and b != 0) else np.nan for a, b in zip(bp_all_inv, bp_inv)]
    sp_impr = [((a / b - 1.0) * 100.0) if (not np.isnan(a) and not np.isnan(b) and b != 0) else np.nan for a, b in zip(sp_all_inv, sp_inv)]

    x = np.arange(len(cols))
    width = 0.35

    # font sizes
    title_fs = 16
    label_fs = 16
    tick_fs = 16
    annot_fs = 14

    fig_width = 8
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 8.3), sharey=False)

    # Left: BP improvement
    ax0 = axes[0]
    bars0 = ax0.bar(x, np.nan_to_num(bp_impr, nan=0.0), width, color='C0')
    ax0.set_xticks(x)
    ax0.set_xticklabels(cols, rotation=45, ha='right', fontsize=tick_fs)
    ax0.tick_params(axis='y', labelsize=tick_fs)
    ax0.set_ylim(0,20)
    ax0.set_xlabel('Dataset', fontsize=label_fs)
    ax0.set_ylabel('Improvement (%)', fontsize=label_fs)
    ax0.set_title('(a) Pack size optimized Bit-packing vs vanilla Bit-packing', fontsize=title_fs)

    top0 = max([v for v in bp_impr if not np.isnan(v)] or [0])
    # if top0 <= 100:
    #     ax0.set_ylim(0, max(100, top0 * 1.05))

    for b, v in zip(bars0, bp_impr):
        if np.isnan(v):
            continue
        ax0.text(b.get_x() + b.get_width() / 2, v + (top0 * 0.01 if top0 > 0 else 1.0), f"{v:.1f}%",
                 ha='center', va='bottom', rotation=15, fontsize=annot_fs)

    # Right: Sprintz improvement
    ax1 = axes[1]
    bars1 = ax1.bar(x, np.nan_to_num(sp_impr, nan=0.0), width, color='C1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cols, rotation=45, ha='right', fontsize=tick_fs)
    ax1.tick_params(axis='y', labelsize=tick_fs)
    ax1.set_ylim(0,20)
    ax1.set_xlabel('Dataset', fontsize=label_fs)
    ax1.set_ylabel('Improvement (%)', fontsize=label_fs)
    ax1.set_title('(b) Pack size optimized Sprintz vs vanilla Sprintz', fontsize=title_fs)

    top1 = max([v for v in sp_impr if not np.isnan(v)] or [0])
    # if top1 <= 100:
    #     ax1.set_ylim(0, max(100, top1 * 1.05))

    for b, v in zip(bars1, sp_impr):
        if np.isnan(v):
            continue
        ax1.text(b.get_x() + b.get_width() / 2, v + (top1 * 0.01 if top1 > 0 else 1.0), f"{v:.1f}%",
                 ha='center', va='bottom', rotation=15, fontsize=annot_fs)

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
    # parser.add_argument('--fallback', '-f', default='compare_camel/camel_ratio4.xlsx', help='Fallback Excel file')
    parser.add_argument('--alg1', default='BP', help='First algorithm row name')
    parser.add_argument('--alg2', default='BP (Prune-RMQ)', help='Second algorithm row name')
    parser.add_argument('--output', '-o', default='figure_for_paper/bp_vs_bpall.png', help='Output image path')
    args = parser.parse_args()

    df = read_ratio_file(args.input)
    if df is None:
        print('Could not find input files. Produce camel_ratio.xlsx first (run combine_results.py).')
        return

    # remove helper/aggregate column if present
    if 'avg_ratio' in df.columns:
        df = df.drop(columns=['avg_ratio'])

    # produce percent-improvement figure (BP(Prune-RMQ) vs BP and Sprintz(Prune-RMQ) vs Sprintz)
    plot_percent_improvements(df, outpath='figure_for_paper/improvement_bp_sprintz.png')


if __name__ == '__main__':
    main()
