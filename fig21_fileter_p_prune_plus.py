import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def collect_filter_counts(input_dir, algorithm_filter=None):
    files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    results = {}
    for fp in files:
        try:
            df = pd.read_csv(fp, encoding='utf-8', dtype=str)
        except Exception:
            try:
                df = pd.read_csv(fp, encoding='latin1', dtype=str)
            except Exception:
                continue
        if df.empty:
            continue
        cols = [c.strip() for c in df.columns]
        df.columns = cols
        if 'Filter Count' not in df.columns:
            continue
        if algorithm_filter and 'Encoding Algorithm' in df.columns:
            df = df[df['Encoding Algorithm'] == algorithm_filter]
        try:
            fc = pd.to_numeric(df['Filter Count'], errors='coerce')
            fc = fc.dropna()
            if fc.empty:
                continue
            mean_val = float(fc.mean())
            pct = mean_val / 1024.0 * 100.0
            value = pct
        except Exception:
            vals = []
            for v in df['Filter Count'].astype(str).tolist():
                s = v.strip()
                if not s:
                    continue
                try:
                    vals.append(float(s))
                except Exception:
                    continue
            if not vals:
                continue
            mean_val = sum(vals) / len(vals)
            value = mean_val / 1024.0 * 100.0
        name = os.path.splitext(os.path.basename(fp))[0]
        results[name] = value
    return results

def plot_bar(results, outpath, title='Count of pruned pack sizes in each dataset'):
    if not results:
        print('No data found to plot.')
        return
    items = sorted(results.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*items)
    fontsize = 20
    title_fs = fontsize
    label_fs = fontsize
    tick_fs = fontsize
    annot_fs = fontsize
    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(values)), values, color='C0')
    plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.xlabel('Dataset', fontsize=label_fs)
    plt.ylabel('Percentage (% of 1024)', fontsize=label_fs, y=0.4)
    plt.title(title, fontsize=title_fs)
    top = max(values) if values else 0
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + top * 0.01, f'{v:.1f}%', ha='center', va='bottom', rotation=30, fontsize=annot_fs)
    if top <= 100:
        plt.ylim(0, max(100, top * 1.05))
    plt.tight_layout()
    outdir = os.path.dirname(outpath) or '.'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(outpath, dpi=400, bbox_inches='tight')
    try:
        eps_path = os.path.splitext(outpath)[0] + '.eps'
        plt.savefig(eps_path, format='eps', dpi=400, bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    print(f'Saved plot: {outpath}')

def sprintz_collect_filter_counts(input_dir, algorithm_filter=None):
    files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    results = {}
    for fp in files:
        try:
            df = pd.read_csv(fp, encoding='utf-8', dtype=str)
        except Exception:
            try:
                df = pd.read_csv(fp, encoding='latin1', dtype=str)
            except Exception:
                continue
        if df.empty:
            continue
        cols = [c.strip() for c in df.columns]
        df.columns = cols
        if 'Filter Count' not in df.columns:
            continue
        if algorithm_filter and 'Encoding Algorithm' in df.columns:
            df = df[df['Encoding Algorithm'] == algorithm_filter]
        try:
            fc = pd.to_numeric(df['Filter Count'], errors='coerce')
            fc = fc.dropna()
            if fc.empty:
                continue
            mean_val = float(fc.mean())
            pct = mean_val / 1024.0 * 100.0
            value = pct
        except Exception:
            vals = []
            for v in df['Filter Count'].astype(str).tolist():
                s = v.strip()
                if not s:
                    continue
                try:
                    vals.append(float(s))
                except Exception:
                    continue
            if not vals:
                continue
            mean_val = sum(vals) / len(vals)
            value = mean_val / 1024.0 * 100.0
        name = os.path.splitext(os.path.basename(fp))[0]
        results[name] = value
    return results

def sprintz_plot_bar(results, outpath, title='Pruning rate of pack sizes on datasets after Sprintz'):
    if not results:
        print('No data found to plot.')
        return
    items = sorted(results.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*items)
    fontsize = 20
    title_fs = fontsize
    label_fs = fontsize
    tick_fs = fontsize
    annot_fs = fontsize
    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(values)), values, color='C0')
    plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)
    plt.ylabel('Percentage (% of 1024)', fontsize=label_fs, y=0.4)
    plt.title(title, fontsize=title_fs)
    top = max(values) if values else 0
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + top * 0.01, f'{v:.1f}%', ha='center', va='bottom', rotation=30, fontsize=annot_fs)
    if top <= 100:
        plt.ylim(0, max(100, top * 1.05))
    plt.tight_layout()
    outdir = os.path.dirname(outpath) or '.'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(outpath, dpi=400, bbox_inches='tight')
    try:
        eps_path = os.path.splitext(outpath)[0] + '.eps'
        plt.savefig(eps_path, format='eps', dpi=400, bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    print(f'Saved plot: {outpath}')
dataset_mapping = {'City-temp.csv': 'CT', 'Wind-Speed.csv': 'WS', 'IR-bio-temp.csv': 'IR', 'PM10-dust.csv': 'PM10', 'Air-pressure.csv': 'AP', 'Dew-point-temp.csv': 'DT', 'Stocks-UK.csv': 'SUK', 'Stocks-USA.csv': 'SUA', 'Stocks-DE.csv': 'SDE', 'Bird-migration.csv': 'BM', 'Food-price.csv': 'FP', 'Blockchain-tr.csv': 'BTR', 'City-lat.csv': 'CLT', 'City-lon.csv': 'CLN', 'Cyber-Vehicle.csv': 'CV', 'TY-Fuel.csv': 'TF', 'TY-Transport.csv': 'TT'}

def plot_two_bars(results1, results2, outpath, title1='(a) Pruning rate of pack sizes on datasets (BP)', title2='(b) Pruning rate of pack sizes on datasets (Sprintz)'):
    if not results1 and (not results2):
        print('No data found to plot.')
        return
    results1 = {k: v for k, v in results1.items() if k in dataset_mapping or f'{k}.csv' in dataset_mapping}
    results2 = {k: v for k, v in results2.items() if k in dataset_mapping or f'{k}.csv' in dataset_mapping}
    fontsize = 20
    title_fs = fontsize
    label_fs = fontsize
    tick_fs = fontsize
    annot_fs = fontsize
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def map_label(name):
        if name in dataset_mapping:
            return dataset_mapping[name]
        alt = f'{name}.csv'
        if alt in dataset_mapping:
            return dataset_mapping[alt]
        return name
    if results1:
        items1 = sorted(results1.items(), key=lambda x: x[1], reverse=True)
        names1, values1 = zip(*items1)
        labels1 = [map_label(n) for n in names1]
    else:
        names1, values1, labels1 = ([], [], [])
    bars = axes[0].bar(range(len(values1)), values1, color='C0')
    axes[0].set_xticks(range(len(names1)))
    axes[0].set_xticklabels(labels1, rotation=90, ha='center', va='top', fontsize=tick_fs)
    axes[0].tick_params(axis='y', labelsize=tick_fs)
    axes[0].set_xlabel('Dataset', fontsize=label_fs)
    axes[0].set_ylabel('Percentage (% of 1024)', fontsize=label_fs, y=0.3)
    axes[0].set_title(title1, fontsize=title_fs)
    top = max(values1) if values1 else 0
    for b, v in zip(bars, values1):
        axes[0].text(b.get_x() + b.get_width() / 2, v + top * 0.01, f'{v:.1f}', ha='center', va='bottom', rotation=90, fontsize=annot_fs)
    if top <= 100:
        axes[0].set_ylim(0, max(100, top * 1.05))
    if results2:
        items2 = sorted(results2.items(), key=lambda x: x[1], reverse=True)
        names2, values2 = zip(*items2)
        labels2 = [map_label(n) for n in names2]
    else:
        names2, values2, labels2 = ([], [], [])
    bars2 = axes[1].bar(range(len(values2)), values2, color='C1')
    axes[1].set_xticks(range(len(names2)))
    axes[1].set_xticklabels(labels2, rotation=90, ha='center', va='top', fontsize=tick_fs)
    axes[1].tick_params(axis='y', labelsize=tick_fs)
    axes[1].set_xlabel('Dataset', fontsize=label_fs)
    axes[1].set_ylabel('Percentage (% of 1024)', fontsize=label_fs, y=0.3)
    axes[1].set_title(title2, fontsize=title_fs, x=0.45)
    top2 = max(values2) if values2 else 0
    for b, v in zip(bars2, values2):
        axes[1].text(b.get_x() + b.get_width() / 2, v + top2 * 0.01, f'{v:.1f}', ha='center', va='bottom', rotation=90, fontsize=annot_fs)
    if top2 <= 100:
        axes[1].set_ylim(0, max(100, top2 * 1.05))
    plt.tight_layout()
    outdir = os.path.dirname(outpath) or '.'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(outpath, dpi=400, bbox_inches='tight')
    try:
        eps_path = os.path.splitext(outpath)[0] + '.eps'
        plt.savefig(eps_path, format='eps', dpi=400, bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    print(f'Saved combined plot: {outpath}')

def main():
    parser = argparse.ArgumentParser(description='Plot Filter Count per dataset for BP and Sprintz on a combined figure')
    parser.add_argument('--input1', default='output_BP_filters_plus', help='Directory containing BP CSV files')
    parser.add_argument('--input2', default='output_Sprintz_filters_plus', help='Directory containing Sprintz CSV files')
    parser.add_argument('--output', '-o', default='figure_for_paper/prune_plus_filters_count_bar_combined.png', help='Output image path')
    parser.add_argument('--algorithm1', default='BP+RMQ+Prune', help='Encoding Algorithm to filter by for BP')
    parser.add_argument('--algorithm2', default='Sprintz-RMQ-Prune', help='Encoding Algorithm to filter by for Sprintz')
    args = parser.parse_args()
    results_bp = collect_filter_counts(args.input1, algorithm_filter=args.algorithm1)
    if not results_bp:
        results_bp = collect_filter_counts(args.input1, algorithm_filter=None)
    results_sp = sprintz_collect_filter_counts(args.input2, algorithm_filter=args.algorithm2)
    if not results_sp:
        results_sp = sprintz_collect_filter_counts(args.input2, algorithm_filter=None)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plot_two_bars(results_bp, results_sp, args.output, title1='(a) Pruning rate after BP', title2='(b) Pruning rate after Sprintz')
if __name__ == '__main__':
    main()
