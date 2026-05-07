import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
plt.rcParams.update({'font.size': 20})

def fig_of_cost_values_bitwidth_in_chunk(csv_dir, output_dir, chunk_size=1024):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_cost.csv')]
    for csv_file in csv_files:
        if not csv_file.startswith('PM10-dust'):
            continue
        path = os.path.join(csv_dir, csv_file)
        print(f'Chunk plotting: {csv_file}')
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f'  Failed to read {csv_file}: {e}')
            continue
        if 'pack size' not in df.columns or 'cost' not in df.columns:
            print(f"  Skipping {csv_file}: missing 'packsize' or 'cost' columns")
            continue
        n = len(df)
        num_chunks = 1
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(n, (i + 1) * chunk_size)
            sub = df.iloc[start:end]
            if sub.empty:
                continue
            sub_sorted = sub.sort_values(by='pack size')
            x = sub_sorted['pack size'].values
            y = sub_sorted['cost'].values
            y1 = sub_sorted['bitwidth_cost'].values
            y2 = sub_sorted['value_cost'].values
            fontsize = 16
            ax = plt.figure(figsize=(8, 4))
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#008000', '#FFC0CB', '#A52A2A', '#808080', '#000000', '#FFD700', '#ADFF2F', '#FF4500', '#DA70D6', '#1E90FF', '#FF6347', '#7CFC00', '#8A2BE2', '#DC143C', '#FFFFFF#9932CC', '#8B0000', '#2E8B57', '#DAA520', '#4B0082', '#808000']
            plt.plot(x, y, linestyle='-', marker='o', markersize=3, color=colors[0], label='Total storage cost')
            min_idx = np.nanargmin(y)
            min_x = x[min_idx]
            print(f'  Minimum total cost at pack size={min_x}, cost={y[min_idx]:.2f}, bit-width cost={y1[min_idx]:.2f}, value cost={y2[min_idx]:.2f}')
            plt.plot(x, y1, linestyle='--', marker='x', markersize=3, color=colors[1], label='Bit width cost')
            plt.plot(x, y2, linestyle='--', marker='s', markersize=3, color=colors[2], label='Value cost')
            plt.xlabel('Pack size $s$', fontsize=fontsize)
            plt.ylabel('Cost (bits)', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=fontsize, labelspacing=0.1, handletextpad=0.1, columnspacing=0.1)
            outname = f'{os.path.splitext(csv_file)[0]}_rows_{start + 1}_{end}_value_and_bit_width.png'
            outpath = os.path.join(output_dir, outname)
            outpath_eps = os.path.join(output_dir, f'{os.path.splitext(csv_file)[0]}_rows_{start + 1}_{end}_value_and_bit_width.eps')
            plt.savefig(outpath_eps, dpi=150, bbox_inches='tight', format='eps')
            plt.savefig(outpath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved chunk plot: {outname}')

def create_chunk_vary_3_plots(csv_dir, output_dir, chunk_size=1024):
    os.makedirs(output_dir, exist_ok=True)
    sequences = [[3, 6, 12, 24, 48, 96, 192, 384, 768]]
    seq_names = ['3*$2^\\beta$ (3,6,...,768)']
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_cost.csv')]
    for csv_file in csv_files:
        if not csv_file.startswith('PM10-dust'):
            continue
        path = os.path.join(csv_dir, csv_file)
        print(f'Chunk plotting: {csv_file}')
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f'  Failed to read {csv_file}: {e}')
            continue
        if 'pack size' not in df.columns or 'cost' not in df.columns:
            print(f"  Skipping {csv_file}: missing 'packsize' or 'cost' columns")
            continue
        n = len(df)
        num_chunks = 1
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(n, (i + 1) * chunk_size)
            sub = df.iloc[start:end]
            if sub.empty:
                continue
            sub_sorted = sub.sort_values(by='pack size')
            x = sub_sorted['pack size'].values
            y_1 = sub_sorted['bitwidth_cost'].values
            y_2 = sub_sorted['value_cost'].values
            y = sub_sorted['cost'].values
            plt.figure(figsize=(6, 2))
            ax2 = plt.subplot(1, 1, 1)
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#008000', '#FFC0CB', '#A52A2A', '#808080', '#000000', '#FFD700', '#ADFF2F', '#FF4500', '#DA70D6', '#1E90FF', '#FF6347', '#7CFC00', '#8A2BE2', '#DC143C', '#00CED1', '#FF8C00', '#9932CC', '#8B0000', '#2E8B57', '#DAA520', '#4B0082', '#808000']
            all_seq_packsizes = set()
            for seq in sequences:
                all_seq_packsizes.update(seq)
            current_packsizes = set(sub['pack size'].unique())
            fontsize = 14
            for idx, (seq, seq_name, color) in enumerate(zip(sequences, seq_names, colors)):
                seq_in_data = [ps for ps in seq if ps in current_packsizes]
                if seq_in_data:
                    seq_data = []
                    for ps in seq_in_data:
                        avg_cost = sub[sub['pack size'] == ps]['cost'].mean()
                        avg_bitwidth_cost = sub[sub['pack size'] == ps]['bitwidth_cost'].mean()
                        avg_value_cost = sub[sub['pack size'] == ps]['value_cost'].mean()
                        seq_data.append((ps, avg_cost, avg_bitwidth_cost, avg_value_cost))
                    seq_data.sort(key=lambda x: x[0])
                    seq_x = [item[0] for item in seq_data]
                    seq_y = [item[1] for item in seq_data]
                    seq_y1 = [item[2] for item in seq_data]
                    seq_y2 = [item[3] for item in seq_data]
                    seq_name = seq_name.split('(')[0].strip()
                    ax2.plot(seq_x, seq_y, linestyle='-', marker='o', markersize=5, color=colors[0], linewidth=2, label=f'Total stoarge cost of {seq_name} ')
                    ax2.plot(seq_x, seq_y1, linestyle='--', marker='x', markersize=5, color=colors[1], linewidth=1, label=f'Bit width cost of {seq_name} ')
                    ax2.plot(seq_x, seq_y2, linestyle='--', marker='s', markersize=5, color=colors[2], linewidth=1, label=f'Value cost of {seq_name}')
            ax2.set_xlabel('Pack size s', fontsize=fontsize)
            ax2.set_ylabel('Cost (bits)', fontsize=fontsize)
            ax2.tick_params(axis='both', labelsize=fontsize)
            str_title = 'Cost of $s$ with $\\alpha$ = 3'
            legend = ax2.legend(loc='upper center', bbox_to_anchor=(0.4, 1.46), ncol=2, fontsize=fontsize, labelspacing=0.1, handletextpad=0.1, columnspacing=0.1)
            outname = f'{os.path.splitext(csv_file)[0]}_rows_{start + 1}_{end}_grouped_by_3.png'
            outpath = os.path.join(output_dir, outname)
            plt.savefig(outpath, dpi=300, bbox_inches='tight')
            outname = f'{os.path.splitext(csv_file)[0]}_rows_{start + 1}_{end}_grouped_by_3.eps'
            outpath = os.path.join(output_dir, outname)
            plt.savefig(outpath, dpi=300, bbox_inches='tight', format='eps')
            plt.close()
            print(f'  Saved chunk plot: {outname}')
if __name__ == '__main__':
    csv_dir = '../packsize_cost_analysis'
    output_dir = '../fig'
    chunk_output = '../fig'
    create_chunk_vary_3_plots(csv_dir, chunk_output, chunk_size=1024)
    chunk_output = '../fig'
    fig_of_cost_values_bitwidth_in_chunk(csv_dir, chunk_output, chunk_size=1024)
