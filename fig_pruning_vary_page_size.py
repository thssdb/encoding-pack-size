import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.axes import Axes
import matplotlib.path as mpath
from collections import OrderedDict


# 定义数据目录和算法映射
data_dirs = {
    'BP-Prune': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_BP_filters_plus_vary_page_size',
    'Sprintz-Prune': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_Sprintz_filters_plus_vary_page_size',
}
# 数据集映射（根据您之前的定义）
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
    'Cpu-usage_right.csv': 'CPU',
    'Disk-usage.csv': 'DISK',
    'Mem-usage.csv': 'MEM',
    
    # 非时间序列数据集
    'Food-price.csv': 'FP',
    'electric_vehicle_charging.csv': 'VC',
    'Blockchain-tr.csv': 'BTR',
    'SSD-bench.csv': 'SB',
    'City-lat.csv': 'CLT',
    'City-lon.csv': 'CLN',
}

# 要分析的pack sizes
vector_sizes = [16*8, 32*8, 64*8, 128*8, 256*8, 512*8, 1024*8]# [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# 初始化数据结构
pruning_rate_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}

# 读取和处理数据
for algorithm, data_dir in data_dirs.items():
    print(f"Processing algorithm: {algorithm}")
    
    # 获取目录中的所有CSV文件
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv') or filename == '.DS_Store' or not filename in dataset_mapping:
            continue
            
        # 获取数据集简称
        dataset_name = dataset_mapping.get(filename, filename)
        print(f"  Processing dataset: {dataset_name} ({filename})")
        
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                
                # 处理每一行数据
                for _, row in df.iterrows():
                    pack_size = row['Page size']
                    
                    # 确保pack size是数值类型
                    try:
                        pack_size = int(pack_size)
                    except:
                        continue
                    
                    if pack_size in vector_sizes:
                        # 存储压缩比（原始数据中已经是压缩比，不需要取倒数）
                        filter_count = float(row['Filter Count'])
                        page_size = float(row['Page size'])
                        pruning_rate_data[algorithm][pack_size].append(filter_count/page_size*100)
                        

            except Exception as e:
                print(f"    Error processing {file_path}: {e}")
                continue

# 计算每个算法在每个pack size下的平均值
avg_filter_rate = {}

for algorithm in data_dirs.keys():
    avg_filter_rate[algorithm] = []

    for size in vector_sizes:
        if pruning_rate_data[algorithm][size]:
            # 计算过滤率的平均值
            avg_filter = np.mean(pruning_rate_data[algorithm][size])
            avg_filter_rate[algorithm].append(avg_filter)
            
        else:
            avg_filter_rate[algorithm].append(0)


# print("\n平均过滤率:")
# for size, ratio in zip(vector_sizes, avg_filter_rate['BP']):
#     print(f"  Pack size {size}: {ratio:.4f}")

# 创建DataFrame用于绘图
df_compression_ratio = pd.DataFrame(avg_filter_rate, index=vector_sizes)
df_compression_ratio.index.name = 'Page Size'


# 重置索引以便Seaborn处理
df_compression_ratio_reset = df_compression_ratio.reset_index().melt(
    id_vars='Page Size', var_name='Algorithm', value_name='Filter Rate'
)



# 创建自定义标记
heart_vertices = [
    (0, 0), (0.5, 0.5), (1, 0), (0.5, -0.5), (0, 0),
    (-0.5, -0.5), (-1, 0), (-0.5, 0.5), (0, 0)
]
heart = mpath.Path(heart_vertices)

t = np.linspace(0, 2*np.pi, 100)
x = 16 * np.sin(t)**3
y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
x = x / np.max(np.abs(x))
y = y / np.max(np.abs(y))
heart_parametric = mpath.Path(np.column_stack([x, y]))

trapezoid_vertices = [
    (-0.8, -1),
    (0.8, -1),
    (0.4, 1),
    (-0.4, 1),
    (-0.8, -1)
]
trapezoid = mpath.Path(trapezoid_vertices)

parallelogram_vertices = [
    (-1, -0.6),
    (0.5, -0.6),
    (1, 0.6),
    (-0.5, 0.6),
    (-1, -0.6)
]
parallelogram = mpath.Path(parallelogram_vertices)

# 定义8种不同的标记
markers = [ '^',  's']

# 算法顺序
algorithm_order = ['BP-Prune', 'Sprintz-Prune']

# 设置颜色映射
algorithm_palette = {
    'BP': '#ff7f0e',
    'BP-Prune': '#9467bd',
    'BP-Prune-RMQ': '#1f77b4',
    'BP-All': '#d62728',
    'Sprintz': '#2ca02c',
    'Sprintz-Prune': '#8c564b',
    'Sprintz-Prune-RMQ': '#1f77b4',
    'Sprintz-All': '#17becf',
}
# 创建2x3子图：第一行为BP家族，第二行为Sprintz家族
fig, axs = plt.subplots(1, 1, figsize=(6, 4))
plt.subplots_adjust(wspace=0.28, hspace=0.35)

# 标注和刻度准备
fontsize = 13
exponents = [int(np.log2(ps)) for ps in vector_sizes]
exponent_labels = [f'$2^{{{exp}}}$' for exp in exponents]

# 定义每一行的算法组
bp_group = ['BP-Prune', 'Sprintz-Prune']

# 顶部行：BP 家族
ax1 = axs
for i, algorithm in enumerate(bp_group):
    data = df_compression_ratio_reset[df_compression_ratio_reset['Algorithm'] == algorithm]
    linestyle = '-' #if 'All' in algorithm or 'RMQ' not in algorithm else '--'
    ax1.plot(data['Page Size'], data['Filter Rate'],
             color=algorithm_palette[algorithm], linestyle=linestyle, marker=markers[i],
             markersize=7, linewidth=2.2, label=algorithm)
ax1.set_xscale('log', base=2)
ax1.set_ylabel('Percentage (% of page size)', fontsize=fontsize)
ax1.set_xlabel(r'Page Size $n$', fontsize=fontsize)
ax1.set_ylim(0, 100)
# ax1.set_title('BP: Compression Ratio', fontsize=fontsize)
ax1.set_xticks(vector_sizes)
ax1.set_xticklabels(exponent_labels)
ax1.tick_params(labelsize=fontsize)

# 添加图例 - 只在第一个子图中添加图例
# ax1.legend(fontsize=fontsize)

all_handles = []
all_labels = []
# for ax_row in axs:
#     for ax in ax_row:
h, l = axs.get_legend_handles_labels()
all_handles.extend(h)
all_labels.extend(l)

by_label = OrderedDict(zip(all_labels, all_handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4,
           labelspacing=0.3,
            handletextpad=0.3,
            columnspacing=0.3,
           fontsize=fontsize, bbox_to_anchor=(0.5, 1))

# 调整子图间距
# plt.tight_layout()

# 保存图片
output_dir = "/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/figure_for_paper"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, 'filter_rating_vary_page_size.png'), dpi=400, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'filter_rating_vary_page_size.eps'), format='eps', dpi=400, bbox_inches='tight')

