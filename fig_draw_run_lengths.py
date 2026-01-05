import os
import pandas as pd
import matplotlib.pyplot as plt

# # 读取CSV文件
# df = pd.read_csv('/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/run-length.csv')
# fontsize = 16
# # 绘制直方图
# plt.figure(figsize=(10, 6))
# plt.hist(df['run_length'], bins=128, edgecolor='black')
# plt.title('Dictribution of run lengths', fontsize=fontsize)
# plt.xlabel('Run Length', fontsize=fontsize)
# plt.ylabel('Frequency', fontsize=fontsize)
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.yscale('log')
# # plt.grid(True, alpha=0.3)

# plt.savefig('run_length_distribution.png', dpi=400, bbox_inches='tight')
# # plt.show()



# 打印统计信息
# print(f"平均游程长度: {df['run_length'].mean():.2f}")


# read both CSV files and draw both histograms on a single canvas (2 rows, 1 column)
df1 = pd.read_csv('/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/features_and_best_p.csv')
df2 = pd.read_csv('/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/features_and_best_p_sprintz.csv')
fontsize = 18

fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Top subplot: original optimal p
axes[0].hist(df1['optimal_pack_size'], bins=1024, edgecolor='black', color='C0')
print(df1['optimal_pack_size'].describe())
print(f"众数: {df1['optimal_pack_size'].mode()[0]}")
axes[0].set_title('(a) Distribution of optimal pack sizes (BP)', fontsize=fontsize)
axes[0].set_xlabel('Pack size', fontsize=fontsize)
axes[0].set_ylabel('Frequency', fontsize=fontsize)
axes[0].tick_params(axis='both', labelsize=fontsize)
axes[0].set_xlim(0.9, 128)
axes[0].set_ylim(0, 520)

# Bottom subplot: after sprintz
axes[1].hist(df2['optimal_pack_size'], bins=1024, edgecolor='black', color='C1')
print(df2['optimal_pack_size'].describe())
print(f"众数: {df2['optimal_pack_size'].mode()[0]}")
axes[1].set_title('(b) Distribution of optimal pack sizes (Sprintz)', fontsize=fontsize)
axes[1].set_xlabel('Pack size', fontsize=fontsize)
axes[1].set_ylabel('Frequency', fontsize=fontsize)
axes[1].tick_params(axis='both', labelsize=fontsize)
axes[1].set_xlim(0.9, 128)
axes[1].set_ylim(0, 520)

plt.tight_layout()
os.makedirs('./figure_for_paper', exist_ok=True)
plt.savefig('./figure_for_paper/optimal_pack_size_distribution_combined.png', dpi=400, bbox_inches='tight')
plt.savefig('./figure_for_paper/optimal_pack_size_distribution_combined.eps', format='eps', dpi=400, bbox_inches='tight')

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

# 再画一个图 显示dataset_mapping中的前3个数据集的df1的optimal p分布，df1[Dataset]和df2[Dataset]为dataset_mapping中的key
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
# plt.subplots_adjust(hspace=0.3)

dataset_file=list(dataset_mapping.items())[3][0]
dataset_abbr=list(dataset_mapping.items())[3][1]
df1_subset = df1[df1['Dataset'] == dataset_file]
axes[0].hist(df1_subset['optimal_pack_size'], bins=128, edgecolor='black', alpha=0.5, label=dataset_abbr)
axes[0].set_title('(a) Dataset '+dataset_abbr, fontsize=fontsize)
axes[0].set_xlabel('Pack size', fontsize=fontsize)
axes[0].set_ylabel('Frequency', fontsize=fontsize)
axes[0].set_ylim(0, 40)
axes[0].tick_params(axis='both', labelsize=fontsize)
axes[0].set_xlim(0.9, 128)
# axes[0].legend(fontsize=fontsize)
dataset_file=list(dataset_mapping.items())[2][0]
dataset_abbr=list(dataset_mapping.items())[2][1]
df1_subset = df1[df1['Dataset'] == dataset_file]
axes[1].hist(df1_subset['optimal_pack_size'], bins=128, edgecolor='black', alpha=0.5, label=dataset_abbr)
axes[1].set_title('(b) Dataset '+dataset_abbr, fontsize=fontsize)
axes[1].set_xlabel('Pack size', fontsize=fontsize)
axes[1].set_ylabel('Frequency', fontsize=fontsize)
axes[1].set_ylim(0, 40)
axes[1].tick_params(axis='both', labelsize=fontsize)
axes[1].set_xlim(0.9, 128)
# # axes[1].legend(fontsize=fontsize)
# dataset_file=list(dataset_mapping.items())[2][0]
# dataset_abbr=list(dataset_mapping.items())[2][1]
# df1_subset = df1[df1['Dataset'] == dataset_file]
# axes[2].hist(df1_subset['optimal_pack_size'], bins=128, edgecolor='black', alpha=0.5, label=dataset_abbr)
# axes[2].set_title('(c) Dataset '+dataset_abbr, fontsize=fontsize)
# axes[2].set_xlabel('Pack size', fontsize=fontsize)
# axes[2].set_ylabel('Frequency', fontsize=fontsize)
# axes[2].set_ylim(0, 40)
# axes[2].tick_params(axis='both', labelsize=fontsize)
# axes[2].set_xlim(0.9, 128)
# # axes[2].legend(fontsize=fontsize)

plt.tight_layout()
os.makedirs('./figure_for_paper', exist_ok=True)
plt.savefig('./figure_for_paper/optimal_pack_size_distribution_selected_datasets_bp.png', dpi=400, bbox_inches='tight')
plt.savefig('./figure_for_paper/optimal_pack_size_distribution_selected_datasets_bp.eps', format='eps', dpi=400, bbox_inches='tight')

# 打印统计信息
# print(f"平均游程长度: {df['run_length'].mean():.2f}")