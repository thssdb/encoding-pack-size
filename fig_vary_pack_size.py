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
    'BP': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_BP_vary_pack_size',
    'Sprintz': '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_Sprintz_vary_pack_size',
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
vector_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# 初始化数据结构
compression_ratio_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}
encode_time_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}
decode_time_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}

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
                    pack_size = row['Pack size']
                    
                    # 确保pack size是数值类型
                    try:
                        pack_size = int(pack_size)
                    except:
                        continue
                    
                    if pack_size in vector_sizes:
                        # 存储压缩比（原始数据中已经是压缩比，不需要取倒数）
                        compression_ratio = float(row['Compression Ratio'])
                        compression_ratio_data[algorithm][pack_size].append(1/compression_ratio)
                        
                        # 存储编码时间
                        encode_time = float(row['Encoding Time'])
                        encode_time_data[algorithm][pack_size].append(1/(encode_time/8000))
                        
                        # 存储解码时间
                        decode_time = float(row['Decoding Time'])
                        decode_time_data[algorithm][pack_size].append(1/(decode_time/8000))
                        
            except Exception as e:
                print(f"    Error processing {file_path}: {e}")
                continue

# 计算每个算法在每个pack size下的平均值
avg_compression_ratio = {}
avg_encode_time = {}
avg_decode_time = {}

for algorithm in data_dirs.keys():
    avg_compression_ratio[algorithm] = []
    avg_encode_time[algorithm] = []
    avg_decode_time[algorithm] = []
    
    for size in vector_sizes:
        if compression_ratio_data[algorithm][size]:
            # 计算压缩比的平均值
            avg_cr = np.mean(compression_ratio_data[algorithm][size])
            avg_compression_ratio[algorithm].append(avg_cr)
            
            # 计算编码时间的平均值
            avg_et = np.mean(encode_time_data[algorithm][size])
            avg_encode_time[algorithm].append(avg_et)
            
            # 计算解码时间的平均值
            avg_dt = np.mean(decode_time_data[algorithm][size])
            avg_decode_time[algorithm].append(avg_dt)
        else:
            avg_compression_ratio[algorithm].append(0)
            avg_encode_time[algorithm].append(0)
            avg_decode_time[algorithm].append(0)

print("\n平均压缩比:")
for size, ratio in zip(vector_sizes, avg_compression_ratio['BP']):
    print(f"  Pack size {size}: {ratio:.4f}")

# 读取camel_ratio.xlsx文件并计算均值
camel_ratio_path = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/camel_ratio.xlsx'
if os.path.exists(camel_ratio_path):
    print(f"\n读取camel_ratio.xlsx文件: {camel_ratio_path}")
    camel_df = pd.read_excel(camel_ratio_path)
    
    # 获取有效的数据集简称（dataset_mapping中的values）
    valid_datasets = set(dataset_mapping.values())
    print(f"有效数据集简称: {valid_datasets}")
    
    # 获取camel_df中的所有列名
    all_columns = camel_df.columns.tolist()
    print(f"Excel文件中的列名: {all_columns}")
    
    # 找出在有效数据集中的列（排除第一列"算法名称"）
    valid_columns = [col for col in all_columns[1:] if col in valid_datasets]
    print(f"参与计算的有效列: {valid_columns}")
    
    # 计算BP-RMQ的均值（只计算有效列）
    bp_rmq_row = camel_df[camel_df.iloc[:, 0] == 'BP (Prune-RMQ)']
    if not bp_rmq_row.empty:
        # 只取有效列的数据
        bp_rmq_values = []
        for col in valid_columns:
            if col in bp_rmq_row.columns:
                val = bp_rmq_row[col].iloc[0]
                if pd.notna(val):  # 只处理非空值
                    bp_rmq_values.append(float(val))
        
        if bp_rmq_values:
            bp_rmq_values = 1 / np.array(bp_rmq_values)
            bp_rmq_mean = np.mean(bp_rmq_values)
            print(f"BP-RMQ均值 (基于{len(bp_rmq_values)}个数据集): {bp_rmq_mean:.4f}")
            print(f"具体值: {bp_rmq_values}")
        else:
            print("BP-RMQ行中没有有效数据")
            bp_rmq_mean = None
    else:
        print("未找到BP-RMQ行")
        bp_rmq_mean = None

    # 计算BP-RMQ的均值（只计算有效列）
    bp_all_row = camel_df[camel_df.iloc[:, 0] == 'BP (All)']
    if not bp_all_row.empty:
        # 只取有效列的数据
        bp_all_values = []
        for col in valid_columns:
            if col in bp_all_row.columns:
                val = bp_all_row[col].iloc[0]
                if pd.notna(val):  # 只处理非空值
                    bp_all_values.append(float(val))

        if bp_all_values:
            bp_all_values = 1 / np.array(bp_all_values)
            bp_all_mean = np.mean(bp_all_values)
            print(f"BP-All均值 (基于{len(bp_all_values)}个数据集): {bp_all_mean:.4f}")
            print(f"具体值: {bp_all_values}")
        else:
            print("BP-All行中没有有效数据")
            bp_all_mean = None
    else:
        print("未找到BP-All行")
        bp_all_mean = None

    # 计算 BP (Prune) 的均值（如果存在由 PackSizeMLTrainerAndEvaluator 生成的 learning_evaluation_results.csv）
    bp_learn_row = camel_df[camel_df.iloc[:, 0] == 'BP (Prune Plus)']
    if not bp_learn_row.empty:
        bp_learn_values = []
        for col in valid_columns:
            if col in bp_learn_row.columns:
                val = bp_learn_row[col].iloc[0]
                if pd.notna(val):
                    bp_learn_values.append(float(val))

        if bp_learn_values:
            bp_learn_values = 1 / np.array(bp_learn_values)
            bp_learn_mean = np.mean(bp_learn_values)
            print(f"BP-Learn均值 (基于{len(bp_learn_values)}个数据集): {bp_learn_mean:.4f}")
            print(f"具体值: {bp_learn_values}")
        else:
            print("BP-Learn行中没有有效数据")
            bp_learn_mean = None
    else:
        print("未找到BP-Learn行")
        bp_learn_mean = None

    # 计算Sprintz-RMQ的均值
    sprintz_rmq_row = camel_df[camel_df.iloc[:, 0] == 'Sprintz (RMQ)']
    if not sprintz_rmq_row.empty:
        # 只取有效列的数据
        sprintz_rmq_values = []
        for col in valid_columns:
            if col in sprintz_rmq_row.columns:
                val = sprintz_rmq_row[col].iloc[0]
                if pd.notna(val):  # 只处理非空值
                    sprintz_rmq_values.append(float(val))
        
        if sprintz_rmq_values:
            sprintz_rmq_values = 1 / np.array(sprintz_rmq_values)
            sprintz_rmq_mean = np.mean(sprintz_rmq_values)
            print(f"Sprintz-RMQ均值 (基于{len(sprintz_rmq_values)}个数据集): {sprintz_rmq_mean:.4f}")
            print(f"具体值: {sprintz_rmq_values}")
        else:
            print("Sprintz-RMQ行中没有有效数据")
            sprintz_rmq_mean = None
    else:
        print("未找到Sprintz-RMQ行")
        sprintz_rmq_mean = None
    
    # 计算Sprintz-RMQ的均值
    sprintz_all_row = camel_df[camel_df.iloc[:, 0] == 'Sprintz (All)']
    if not sprintz_all_row.empty:
        # 只取有效列的数据
        sprintz_all_values = []
        for col in valid_columns:
            if col in sprintz_all_row.columns:
                val = sprintz_all_row[col].iloc[0]
                if pd.notna(val):  # 只处理非空值
                    sprintz_all_values.append(float(val))

        if sprintz_all_values:
            sprintz_all_values = 1 / np.array(sprintz_all_values)
            sprintz_all_mean = np.mean(sprintz_all_values)
            print(f"Sprintz-All均值 (基于{len(sprintz_all_values)}个数据集): {sprintz_all_mean:.4f}")
            print(f"具体值: {sprintz_all_values}")
        else:
            print("Sprintz-All行中没有有效数据")
            sprintz_all_mean = None
    else:
        print("未找到Sprintz-All行")
        sprintz_all_mean = None
    # 计算 Sprintz (Prune) 的均值（如果 camel_ratio 中存在该行）
    sprintz_prune_row = camel_df[camel_df.iloc[:, 0] == 'Sprintz (Prune Plus)']
    if not sprintz_prune_row.empty:
        sprintz_prune_values = []
        for col in valid_columns:
            if col in sprintz_prune_row.columns:
                val = sprintz_prune_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_prune_values.append(float(val))

        if sprintz_prune_values:
            sprintz_prune_values = 1 / np.array(sprintz_prune_values)
            sprintz_prune_mean = np.mean(sprintz_prune_values)
            print(f"Sprintz-Prune均值 (基于{len(sprintz_prune_values)}个数据集): {sprintz_prune_mean:.4f}")
            print(f"具体值: {sprintz_prune_values}")
        else:
            print("Sprintz-Prune行中没有有效数据")
            sprintz_prune_mean = None
    else:
        print("未找到Sprintz-Prune行")
        sprintz_prune_mean = None
else:
    print(f"\ncamel_ratio.xlsx文件不存在: {camel_ratio_path}")
    bp_rmq_mean = None
    bp_all_mean = None
    sprintz_rmq_mean = None
    sprintz_all_mean = None

# 读取编码和解码吞吐率数据
camel_encode_path = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/compression_time.xlsx'
camel_decode_path = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/decompression_time.xlsx'

# 初始化编码和解码吞吐率均值
bp_rmq_encode_mean = None
sprintz_rmq_encode_mean = None
bp_rmq_decode_mean = None
sprintz_rmq_decode_mean = None
bp_learn_encode_mean = None
bp_learn_decode_mean = None
sprintz_prune_encode_mean = None
sprintz_prune_decode_mean = None

# 读取编码吞吐率数据
if os.path.exists(camel_encode_path):
    print(f"\n读取camel_encode.xlsx文件: {camel_encode_path}")
    encode_df = pd.read_excel(camel_encode_path)
    
    # 获取有效的数据集简称（dataset_mapping中的values）
    valid_datasets = set(dataset_mapping.values())
    
    # 获取encode_df中的所有列名
    all_columns = encode_df.columns.tolist()
    
    # 找出在有效数据集中的列（排除第一列"算法名称"）
    valid_columns = [col for col in all_columns[1:] if col in valid_datasets]
    
    # 计算BP-RMQ的编码吞吐率均值
    bp_rmq_encode_row = encode_df[encode_df.iloc[:, 0] == 'BP (Prune-RMQ)']
    if not bp_rmq_encode_row.empty:
        bp_rmq_encode_values = []
        for col in valid_columns:
            if col in bp_rmq_encode_row.columns:
                val = bp_rmq_encode_row[col].iloc[0]
                if pd.notna(val):
                    bp_rmq_encode_values.append(1/(float(val)/8000))
        
        if bp_rmq_encode_values:
            bp_rmq_encode_mean = np.mean(bp_rmq_encode_values)
            print(f"BP-RMQ编码吞吐率均值 (基于{len(bp_rmq_encode_values)}个数据集): {bp_rmq_encode_mean:.2f} MB/s")
        else:
            print("BP-RMQ行中没有有效的编码吞吐率数据")
    else:
        print("未找到BP-RMQ的编码吞吐率行")
    
    # 计算BP-All的编码吞吐率均值
    bp_all_encode_row = encode_df[encode_df.iloc[:, 0] == 'BP (All)']
    if not bp_all_encode_row.empty:
        bp_all_encode_values = []
        for col in valid_columns:
            if col in bp_all_encode_row.columns:
                val = bp_all_encode_row[col].iloc[0]
                if pd.notna(val):
                    bp_all_encode_values.append(1/(float(val)/8000))

        if bp_all_encode_values:
            bp_all_encode_mean = np.mean(bp_all_encode_values)
            print(f"BP-All编码吞吐率均值 (基于{len(bp_all_encode_values)}个数据集): {bp_all_encode_mean:.2f} MB/s")
        else:
            print("BP-All行中没有有效的编码吞吐率数据")
    else:
        print("未找到BP-All的编码吞吐率行")

    # 读取 BP (Prune) 的编码吞吐率（如果 camel_encode 中有该行）
    bp_learn_encode_row = encode_df[encode_df.iloc[:, 0] == 'BP (Prune Plus)']
    if not bp_learn_encode_row.empty:
        bp_learn_encode_values = []
        for col in valid_columns:
            if col in bp_learn_encode_row.columns:
                val = bp_learn_encode_row[col].iloc[0]
                if pd.notna(val):
                    bp_learn_encode_values.append(1/(float(val)/8000))
        if bp_learn_encode_values:
            bp_learn_encode_mean = np.mean(bp_learn_encode_values)
            print(f"BP-Learn编码吞吐率均值 (基于{len(bp_learn_encode_values)}个数据集): {bp_learn_encode_mean:.2f} MB/s")
        else:
            print("BP-Learn行中没有有效的编码吞吐率数据")
    else:
        print("未找到BP-Learn的编码吞吐率行")
    
    # 计算Sprintz-RMQ的编码吞吐率均值
    sprintz_rmq_encode_row = encode_df[encode_df.iloc[:, 0] == 'Sprintz (RMQ)']
    if not sprintz_rmq_encode_row.empty:
        sprintz_rmq_encode_values = []
        for col in valid_columns:
            if col in sprintz_rmq_encode_row.columns:
                val = sprintz_rmq_encode_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_rmq_encode_values.append(1/(float(val)/8000))
        
        if sprintz_rmq_encode_values:
            sprintz_rmq_encode_mean = np.mean(sprintz_rmq_encode_values)
            print(f"Sprintz-RMQ编码吞吐率均值 (基于{len(sprintz_rmq_encode_values)}个数据集): {sprintz_rmq_encode_mean:.2f} MB/s")
        else:
            print("Sprintz-RMQ行中没有有效的编码吞吐率数据")
    else:
        print("未找到Sprintz-RMQ的编码吞吐率行")

    # 计算Sprintz-All的编码吞吐率均值
    sprintz_all_encode_row = encode_df[encode_df.iloc[:, 0] == 'Sprintz (All)']
    if not sprintz_all_encode_row.empty:
        sprintz_all_encode_values = []
        for col in valid_columns:
            if col in sprintz_all_encode_row.columns:
                val = sprintz_all_encode_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_all_encode_values.append(1/(float(val)/8000))

        if sprintz_all_encode_values:
            sprintz_all_encode_mean = np.mean(sprintz_all_encode_values)
            print(f"Sprintz-All编码吞吐率均值 (基于{len(sprintz_all_encode_values)}个数据集): {sprintz_all_encode_mean:.2f} MB/s")
        else:
            print("Sprintz-All行中没有有效的编码吞吐率数据")
    else:
        print("未找到Sprintz-All的编码吞吐率行")

    # 读取 Sprintz (Prune) 的编码吞吐率（如果 camel_encode 中有该行）
    sprintz_prune_encode_row = encode_df[encode_df.iloc[:, 0] == 'Sprintz (Prune Plus)']
    if not sprintz_prune_encode_row.empty:
        sprintz_prune_encode_values = []
        for col in valid_columns:
            if col in sprintz_prune_encode_row.columns:
                val = sprintz_prune_encode_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_prune_encode_values.append(1/(float(val)/8000))

        if sprintz_prune_encode_values:
            sprintz_prune_encode_mean = np.mean(sprintz_prune_encode_values)
            print(f"Sprintz-Prune编码吞吐率均值 (基于{len(sprintz_prune_encode_values)}个数据集): {sprintz_prune_encode_mean:.2f} MB/s")
        else:
            print("Sprintz-Prune行中没有有效的编码吞吐率数据")
    else:
        print("未找到Sprintz-Prune的编码吞吐率行")
else:
    print(f"\ncamel_encode.xlsx文件不存在: {camel_encode_path}")

# 读取解码吞吐率数据
if os.path.exists(camel_decode_path):
    print(f"\n读取camel_decode.xlsx文件: {camel_decode_path}")
    decode_df = pd.read_excel(camel_decode_path)
    
    # 获取有效的数据集简称（dataset_mapping中的values）
    valid_datasets = set(dataset_mapping.values())
    
    # 获取decode_df中的所有列名
    all_columns = decode_df.columns.tolist()
    
    # 找出在有效数据集中的列（排除第一列"算法名称"）
    valid_columns = [col for col in all_columns[1:] if col in valid_datasets]
    
    # 计算BP-RMQ的解码吞吐率均值
    bp_rmq_decode_row = decode_df[decode_df.iloc[:, 0] == 'BP (Prune-RMQ)']
    if not bp_rmq_decode_row.empty:
        bp_rmq_decode_values = []
        for col in valid_columns:
            if col in bp_rmq_decode_row.columns:
                val = bp_rmq_decode_row[col].iloc[0]
                if pd.notna(val):
                    bp_rmq_decode_values.append(1/(float(val)/8000))
        
        if bp_rmq_decode_values:
            bp_rmq_decode_mean = np.mean(bp_rmq_decode_values)
            print(f"BP-RMQ解码吞吐率均值 (基于{len(bp_rmq_decode_values)}个数据集): {bp_rmq_decode_mean:.2f} MB/s")
        else:
            print("BP-RMQ行中没有有效的解码吞吐率数据")
    else:
        print("未找到BP-RMQ的解码吞吐率行")

    # 计算BP-All的解码吞吐率均值
    bp_all_decode_row = decode_df[decode_df.iloc[:, 0] == 'BP (All)']
    if not bp_all_decode_row.empty:
        bp_all_decode_values = []
        for col in valid_columns:
            if col in bp_all_decode_row.columns:
                val = bp_all_decode_row[col].iloc[0]
                if pd.notna(val):
                    bp_all_decode_values.append(1/(float(val)/8000))

        if bp_all_decode_values:
            bp_all_decode_mean = np.mean(bp_all_decode_values)
            print(f"BP-All解码吞吐率均值 (基于{len(bp_all_decode_values)}个数据集): {bp_all_decode_mean:.2f} MB/s")
        else:
            print("BP-All行中没有有效的解码吞吐率数据")
    else:
        print("未找到BP-All的解码吞吐率行")

    # 读取 BP (learn) 的解码吞吐率
    bp_learn_decode_row = decode_df[decode_df.iloc[:, 0] == 'BP (Prune Plus)']
    if not bp_learn_decode_row.empty:
        bp_learn_decode_values = []
        for col in valid_columns:
            if col in bp_learn_decode_row.columns:
                val = bp_learn_decode_row[col].iloc[0]
                if pd.notna(val):
                    bp_learn_decode_values.append(1/(float(val)/8000))
        if bp_learn_decode_values:
            bp_learn_decode_mean = np.mean(bp_learn_decode_values)
            print(f"BP-Learn解码吞吐率均值 (基于{len(bp_learn_decode_values)}个数据集): {bp_learn_decode_mean:.2f} MB/s")
        else:
            print("BP-Learn行中没有有效的解码吞吐率数据")
    else:
        print("未找到BP-Learn的解码吞吐率行")

    # 计算Sprintz-RMQ的解码吞吐率均值
    sprintz_rmq_decode_row = decode_df[decode_df.iloc[:, 0] == 'Sprintz (RMQ)']
    if not sprintz_rmq_decode_row.empty:
        sprintz_rmq_decode_values = []
        for col in valid_columns:
            if col in sprintz_rmq_decode_row.columns:
                val = sprintz_rmq_decode_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_rmq_decode_values.append(1/(float(val)/8000))
        
        if sprintz_rmq_decode_values:
            sprintz_rmq_decode_mean = np.mean(sprintz_rmq_decode_values)
            print(f"Sprintz-RMQ解码吞吐率均值 (基于{len(sprintz_rmq_decode_values)}个数据集): {sprintz_rmq_decode_mean:.2f} MB/s")
        else:
            print("Sprintz-RMQ行中没有有效的解码吞吐率数据")
    else:
        print("未找到Sprintz-RMQ的解码吞吐率行")

    # 计算Sprintz-All的解码吞吐率均值
    sprintz_all_decode_row = decode_df[decode_df.iloc[:, 0] == 'Sprintz (All)']
    if not sprintz_all_decode_row.empty:
        sprintz_all_decode_values = []
        for col in valid_columns:
            if col in sprintz_all_decode_row.columns:
                val = sprintz_all_decode_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_all_decode_values.append(1/(float(val)/8000))

        if sprintz_all_decode_values:
            sprintz_all_decode_mean = np.mean(sprintz_all_decode_values)
            print(f"Sprintz-All解码吞吐率均值 (基于{len(sprintz_all_decode_values)}个数据集): {sprintz_all_decode_mean:.2f} MB/s")
        else:
            print("Sprintz-All行中没有有效的解码吞吐率数据")
    else:
        print("未找到Sprintz-All的解码吞吐率行")

    # 读取 Sprintz (Prune) 的解码吞吐率（如果 camel_decode 中有该行）
    sprintz_prune_decode_row = decode_df[decode_df.iloc[:, 0] == 'Sprintz (Prune Plus)']
    if not sprintz_prune_decode_row.empty:
        sprintz_prune_decode_values = []
        for col in valid_columns:
            if col in sprintz_prune_decode_row.columns:
                val = sprintz_prune_decode_row[col].iloc[0]
                if pd.notna(val):
                    sprintz_prune_decode_values.append(1/(float(val)/8000))

        if sprintz_prune_decode_values:
            sprintz_prune_decode_mean = np.mean(sprintz_prune_decode_values)
            print(f"Sprintz-Prune解码吞吐率均值 (基于{len(sprintz_prune_decode_values)}个数据集): {sprintz_prune_decode_mean:.2f} MB/s")
        else:
            print("Sprintz-Prune行中没有有效的解码吞吐率数据")
    else:
        print("未找到Sprintz-Prune的解码吞吐率行")

else:
    print(f"\ncamel_decode.xlsx文件不存在: {camel_decode_path}")

# 创建DataFrame用于绘图
df_compression_ratio = pd.DataFrame(avg_compression_ratio, index=vector_sizes)
df_compression_ratio.index.name = 'Pack Size'

df_encode_time = pd.DataFrame(avg_encode_time, index=vector_sizes)
df_encode_time.index.name = 'Pack Size'

df_decode_time = pd.DataFrame(avg_decode_time, index=vector_sizes)
df_decode_time.index.name = 'Pack Size'

# 重置索引以便Seaborn处理
df_compression_ratio_reset = df_compression_ratio.reset_index().melt(
    id_vars='Pack Size', var_name='Algorithm', value_name='Compression Ratio'
)

df_encode_time_reset = df_encode_time.reset_index().melt(
    id_vars='Pack Size', var_name='Algorithm', value_name='Encoding Time (MB/s)'
)

df_decode_time_reset = df_decode_time.reset_index().melt(
    id_vars='Pack Size', var_name='Algorithm', value_name='Decoding Time (MB/s)'
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
markers = ['o', '^', parallelogram, heart, 's', 'v', trapezoid, heart_parametric]

# 算法顺序（包含 RMQ / All）
algorithm_order = ['BP', 'BP (RMQ)', 'BP (All)', 'Sprintz', 'Sprintz (RMQ)', 'Sprintz (All)']

# 设置颜色映射（同色分组）
algorithm_palette = {
    'BP': '#ff7f0e',        # BP (orange)
    # 'BP (RMQ)': '#d62728',  # BP (RMQ) (red)
    'BP (All)': '#9467bd',  # BP (All) (purple)
    'Sprintz': '#2ca02c',        # Sprintz (green)
    # 'Sprintz (RMQ)': '#17becf',  # Sprintz (RMQ) (cyan)
    'Sprintz (All)': '#8c564b',  # Sprintz (All) (brown)
}

# 创建2x3子图：第一行为BP家族，第二行为Sprintz家族
fig, axs = plt.subplots(3, 2, figsize=(9, 13))
plt.subplots_adjust(wspace=0.45, hspace=0.35)

# 标注和刻度准备
fontsize = 18
exponents = [int(np.log2(ps)) for ps in vector_sizes]
exponent_labels = [f'$2^{{{exp}}}$' for exp in exponents]
plt.rcParams.update({'font.size': fontsize})
# 定义每一行的算法组
bp_group = ['BP'] #, 'BP (RMQ)', 'BP (All)'
sprintz_group = ['Sprintz'] #, 'Sprintz (RMQ)', 'Sprintz (All)'

# 顶部行：BP 家族
ax1 = axs[0, 0]
for i, algorithm in enumerate(bp_group):
    data = df_compression_ratio_reset[df_compression_ratio_reset['Algorithm'] == algorithm]
    linestyle = '-' if 'All' in algorithm or 'RMQ' not in algorithm else '--'
    ax1.plot(data['Pack Size'], data['Compression Ratio'],
             color=algorithm_palette[algorithm], linestyle=linestyle, marker=markers[i],
             markersize=7, linewidth=2.2, label=algorithm)
# 添加BP的 RMQ/All 平均值（来自 camel_ratio.xlsx）

if bp_all_mean is not None:
    ax1.axhline(y=bp_all_mean, color='#9467bd', linestyle='--', linewidth=1.8, label='BP-All')
if bp_learn_mean is not None:
    ax1.axhline(y=bp_learn_mean, color='#1f77b4', linestyle='-.', linewidth=1.8, label='BP-Prune')
if bp_rmq_mean is not None:
    ax1.axhline(y=bp_rmq_mean, color='#d62728', linestyle='--', linewidth=1.8, label='BP-Prune-RMQ')
ax1.set_xscale('log', base=2)
ax1.set_ylabel('Compression Ratio', fontsize=fontsize)
ax1.set_xlabel(r'Pack Size $s$', fontsize=fontsize)
ax1.set_title('(a) BP: Compression Ratio', fontsize=fontsize,x=0.4)
ax1.set_xticks(vector_sizes)
ax1.set_xticklabels(exponent_labels)
ax1.tick_params(labelsize=fontsize)

ax2 = axs[1, 0]
for i, algorithm in enumerate(bp_group):
    data = df_encode_time_reset[df_encode_time_reset['Algorithm'] == algorithm]
    linestyle = '-' if 'All' in algorithm or 'RMQ' not in algorithm else '--'
    ax2.plot(data['Pack Size'], data['Encoding Time (MB/s)'],
             color=algorithm_palette[algorithm], linestyle=linestyle, marker=markers[i],
             markersize=7, linewidth=2.2, label=algorithm)
if bp_all_encode_mean is not None:
    ax2.axhline(y=bp_all_encode_mean, color='#9467bd',linestyle='--',  linewidth=1.8, label='BP-All')
if bp_learn_encode_mean is not None:
    ax2.axhline(y=bp_learn_encode_mean, color='#1f77b4', linestyle='-.', linewidth=1.8, label='BP-Prune')
if bp_rmq_encode_mean is not None:
    ax2.axhline(y=bp_rmq_encode_mean, color='#d62728',linestyle='--',  linewidth=1.8, label='BP-Prune-RMQ')
ax2.set_xscale('log', base=2)
# ax2.set_yscale('log')
ax2.set_ylim(0,1210)
ax2.set_ylabel('Time (ns/point)', fontsize=fontsize)
ax2.set_xlabel(r'Pack Size $s$', fontsize=fontsize)
ax2.set_title('(c) BP: Compression Time', fontsize=fontsize, x=0.4)
ax2.set_xticks(vector_sizes)
ax2.set_xticklabels(exponent_labels)
ax2.tick_params(labelsize=fontsize)

ax3 = axs[2, 0]
for i, algorithm in enumerate(bp_group):
    data = df_decode_time_reset[df_decode_time_reset['Algorithm'] == algorithm]
    linestyle = '-' if 'All' in algorithm or 'RMQ' not in algorithm else '--'
    ax3.plot(data['Pack Size'], data['Decoding Time (MB/s)'],
             color=algorithm_palette[algorithm], linestyle=linestyle, marker=markers[i],
             markersize=7, linewidth=2.2, label=algorithm)
if bp_all_decode_mean is not None:
    ax3.axhline(y=bp_all_decode_mean, color='#9467bd', linestyle='--', linewidth=1.8, label='BP-All')
if bp_learn_decode_mean is not None:
    ax3.axhline(y=bp_learn_decode_mean, color='#1f77b4', linestyle='-.', linewidth=1.8, label='BP-Prune')
if bp_rmq_decode_mean is not None:
    ax3.axhline(y=bp_rmq_decode_mean, color='#d62728', linestyle='--', linewidth=1.8, label='BP-Prune-RMQ')
ax3.set_xscale('log', base=2)
# ax3.set_yscale('log')
ax3.set_ylim(0,1210)
ax3.set_ylabel('Time (ns/point)', fontsize=fontsize)
ax3.set_xlabel(r'Pack Size $s$', fontsize=fontsize)
ax3.set_title('(e) BP: Decompression Time', fontsize=fontsize, x=0.4)
ax3.set_xticks(vector_sizes)
ax3.set_xticklabels(exponent_labels)
ax3.tick_params(labelsize=fontsize)

# 底部行：Sprintz 家族
ax4 = axs[0, 1]
for i, algorithm in enumerate(sprintz_group):
    data = df_compression_ratio_reset[df_compression_ratio_reset['Algorithm'] == algorithm]
    linestyle = '-' if 'All' in algorithm or 'RMQ' not in algorithm else '--'
    ax4.plot(data['Pack Size'], data['Compression Ratio'],
             color=algorithm_palette[algorithm], linestyle=linestyle, marker=markers[i+3],
             markersize=7, linewidth=2.2, label=algorithm)
if sprintz_all_mean is not None:
    ax4.axhline(y=sprintz_all_mean, color='#8c564b', linestyle='--', linewidth=1.8, label='Sprintz-All')
if 'sprintz_prune_mean' in globals() and sprintz_prune_mean is not None:
    ax4.axhline(y=sprintz_prune_mean, color='#1f77b4', linestyle='-.', linewidth=1.8, label='Sprintz-Prune')
if sprintz_rmq_mean is not None:
    ax4.axhline(y=sprintz_rmq_mean, color='#17becf', linestyle='--', linewidth=1.8, label='Sprintz-Prune-RMQ')
ax4.set_xscale('log', base=2)
ax4.set_ylabel('Compression Ratio', fontsize=fontsize)
ax4.set_xlabel(r'Pack Size $s$', fontsize=fontsize)
ax4.set_title('(b) Sprintz: Compression Ratio', fontsize=fontsize,x=0.34)
ax4.set_xticks(vector_sizes)
ax4.set_xticklabels(exponent_labels)
ax4.tick_params(labelsize=fontsize)

ax5 = axs[1, 1]
for i, algorithm in enumerate(sprintz_group):
    data = df_encode_time_reset[df_encode_time_reset['Algorithm'] == algorithm]
    linestyle = '-' if 'All' in algorithm or 'RMQ' not in algorithm else '--'
    ax5.plot(data['Pack Size'], data['Encoding Time (MB/s)'],
             color=algorithm_palette[algorithm], linestyle=linestyle, marker=markers[i+3],
             markersize=7, linewidth=2.2, label=algorithm)
if sprintz_all_encode_mean is not None:
    ax5.axhline(y=sprintz_all_encode_mean, color='#8c564b', linestyle='--', linewidth=1.8, label='Sprintz-All')
if 'sprintz_prune_encode_mean' in globals() and sprintz_prune_encode_mean is not None:
    ax5.axhline(y=sprintz_prune_encode_mean, color='#1f77b4', linestyle='-.', linewidth=1.8, label='Sprintz-Prune')
if sprintz_rmq_encode_mean is not None:
    ax5.axhline(y=sprintz_rmq_encode_mean, color='#17becf', linestyle='--', linewidth=1.8, label='Sprintz-Prune-RMQ')

ax5.set_xscale('log', base=2)
# ax5.set_yscale('log')
ax5.set_ylim(0,1210)
ax5.set_ylabel('Time (ns/point)', fontsize=fontsize)
ax5.set_xlabel(r'Pack Size $s$', fontsize=fontsize)
ax5.set_title('(d) Sprintz: Compression Time', fontsize=fontsize,x=0.34)
ax5.set_xticks(vector_sizes)
ax5.set_xticklabels(exponent_labels)
ax5.tick_params(labelsize=fontsize)

ax6 = axs[2, 1]
for i, algorithm in enumerate(sprintz_group):
    data = df_decode_time_reset[df_decode_time_reset['Algorithm'] == algorithm]
    linestyle = '-' if 'All' in algorithm or 'RMQ' not in algorithm else '--'
    ax6.plot(data['Pack Size'], data['Decoding Time (MB/s)'],
             color=algorithm_palette[algorithm], linestyle=linestyle, marker=markers[i+3],
             markersize=7, linewidth=2.2, label=algorithm)
if sprintz_all_decode_mean is not None:
    ax6.axhline(y=sprintz_all_decode_mean, color='#8c564b', linestyle='--', linewidth=1.8, label='Sprintz-All')
if 'sprintz_prune_decode_mean' in globals() and sprintz_prune_decode_mean is not None:
    ax6.axhline(y=sprintz_prune_decode_mean, color='#1f77b4', linestyle='-.', linewidth=1.8, label='Sprintz-Prune')
if sprintz_rmq_decode_mean is not None:
    ax6.axhline(y=sprintz_rmq_decode_mean, color='#17becf', linestyle='--', linewidth=1.8, label='Sprintz-Prune-RMQ')
ax6.set_xscale('log', base=2)
# ax6.set_yscale('log')
ax6.set_ylim(0,1210)
ax6.set_ylabel('Time (ns/point)', fontsize=fontsize)
ax6.set_xlabel(r'Pack Size $s$', fontsize=fontsize)
ax6.set_title('(f) Sprintz: Decompression Time', fontsize=fontsize,x=0.33)
ax6.set_xticks(vector_sizes)
ax6.set_xticklabels(exponent_labels)
ax6.tick_params(labelsize=fontsize)


# Move legend to the top center of the entire figure
# 收集所有子图中的句柄和标签，去重后在画布顶部居中显示
all_handles = []
all_labels = []
for ax_row in axs:
    for ax in ax_row:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)

by_label = OrderedDict(zip(all_labels, all_handles))
fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4,
           labelspacing=0.1,
            handletextpad=0.1,
            columnspacing=0.1,
           fontsize=fontsize-1, bbox_to_anchor=(0.45, 0.98))

# 为图例在顶部留出空间，然后紧凑布局子图
# plt.tight_layout(rect=[0, 0, 1, 0.92])

# 保存图片
output_dir = "/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/figure_for_paper"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, 'bp_vary_pack_size.png'), dpi=400, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'bp_vary_pack_size.eps'), format='eps', dpi=400, bbox_inches='tight')
#legend改为正上方


# 显示图形
# plt.show()

# 创建更详细的统计信息表格
print("\n详细统计信息:")
print("="*70)
print(f"{'Pack Size':>10} {'Compression Ratio':>20} {'Encode Throughput':>20} {'Decode Throughput':>20}")
print("-"*70)

for i, size in enumerate(vector_sizes):
    ratio = avg_compression_ratio['BP'][i]
    encode_tp = avg_encode_time['BP'][i]
    decode_tp = avg_decode_time['BP'][i]
    print(f"{size:>10} {ratio:>20.4f} {encode_tp:>20.2f} {decode_tp:>20.2f}")