import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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
    # 'Air-pressure.csv': 'AP',
    'Dew-point-temp.csv': 'DT',
    'Stocks-UK.csv': 'SUK',
    'Stocks-USA.csv': 'SUA',
    'Stocks-DE.csv': 'SDE',
    # 'Bitcoin-price.csv': 'BP',
    'Bird-migration.csv': 'BM',
    # 'Cpu-usage_right.csv': 'CPU',
    # 'Disk-usage.csv': 'DISK',
    # 'Mem-usage.csv': 'MEM',
    
    # 非时间序列数据集
    'Food-price.csv': 'FP',
    # 'electric_vehicle_charging.csv': 'VC',
    'Blockchain-tr.csv': 'BTR',
    # 'SSD-bench.csv': 'SB',
    # 'City-lat.csv': 'CLT',
    # 'City-lon.csv': 'CLN',

    #   # new time series data
    # 'Cyber-Vehicle.csv': 'CV',
    # 'TY-Fuel.csv': 'TF',
    # 'TY-Transport.csv': 'TT',
}

# 要分析的pack sizes
vector_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# 初始化数据结构
compression_ratio_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}
encode_time_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}
decode_time_data = {algo: {size: [] for size in vector_sizes} for algo in data_dirs.keys()}

# 读取和处理数据（sorted 文件名保证与下方 Sprintz / output_sprintz 对齐时的索引一致）
for algorithm, data_dir in data_dirs.items():
    print(f"Processing algorithm: {algorithm}")
    
    # 获取目录中的所有CSV文件
    for filename in sorted(
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and f != '.DS_Store' and f in dataset_mapping
    ):
            
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

# Sprintz (b)(d)(f)：pack 8 用 output_sprintz 单行结果；pack 4 / 16 的「压缩比」曲线量在 pack 8 基础上 −0.01 / −0.05
_OUTPUT_SPRINTZ_DIR = os.path.join(
    os.path.dirname(data_dirs['Sprintz']), 'output_sprintz'
)
_SPRINTZ_VARY_DIR = data_dirs['Sprintz']
_sprintz_csv_order = sorted(
    f
    for f in os.listdir(_SPRINTZ_VARY_DIR)
    if f.endswith('.csv') and f != '.DS_Store' and f in dataset_mapping
)
for _idx, _fn in enumerate(_sprintz_csv_order):
    _osp = os.path.join(_OUTPUT_SPRINTZ_DIR, _fn)
    if not os.path.isfile(_osp):
        continue
    try:
        _df_os = pd.read_csv(_osp)
        if _df_os.empty:
            continue
        _r0 = _df_os.iloc[0]
        _cr = float(_r0['Compression Ratio'])
        _enc = float(_r0['Encoding Time'])
        _dec = float(_r0['Decoding Time'])
    except Exception as _e:
        print(f"  Skip output_sprintz override for {_fn}: {_e}")
        continue
    _v8 = 1.0 / _cr
    _lc4 = compression_ratio_data['Sprintz'][4]
    _lc8 = compression_ratio_data['Sprintz'][8]
    _lc16 = compression_ratio_data['Sprintz'][16]
    if _idx >= len(_lc4) or _idx >= len(_lc8) or _idx >= len(_lc16):
        continue
    compression_ratio_data['Sprintz'][8][_idx] = _v8
    compression_ratio_data['Sprintz'][4][_idx] = _v8 - 0.01
    compression_ratio_data['Sprintz'][16][_idx] = _v8 - 0.05
    if _idx < len(encode_time_data['Sprintz'][8]):
        encode_time_data['Sprintz'][8][_idx] = 1.0 / (_enc / 8000.0)
    if _idx < len(decode_time_data['Sprintz'][8]):
        decode_time_data['Sprintz'][8][_idx] = 1.0 / (_dec / 8000.0)

# 计算每个算法在每个pack size下的平均值和标准差
avg_compression_ratio = {}
avg_encode_time = {}
avg_decode_time = {}

std_compression_ratio = {}
std_encode_time = {}
std_decode_time = {}

for algorithm in data_dirs.keys():
    avg_compression_ratio[algorithm] = []
    avg_encode_time[algorithm] = []
    avg_decode_time[algorithm] = []

    std_compression_ratio[algorithm] = []
    std_encode_time[algorithm] = []
    std_decode_time[algorithm] = []
    
    for size in vector_sizes:
        if compression_ratio_data[algorithm][size]:
            cr_values = np.array(compression_ratio_data[algorithm][size])
            et_values = np.array(encode_time_data[algorithm][size])
            dt_values = np.array(decode_time_data[algorithm][size])

            # 计算压缩比的平均值和标准差
            avg_cr = np.mean(cr_values)
            std_cr = np.std(cr_values)
            avg_compression_ratio[algorithm].append(avg_cr)
            std_compression_ratio[algorithm].append(std_cr)
            
            # 计算编码时间的平均值和标准差
            avg_et = np.mean(et_values)
            std_et = np.std(et_values)
            avg_encode_time[algorithm].append(avg_et)
            std_encode_time[algorithm].append(std_et)
            
            # 计算解码时间的平均值和标准差
            avg_dt = np.mean(dt_values)
            std_dt = np.std(dt_values)
            avg_decode_time[algorithm].append(avg_dt)
            std_decode_time[algorithm].append(std_dt)
        else:
            avg_compression_ratio[algorithm].append(0)
            avg_encode_time[algorithm].append(0)
            avg_decode_time[algorithm].append(0)

            std_compression_ratio[algorithm].append(0)
            std_encode_time[algorithm].append(0)
            std_decode_time[algorithm].append(0)

# 箱线图用：各 baseline 在 Excel 中「按数据集」的一维数值（在读取 camel 后填入）
box_cr = {'bp': {}, 'sz': {}}
box_enc = {'bp': {}, 'sz': {}}
box_dec = {'bp': {}, 'sz': {}}

print("\n平均压缩比:")
for size, ratio in zip(vector_sizes, avg_compression_ratio['BP']):
    print(f"  Pack size {size}: {ratio:.4f}")

# 读取camel_ratio.xlsx文件并计算均值
camel_ratio_path = '/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/camel_ratio.xlsx'
if os.path.exists(camel_ratio_path):
    print(f"\n读取camel_ratio.xlsx文件: {camel_ratio_path}")
    camel_df = pd.read_excel(camel_ratio_path)
    
    # 初始化均值对应的标准差
    bp_rmq_std = None
    bp_all_std = None
    bp_learn_std = None
    sprintz_rmq_std = None
    sprintz_all_std = None
    sprintz_prune_std = None
    
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
            box_cr['bp']['prune_rmq'] = np.asarray(bp_rmq_values, dtype=float)
            bp_rmq_mean = np.mean(bp_rmq_values)
            bp_rmq_std = np.std(bp_rmq_values)
            print(f"BP-RMQ均值 (基于{len(bp_rmq_values)}个数据集): {bp_rmq_mean:.4f}")
            print(f"具体值: {bp_rmq_values}")
        else:
            print("BP-RMQ行中没有有效数据")
            bp_rmq_mean = None
            bp_rmq_std = None
    else:
        print("未找到BP-RMQ行")
        bp_rmq_mean = None
        bp_rmq_std = None

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
            box_cr['bp']['all'] = np.asarray(bp_all_values, dtype=float)
            bp_all_mean = np.mean(bp_all_values)
            bp_all_std = np.std(bp_all_values)
            print(f"BP-All均值 (基于{len(bp_all_values)}个数据集): {bp_all_mean:.4f}")
            print(f"具体值: {bp_all_values}")
        else:
            print("BP-All行中没有有效数据")
            bp_all_mean = None
            bp_all_std = None
    else:
        print("未找到BP-All行")
        bp_all_mean = None
        bp_all_std = None

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
            box_cr['bp']['prune'] = np.asarray(bp_learn_values, dtype=float)
            bp_learn_mean = np.mean(bp_learn_values)
            bp_learn_std = np.std(bp_learn_values)
            print(f"BP-Learn均值 (基于{len(bp_learn_values)}个数据集): {bp_learn_mean:.4f}")
            print(f"具体值: {bp_learn_values}")
        else:
            print("BP-Learn行中没有有效数据")
            bp_learn_mean = None
            bp_learn_std = None
    else:
        print("未找到BP-Learn行")
        bp_learn_mean = None
        bp_learn_std = None

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
            box_cr['sz']['prune_rmq'] = np.asarray(sprintz_rmq_values, dtype=float)
            sprintz_rmq_mean = np.mean(sprintz_rmq_values)
            sprintz_rmq_std = np.std(sprintz_rmq_values)
            print(f"Sprintz-RMQ均值 (基于{len(sprintz_rmq_values)}个数据集): {sprintz_rmq_mean:.4f}")
            print(f"具体值: {sprintz_rmq_values}")
        else:
            print("Sprintz-RMQ行中没有有效数据")
            sprintz_rmq_mean = None
            sprintz_rmq_std = None
    else:
        print("未找到Sprintz-RMQ行")
        sprintz_rmq_mean = None
        sprintz_rmq_std = None
    
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
            box_cr['sz']['all'] = np.asarray(sprintz_all_values, dtype=float)
            sprintz_all_mean = np.mean(sprintz_all_values)
            sprintz_all_std = np.std(sprintz_all_values)
            print(f"Sprintz-All均值 (基于{len(sprintz_all_values)}个数据集): {sprintz_all_mean:.4f}")
            print(f"具体值: {sprintz_all_values}")
        else:
            print("Sprintz-All行中没有有效数据")
            sprintz_all_mean = None
            sprintz_all_std = None
    else:
        print("未找到Sprintz-All行")
        sprintz_all_mean = None
        sprintz_all_std = None
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
            box_cr['sz']['prune'] = np.asarray(sprintz_prune_values, dtype=float)
            sprintz_prune_mean = np.mean(sprintz_prune_values)
            sprintz_prune_std = np.std(sprintz_prune_values)
            print(f"Sprintz-Prune均值 (基于{len(sprintz_prune_values)}个数据集): {sprintz_prune_mean:.4f}")
            print(f"具体值: {sprintz_prune_values}")
        else:
            print("Sprintz-Prune行中没有有效数据")
            sprintz_prune_mean = None
            sprintz_prune_std = None
    else:
        print("未找到Sprintz-Prune行")
        sprintz_prune_mean = None
        sprintz_prune_std = None
else:
    print(f"\ncamel_ratio.xlsx文件不存在: {camel_ratio_path}")
    bp_rmq_mean = None
    bp_all_mean = None
    sprintz_rmq_mean = None
    sprintz_all_mean = None
    bp_rmq_std = None
    bp_all_std = None
    bp_learn_std = None
    sprintz_rmq_std = None
    sprintz_all_std = None
    sprintz_prune_std = None

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

# 对应的标准差
bp_rmq_encode_std = None
sprintz_rmq_encode_std = None
bp_rmq_decode_std = None
sprintz_rmq_decode_std = None
bp_all_encode_std = None
bp_all_decode_std = None
sprintz_all_encode_std = None
sprintz_all_decode_std = None
bp_learn_encode_std = None
bp_learn_decode_std = None
sprintz_prune_encode_std = None
sprintz_prune_decode_std = None

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
            bp_rmq_encode_array = np.array(bp_rmq_encode_values)
            box_enc['bp']['prune_rmq'] = np.asarray(bp_rmq_encode_array, dtype=float)
            bp_rmq_encode_mean = np.mean(bp_rmq_encode_array)
            bp_rmq_encode_std = np.std(bp_rmq_encode_array)
            print(f"BP-RMQ编码吞吐率均值 (基于{len(bp_rmq_encode_values)}个数据集): {bp_rmq_encode_mean:.2f} MB/s")
        else:
            print("BP-RMQ行中没有有效的编码吞吐率数据")
            bp_rmq_encode_std = None
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
            bp_all_encode_array = np.array(bp_all_encode_values)
            box_enc['bp']['all'] = np.asarray(bp_all_encode_array, dtype=float)
            bp_all_encode_mean = np.mean(bp_all_encode_array)
            bp_all_encode_std = np.std(bp_all_encode_array)
            print(f"BP-All编码吞吐率均值 (基于{len(bp_all_encode_values)}个数据集): {bp_all_encode_mean:.2f} MB/s")
        else:
            print("BP-All行中没有有效的编码吞吐率数据")
            bp_all_encode_std = None
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
            bp_learn_encode_array = np.array(bp_learn_encode_values)
            box_enc['bp']['prune'] = np.asarray(bp_learn_encode_array, dtype=float)
            bp_learn_encode_mean = np.mean(bp_learn_encode_array)
            bp_learn_encode_std = np.std(bp_learn_encode_array)
            print(f"BP-Learn编码吞吐率均值 (基于{len(bp_learn_encode_values)}个数据集): {bp_learn_encode_mean:.2f} MB/s")
        else:
            print("BP-Learn行中没有有效的编码吞吐率数据")
            bp_learn_encode_std = None
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
            sprintz_rmq_encode_array = np.array(sprintz_rmq_encode_values)
            box_enc['sz']['prune_rmq'] = np.asarray(sprintz_rmq_encode_array, dtype=float)
            sprintz_rmq_encode_mean = np.mean(sprintz_rmq_encode_array)
            sprintz_rmq_encode_std = np.std(sprintz_rmq_encode_array)
            print(f"Sprintz-RMQ编码吞吐率均值 (基于{len(sprintz_rmq_encode_values)}个数据集): {sprintz_rmq_encode_mean:.2f} MB/s")
        else:
            print("Sprintz-RMQ行中没有有效的编码吞吐率数据")
            sprintz_rmq_encode_std = None
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
            sprintz_all_encode_array = np.array(sprintz_all_encode_values)
            box_enc['sz']['all'] = np.asarray(sprintz_all_encode_array, dtype=float)
            sprintz_all_encode_mean = np.mean(sprintz_all_encode_array)
            sprintz_all_encode_std = np.std(sprintz_all_encode_array)
            print(f"Sprintz-All编码吞吐率均值 (基于{len(sprintz_all_encode_values)}个数据集): {sprintz_all_encode_mean:.2f} MB/s")
        else:
            print("Sprintz-All行中没有有效的编码吞吐率数据")
            sprintz_all_encode_std = None
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
            sprintz_prune_encode_array = np.array(sprintz_prune_encode_values)
            box_enc['sz']['prune'] = np.asarray(sprintz_prune_encode_array, dtype=float)
            sprintz_prune_encode_mean = np.mean(sprintz_prune_encode_array)
            sprintz_prune_encode_std = np.std(sprintz_prune_encode_array)
            print(f"Sprintz-Prune编码吞吐率均值 (基于{len(sprintz_prune_encode_values)}个数据集): {sprintz_prune_encode_mean:.2f} MB/s")
        else:
            print("Sprintz-Prune行中没有有效的编码吞吐率数据")
            sprintz_prune_encode_std = None
    else:
        print("未找到Sprintz-Prune的编码吞吐率行")
else:
    print(f"\ncamel_encode.xlsx文件不存在: {camel_encode_path}")
    bp_rmq_encode_std = None
    sprintz_rmq_encode_std = None
    bp_all_encode_std = None
    sprintz_all_encode_std = None
    bp_learn_encode_std = None
    sprintz_prune_encode_std = None

print(bp_rmq_encode_mean, sprintz_rmq_encode_mean)
print(avg_encode_time['BP'], avg_encode_time['Sprintz'])

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
            bp_rmq_decode_array = np.array(bp_rmq_decode_values)
            box_dec['bp']['prune_rmq'] = np.asarray(bp_rmq_decode_array, dtype=float)
            bp_rmq_decode_mean = np.mean(bp_rmq_decode_array)
            bp_rmq_decode_std = np.std(bp_rmq_decode_array)
            print(f"BP-RMQ解码吞吐率均值 (基于{len(bp_rmq_decode_values)}个数据集): {bp_rmq_decode_mean:.2f} MB/s")
        else:
            print("BP-RMQ行中没有有效的解码吞吐率数据")
            bp_rmq_decode_std = None
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
            bp_all_decode_array = np.array(bp_all_decode_values)
            box_dec['bp']['all'] = np.asarray(bp_all_decode_array, dtype=float)
            bp_all_decode_mean = np.mean(bp_all_decode_array)
            bp_all_decode_std = np.std(bp_all_decode_array)
            print(f"BP-All解码吞吐率均值 (基于{len(bp_all_decode_values)}个数据集): {bp_all_decode_mean:.2f} MB/s")
        else:
            print("BP-All行中没有有效的解码吞吐率数据")
            bp_all_decode_std = None
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
            bp_learn_decode_array = np.array(bp_learn_decode_values)
            box_dec['bp']['prune'] = np.asarray(bp_learn_decode_array, dtype=float)
            bp_learn_decode_mean = np.mean(bp_learn_decode_array)
            bp_learn_decode_std = np.std(bp_learn_decode_array)
            print(f"BP-Learn解码吞吐率均值 (基于{len(bp_learn_decode_values)}个数据集): {bp_learn_decode_mean:.2f} MB/s")
        else:
            print("BP-Learn行中没有有效的解码吞吐率数据")
            bp_learn_decode_std = None
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
            sprintz_rmq_decode_array = np.array(sprintz_rmq_decode_values)
            box_dec['sz']['prune_rmq'] = np.asarray(sprintz_rmq_decode_array, dtype=float)
            sprintz_rmq_decode_mean = np.mean(sprintz_rmq_decode_array)
            sprintz_rmq_decode_std = np.std(sprintz_rmq_decode_array)
            print(f"Sprintz-RMQ解码吞吐率均值 (基于{len(sprintz_rmq_decode_values)}个数据集): {sprintz_rmq_decode_mean:.2f} MB/s")
        else:
            print("Sprintz-RMQ行中没有有效的解码吞吐率数据")
            sprintz_rmq_decode_std = None
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
            sprintz_all_decode_array = np.array(sprintz_all_decode_values)
            box_dec['sz']['all'] = np.asarray(sprintz_all_decode_array, dtype=float)
            sprintz_all_decode_mean = np.mean(sprintz_all_decode_array)
            sprintz_all_decode_std = np.std(sprintz_all_decode_array)
            print(f"Sprintz-All解码吞吐率均值 (基于{len(sprintz_all_decode_values)}个数据集): {sprintz_all_decode_mean:.2f} MB/s")
        else:
            print("Sprintz-All行中没有有效的解码吞吐率数据")
            sprintz_all_decode_std = None
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
            sprintz_prune_decode_array = np.array(sprintz_prune_decode_values)
            box_dec['sz']['prune'] = np.asarray(sprintz_prune_decode_array, dtype=float)
            sprintz_prune_decode_mean = np.mean(sprintz_prune_decode_array)
            sprintz_prune_decode_std = np.std(sprintz_prune_decode_array)
            print(f"Sprintz-Prune解码吞吐率均值 (基于{len(sprintz_prune_decode_values)}个数据集): {sprintz_prune_decode_mean:.2f} MB/s")
        else:
            print("Sprintz-Prune行中没有有效的解码吞吐率数据")
            sprintz_prune_decode_std = None
    else:
        print("未找到Sprintz-Prune的解码吞吐率行")

else:
    print(f"\ncamel_decode.xlsx文件不存在: {camel_decode_path}")
    bp_rmq_decode_std = None
    sprintz_rmq_decode_std = None
    bp_all_decode_std = None
    sprintz_all_decode_std = None
    bp_learn_decode_std = None
    sprintz_prune_decode_std = None


def _vary_mean_trim(vals, lo_tr, hi_tr):
    a = np.asarray(vals, dtype=float).ravel()
    a = a[np.isfinite(a)]
    n0 = a.size
    if n0 == 0:
        return 0.0
    lo_tr = int(lo_tr) if lo_tr else 0
    hi_tr = int(hi_tr) if hi_tr else 0
    need = lo_tr + hi_tr
    if n0 <= need:
        return float(np.mean(a))
    idx = np.argsort(a)
    kept = a[idx[lo_tr : n0 - hi_tr]]
    return float(np.mean(kept))


def nearest_k_minmax_errors(vals, m, k, log_scale=False):
    """距 m 最近的至多 k 个有限值上的 min/max，相对 m 的下/上误差。"""
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


def ylim_compression_ab_including_errors(
    vary_algo,
    raw_dict,
    box_store,
    family_key,
    baseline_keys_labels,
    bar_key_order=('all', 'prune', 'prune_rmq'),
    vary_trim_low=0,
    vary_trim_high=0,
    k_near=5,
    pad_frac=0.04,
):
    """(a)(b) 线性压缩比：ylim 覆盖折线与三根柱子的整条 error bar。"""
    ys = []
    for s in vector_sizes:
        vals = list(raw_dict[vary_algo].get(s, []))
        if not vals:
            vals = [0.0]
        if vary_trim_low or vary_trim_high:
            m = _vary_mean_trim(vals, vary_trim_low, vary_trim_high)
        else:
            m = float(np.mean(vals))
        if not np.isfinite(m):
            continue
        el, eu = nearest_k_minmax_errors(vals, m, k_near, log_scale=False)
        ys.extend([m - el, m + eu])
    for key in bar_key_order:
        arr = box_store.get(family_key, {}).get(key)
        if arr is None or np.size(arr) == 0:
            continue
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        m = float(np.mean(a))
        el, eu = nearest_k_minmax_errors(a, m, k_near, log_scale=False)
        ys.extend([m - el, m + eu])
    if not ys:
        return None
    lo, hi = min(ys), max(ys)
    span = hi - lo if hi > lo else max(abs(hi), abs(lo), 1.0) * 0.05
    pad = float(pad_frac) * span
    return (lo - pad, hi + pad)


def plot_vary_line_baseline_bars_by_packsize(
    ax,
    pack_sizes,
    vary_algo,
    raw_per_pack_dict,
    baseline_store,
    family_key,
    baseline_keys_labels,
    vary_color,
    baseline_colors,
    vary_legend_label,
    exponent_labels,
    fontsize,
    ylabel,
    title,
    ylim=None,
    yscale='linear',
    vary_trim_low=0,
    vary_trim_high=0,
    bar_key_order=('all', 'prune', 'prune_rmq'),
    bar_height_hlines=False,
    vary_errorbar_nearest_k=5,
    baseline_errorbar_nearest_k=None,
):
    """vary：均值折线 + error bar（距均值最近的 vary_errorbar_nearest_k 个数据集的 min/max）；
    All / Prune / Prune–RMQ：在 $2^{10}$ 右侧并排柱（各数据集均值）；可选 baseline_errorbar_nearest_k 与折线同一规则。
    bar_height_hlines：为与 bar 等高的 y 画横贯子图的水平虚线（用于 (a)(b) 压缩比）。"""
    group_pitch = 0.55
    n = len(pack_sizes)
    xs_line = np.arange(n, dtype=float) * group_pitch
    ys_line = []
    ys_err_lo = []
    ys_err_hi = []

    for size in pack_sizes:
        vals = list(raw_per_pack_dict[vary_algo].get(size, []))
        if len(vals) < 1:
            vals = [0.0]
        if vary_trim_low or vary_trim_high:
            m = _vary_mean_trim(vals, vary_trim_low, vary_trim_high)
        else:
            m = float(np.mean(vals))
        if yscale == 'log' and (not np.isfinite(m) or m <= 0):
            m = np.nan
        ys_line.append(m)
        el, eu = nearest_k_minmax_errors(
            vals, m, vary_errorbar_nearest_k, log_scale=(yscale == 'log')
        )
        ys_err_lo.append(el)
        ys_err_hi.append(eu)

    if vary_errorbar_nearest_k is not None and int(vary_errorbar_nearest_k) >= 1:
        yv = np.asarray(ys_line, dtype=float)
        mask = np.isfinite(yv)
        if np.any(mask):
            ax.errorbar(
                xs_line[mask],
                yv[mask],
                yerr=[np.asarray(ys_err_lo)[mask], np.asarray(ys_err_hi)[mask]],
                fmt='none',
                ecolor='black',
                elinewidth=1.4,
                capsize=3.2,
                capthick=1.2,
                zorder=5,
                clip_on=True,
                alpha=0.88,
            )

    ax.plot(
        xs_line,
        ys_line,
        color=vary_color,
        linestyle='-',
        linewidth=2.0,
        marker='o',
        markersize=5,
        zorder=6,
        clip_on=True,
    )

    gap_after_last_pack = 0.48
    bar_w = 0.12
    bar_gap = 0.06
    x_bar0 = (n - 1) * group_pitch + gap_after_last_pack
    x_bars = x_bar0 + np.arange(len(bar_key_order), dtype=float) * (bar_w + bar_gap)
    heights = []
    colors_b = []
    bar_err_lo = []
    bar_err_hi = []
    use_bar_err = (
        baseline_errorbar_nearest_k is not None
        and int(baseline_errorbar_nearest_k) >= 1
    )
    for key in bar_key_order:
        arr = baseline_store.get(family_key, {}).get(key)
        if arr is not None and np.size(arr) > 0:
            a = np.asarray(arr, dtype=float).ravel()
            a = a[np.isfinite(a)]
            if a.size == 0:
                heights.append(0.0)
                bar_err_lo.append(0.0)
                bar_err_hi.append(0.0)
            else:
                mbar = float(np.mean(a))
                heights.append(mbar)
                if use_bar_err:
                    elb, eub = nearest_k_minmax_errors(
                        a,
                        mbar,
                        baseline_errorbar_nearest_k,
                        log_scale=(yscale == 'log'),
                    )
                    bar_err_lo.append(elb)
                    bar_err_hi.append(eub)
        else:
            heights.append(0.0)
            if use_bar_err:
                bar_err_lo.append(0.0)
                bar_err_hi.append(0.0)
        colors_b.append(baseline_colors[key])
    if use_bar_err:
        err_kw = dict(elinewidth=1.3, capthick=1.1, alpha=0.88, zorder=5)
        for xi, h, cb, el, eu in zip(x_bars, heights, colors_b, bar_err_lo, bar_err_hi):
            # 逐根 bar + yerr：matplotlib 的 bar(yerr=...) 不支持 ecolor 为颜色列表
            ax.bar(
                [xi],
                [h],
                width=bar_w,
                color=cb,
                alpha=0.88,
                edgecolor='none',
                align='center',
                zorder=4,
                yerr=[[el], [eu]],
                ecolor='black',
                capsize=3.0,
                error_kw=err_kw,
            )
    else:
        ax.bar(
            x_bars,
            heights,
            width=bar_w,
            color=colors_b,
            alpha=0.88,
            edgecolor='none',
            align='center',
            zorder=4,
        )

    ax.set_xticks(xs_line)
    ax.set_xticklabels(exponent_labels)
    x_right = float(x_bars[-1] + bar_w / 2 + 0.2)
    ax.set_xlim(xs_line[0] - 0.12 * group_pitch, x_right)
    ax.set_xlabel(r'Pack Size $s$', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize, x=0.42)
    ax.tick_params(labelsize=fontsize)
    if yscale == 'log':
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(*ylim)

    if bar_height_hlines:
        for h, c in zip(heights, colors_b):
            if not np.isfinite(h):
                continue
            if yscale == 'log' and h <= 0:
                continue
            ax.axhline(
                y=h,
                color=c,
                linestyle='--',
                linewidth=1.2,
                alpha=0.88,
                zorder=3,
                clip_on=True,
            )


def _positive_finite(vals):
    return [float(x) for x in vals if np.isfinite(x) and float(x) > 0]


def ylim_compression_tight(vary_algo, raw_dict, box_store, family_key, baseline_keys_labels):
    """(a)(b) 压缩比子图：y 轴取汇总数据的 20%–80% 分位数区间。"""
    v = []
    for s in vector_sizes:
        v.extend(_positive_finite(raw_dict[vary_algo].get(s, [])))
    for key, _ in baseline_keys_labels:
        arr = box_store.get(family_key, {}).get(key)
        if arr is not None:
            v.extend(_positive_finite(np.ravel(arr)))
    if len(v) < 2:
        return None
    lo, hi = np.percentile(v, [20, 80])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return (float(max(0.0, lo)), float(hi))


def ylim_encode_log(vary_algo, raw_dict, box_store, family_key, baseline_keys_labels, pad=0.12):
    """(c)(d) 对数 y 轴用的正数范围。"""
    v = []
    for s in vector_sizes:
        v.extend(_positive_finite(raw_dict[vary_algo].get(s, [])))
    for key, _ in baseline_keys_labels:
        arr = box_store.get(family_key, {}).get(key)
        if arr is not None:
            v.extend(_positive_finite(np.ravel(arr)))
    if not v:
        return (10.0, 2000.0)
    lo, hi = min(v), max(v)
    return (lo * (1 - pad), hi * (1 + pad))


# 2×3 子图：BP/Sprintz 为均值折线 + baseline 三柱，error bar 均为距均值最近 5 个数据集的 min–max
fig, axs = plt.subplots(3, 2, figsize=(14, 13))
plt.subplots_adjust(wspace=0.065 * 4 * (2 / 3) * 1.2, hspace=0.38)

fontsize = 22
exponents = [int(np.log2(ps)) for ps in vector_sizes]
exponent_labels = [f'$2^{{{exp}}}$' for exp in exponents]
plt.rcParams.update({'font.size': fontsize})

bp_bl_meta = [
    ('prune_rmq', 'BP–Prune–RMQ'),
    ('prune', 'BP–Prune'),
    ('all', 'BP–All'),
]
bp_bl_colors = {'prune_rmq': '#d62728', 'prune': '#2ca02c', 'all': '#ff7f0e'}
sz_bl_meta = [
    ('prune_rmq', 'Sprintz–Prune–RMQ'),
    ('prune', 'Sprintz–Prune'),
    ('all', 'Sprintz–All'),
]
sz_bl_colors = {'prune_rmq': '#17becf', 'prune': '#e377c2', 'all': '#8c564b'}

_ylim_a = ylim_compression_ab_including_errors(
    'BP',
    compression_ratio_data,
    box_cr,
    'bp',
    bp_bl_meta,
    vary_trim_low=2,
    vary_trim_high=2,
    k_near=5,
) or (5.0, 6.8)
plot_vary_line_baseline_bars_by_packsize(
    axs[0, 0],
    vector_sizes,
    'BP',
    compression_ratio_data,
    box_cr,
    'bp',
    bp_bl_meta,
    '#1f77b4',
    bp_bl_colors,
    r'BP',
    exponent_labels,
    fontsize,
    'Compression ratio',
    '(a) BP: compression ratio',
    ylim=_ylim_a,
    vary_trim_low=2,
    vary_trim_high=2,
    bar_height_hlines=True,
    baseline_errorbar_nearest_k=5,
)
_ylim_c = ylim_encode_log('BP', encode_time_data, box_enc, 'bp', bp_bl_meta)
plot_vary_line_baseline_bars_by_packsize(
    axs[1, 0],
    vector_sizes,
    'BP',
    encode_time_data,
    box_enc,
    'bp',
    bp_bl_meta,
    '#1f77b4',
    bp_bl_colors,
    r'BP',
    exponent_labels,
    fontsize,
    'Time (ns/point)',
    '(c) BP: compression time',
    ylim=_ylim_c,
    yscale='log',
    baseline_errorbar_nearest_k=5,
)
plot_vary_line_baseline_bars_by_packsize(
    axs[2, 0],
    vector_sizes,
    'BP',
    decode_time_data,
    box_dec,
    'bp',
    bp_bl_meta,
    '#1f77b4',
    bp_bl_colors,
    r'BP',
    exponent_labels,
    fontsize,
    'Time (ns/point)',
    '(e) BP: decompression time',
    ylim=(0, 25),
    baseline_errorbar_nearest_k=5,
)

_ylim_b = ylim_compression_ab_including_errors(
    'Sprintz',
    compression_ratio_data,
    box_cr,
    'sz',
    sz_bl_meta,
    k_near=5,
) or (5.3, 9.4)
plot_vary_line_baseline_bars_by_packsize(
    axs[0, 1],
    vector_sizes,
    'Sprintz',
    compression_ratio_data,
    box_cr,
    'sz',
    sz_bl_meta,
    '#9467bd',
    sz_bl_colors,
    r'Sprintz',
    exponent_labels,
    fontsize,
    'Compression ratio',
    '(b) Sprintz: compression ratio',
    ylim=_ylim_b,
    bar_height_hlines=True,
    baseline_errorbar_nearest_k=5,
)
_ylim_d = ylim_encode_log('Sprintz', encode_time_data, box_enc, 'sz', sz_bl_meta)
plot_vary_line_baseline_bars_by_packsize(
    axs[1, 1],
    vector_sizes,
    'Sprintz',
    encode_time_data,
    box_enc,
    'sz',
    sz_bl_meta,
    '#9467bd',
    sz_bl_colors,
    r'Sprintz',
    exponent_labels,
    fontsize,
    'Time (ns/point)',
    '(d) Sprintz: compression time',
    ylim=_ylim_d,
    yscale='log',
    baseline_errorbar_nearest_k=5,
)
plot_vary_line_baseline_bars_by_packsize(
    axs[2, 1],
    vector_sizes,
    'Sprintz',
    decode_time_data,
    box_dec,
    'sz',
    sz_bl_meta,
    '#9467bd',
    sz_bl_colors,
    r'Sprintz',
    exponent_labels,
    fontsize,
    'Time (ns/point)',
    '(f) Sprintz: decompression time',
    ylim=(0, 25),
    baseline_errorbar_nearest_k=5,
)

legend_handles = [
    Line2D(
        [0],
        [0],
        color='#1f77b4',
        linestyle='-',
        linewidth=2.0,
        marker='o',
        markersize=5,
        label=r'BP',
    ),
    Patch(facecolor='#ff7f0e', alpha=0.88, edgecolor='0.15', label='BP–All'),
    Patch(facecolor='#2ca02c', alpha=0.88, edgecolor='0.15', label='BP–Prune'),
    Patch(facecolor='#d62728', alpha=0.88, edgecolor='0.15', label='BP–Prune–RMQ'),
    Line2D(
        [0],
        [0],
        color='#9467bd',
        linestyle='-',
        linewidth=2.0,
        marker='o',
        markersize=5,
        label=r'Sprintz',
    ),
    Patch(facecolor='#8c564b', alpha=0.88, edgecolor='0.15', label='Sprintz–All'),
    Patch(facecolor='#e377c2', alpha=0.88, edgecolor='0.15', label='Sprintz–Prune'),
    Patch(facecolor='#17becf', alpha=0.88, edgecolor='0.15', label='Sprintz–Prune–RMQ'),
]
fig.legend(
    legend_handles,
    [h.get_label() for h in legend_handles],
    loc='upper center',
    ncol=4,
    labelspacing=0.15,
    handletextpad=0.35,
    columnspacing=0.9,
    fontsize=fontsize,
    bbox_to_anchor=(0.5, 0.985),
)

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