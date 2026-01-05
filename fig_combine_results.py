import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.stats import friedmanchisquare, chi2
from decimal import Decimal
import csv
import math

def compute_log(x):
    """计算 ⌈log2(x+1)⌉（处理x=0的特殊情况）"""
    return math.ceil(math.log2(x + 1)) if x > 0 else 1

def analysis_data():
    numbers = []
    decimal_places = []

    # # Read data and compute decimal places
    # with open(file_path, 'r') as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         for item in row:
    #             num_str = item.strip()
    #             if num_str:
    #                 numbers.append(num_str)
                    # Compute decimal places
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
    from scipy.stats import friedmanchisquare, chi2
    from decimal import Decimal
    import csv
    import math


    # def compute_log(x):
    #     """计算 ⌈log2(x+1)⌉（处理x=0的特殊情况）"""
    #     return math.ceil(math.log2(x + 1)) if x > 0 else 1


    # def analysis_data(file_path):
    #     numbers = []

    #     # Read data and compute decimal places
    #     with open(file_path, 'r') as csvfile:
    #         reader = csv.reader(csvfile)
    #         for row in reader:
    #             for item in row:
    #                 num_str = item.strip()
    #                 if num_str:
    #                     numbers.append(num_str)

    #     return len(numbers)


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


    # 1. 读取数据并预处理
    df_4 = pd.read_excel("./compare_camel/camel_ratio4.xlsx", index_col=0)
    output_dir = "./output_BP"
    rl_results = {}

    output_dir_bp_n2 = "./output_BP_all_no8"
    output_dir_bprmq = "./output_BP_RMQ_all_no8"
    output_dir_bp_tri_search = "./output_BP_Prune_all_no8"
    output_dir_bp_only_prune = "./output_BP_only_Prune_all_no8"
    output_dir_bp_only_prune_plus = "./output_BP_only_Prune_Plus_all_no8"
    output_dir_sprintz_search = "./output_Sprintz_Prune_all_no8"
    output_dir_sprintz_only_prune = "./output_Sprintz_only_Prune_all_no8"
    output_dir_sprintz_only_prune_plus = "./output_Sprintz_only_Prune_Plus_all_no8"
    output_dir_bp = "./output_BP"
    output_dir_bprl = "./output_BP_RF_10F"
    output_dir_sprintz = "./output_sprintz"
    output_dir_sprintz_rmq = "./output_BP_RMQ_all_no8_sprintz"
    output_dir_sprintz_n2 = "./output_Sprintz_N2_all_no8"

    bprl_results = {}
    bpn2_results = {}
    bprmq_results = {}
    bp_tri_search_results = {}
    bp_only_prune_results = {}
    bp_only_prune_plus_results = {}
    sprintz_only_prune_plus_results = {}
    bp_results = {}
    sprintz_results = {}
    sprintz_rmq_results = {}
    sprintz_n2_results = {}
    sprintz_search_results = {}
    bp_learn_results = {}
    sprintz_only_prune_results = {}

    # BP (learn) results from Java evaluator
    bp_learn_results = {}


    dir_camel = "./ElfTestData_camel/"

    dataset_points = {}
    # for dataset_name in dataset_mapping.keys():
    #     file_path = os.path.join(dir_camel, dataset_name)
    #     try:
    #         points = analysis_data(file_path)
    #         dataset_points[dataset_name] = points
    #     except FileNotFoundError:
    #         print(f"Warning: Could not find file {dataset_name}")
    #         continue


    # 2. 读取结果并填表
    for filename in os.listdir(output_dir):
        if filename in dataset_mapping:
            result_path = os.path.join(output_dir, filename)
            try:
                df_rl = pd.read_csv(result_path)
                rl_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
                print(f"Warning: Could not process {filename} in {output_dir}")

            # other algorithms
            try:
                df_rl = pd.read_csv(os.path.join(output_dir_bp_n2, filename))
                bpn2_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass

            try:
                df_rl = pd.read_csv(os.path.join(output_dir_bprl, filename))
                bprl_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass

            try:
                df_rl = pd.read_csv(os.path.join(output_dir_bprmq, filename))
                bprmq_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass
            
            try:
                df_rl = pd.read_csv(os.path.join(output_dir_bp_only_prune, filename))
                bp_only_prune_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass

            try:
                df_rl = pd.read_csv(os.path.join(output_dir_bp_only_prune_plus, filename))
                bp_only_prune_plus_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass
            try:
                df_rl = pd.read_csv(os.path.join(output_dir_sprintz_only_prune_plus, filename))
                sprintz_only_prune_plus_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass
            try:
                df_rl = pd.read_csv(os.path.join(output_dir_bp_tri_search, filename))
                bp_tri_search_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass

            try:
                df_rl = pd.read_csv(os.path.join(output_dir_sprintz_search, filename))
                sprintz_search_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass
            try:
                df_rl = pd.read_csv(os.path.join(output_dir_sprintz, filename))
                sprintz_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass

            try:
                df_rl = pd.read_csv(os.path.join(output_dir_sprintz_rmq, filename))
                sprintz_rmq_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass

            try:
                df_rl = pd.read_csv(os.path.join(output_dir_sprintz_n2, filename))
                sprintz_n2_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass

            try:
                df_rl = pd.read_csv(os.path.join(output_dir_bp, filename))
                bp_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass
            try:
                df_rl = pd.read_csv(os.path.join(output_dir_sprintz_only_prune, filename))
                sprintz_only_prune_results[filename] = {
                    'compression_ratio': round(df_rl['Compression Ratio'].values[0], 3),
                    'encoding_time': df_rl['Encoding Time'].values[0],
                    'decoding_time': df_rl['Decoding Time'].values[0]
                }
            except Exception:
                pass



    # Read learning evaluation CSV if present (produced by PackSizeMLTrainerAndEvaluator)
    learn_path = os.path.join('.', 'learning_evaluation_results.csv')
    if os.path.exists(learn_path):
        try:
            df_learn = pd.read_csv(learn_path)
            for _, row in df_learn.iterrows():
                fname = row.get('InputFile')
                if not isinstance(fname, str):
                    continue
                bp_learn_results[fname] = {
                    'compression_ratio': round(float(row.get('Compression Ratio', 0.0)), 3),
                    'encoding_time': float(row.get('Encoding Time', 0.0)),
                    'decoding_time': float(row.get('Decoding Time', 0.0))
                }
        except Exception:
            print('Warning: failed to read learning_evaluation_results.csv')


    for dataset_name, dataset_abbr in dataset_mapping.items():
        # proceed if we have baseline RL results or learning results
        if dataset_name in rl_results or dataset_name in bp_learn_results:
            df_4.loc['BP-RF', dataset_abbr] = round(bprl_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['Sprintz (RMQ)', dataset_abbr] = round(sprintz_rmq_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['Sprintz (All)', dataset_abbr] = round(sprintz_n2_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['Sprintz (Prune-RMQ)', dataset_abbr] = round(sprintz_search_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['Sprintz (Prune)', dataset_abbr] = round(sprintz_only_prune_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['Sprintz (Prune Plus)', dataset_abbr] = round(sprintz_only_prune_plus_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['Sprintz', dataset_abbr] = round(sprintz_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['BP (All)', dataset_abbr] = round(bpn2_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['BP (RMQ)', dataset_abbr] = round(bprmq_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['BP (Prune)', dataset_abbr] = round(bp_only_prune_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['BP (Prune Plus)', dataset_abbr] = round(bp_only_prune_plus_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['BP (Prune-RMQ)', dataset_abbr] = round(bp_tri_search_results[dataset_name]['compression_ratio'], 3)
            df_4.loc['BP', dataset_abbr] = round(bp_results[dataset_name]['compression_ratio'], 3)
            # BP (learn) from PackSizeMLTrainerAndEvaluator
            if dataset_name in bp_learn_results:
                df_4.loc['BP (learn)', dataset_abbr] = round(bp_learn_results[dataset_name]['compression_ratio'], 3)
            # BP (learn) from Java trainer/evaluator
            

    exclude_datasets = ['BW', 'BT', 'AS', 'PLT', 'PLN']
    df_4 = df_4.loc[:, ~df_4.columns.isin(exclude_datasets)]
    df_4 = df_4.loc[~df_4.index.str.contains("TSDIFF\\+Subcolumn", na=False)]
    df_4 = df_4.loc[~df_4.index.str.contains("Sprintz\\+Subcolumn", na=False)]
    df_4 = df_4.loc[~df_4.index.str.contains("Subcolumn", na=False)]


    weighted_sums = {}
    for algorithm in df_4.index:
        total_sum = 0
        total_weight = 0

        for dataset_name, dataset_abbr in dataset_mapping.items():
            if dataset_abbr in df_4.columns and dataset_name in dataset_points:
                ratio = df_4.loc[algorithm, dataset_abbr]
                weight = dataset_points[dataset_name]
                if pd.notna(ratio):
                    total_sum += ratio * weight
                    total_weight += weight

        if total_weight > 0:
            weighted_average = total_sum / total_weight
            weighted_sums[algorithm] = weighted_average
        else:
            weighted_sums[algorithm] = None

    df_4['avg_ratio'] = pd.Series(weighted_sums)
    df_4.to_excel("camel_ratio.xlsx")

    df_time_4 = pd.read_excel("./compare_camel/camel_compression_rate4.xlsx", index_col=0)

    for dataset_name, dataset_abbr in dataset_mapping.items():
        if dataset_name in rl_results:
            df_time_4.loc['BP-RF', dataset_abbr] = bprl_results[dataset_name]['encoding_time']
            df_time_4.loc['BP (All)', dataset_abbr] = bpn2_results[dataset_name]['encoding_time']
            df_time_4.loc['BP (RMQ)', dataset_abbr] = bprmq_results[dataset_name]['encoding_time']
            df_time_4.loc['BP (Prune)', dataset_abbr] = bp_only_prune_results[dataset_name]['encoding_time']
            df_time_4.loc['BP (Prune-RMQ)', dataset_abbr] = bp_tri_search_results[dataset_name]['encoding_time']
            df_time_4.loc['BP (Prune Plus)', dataset_abbr] = bp_only_prune_plus_results[dataset_name]['encoding_time']

            df_time_4.loc['Sprintz (RMQ)', dataset_abbr] = sprintz_rmq_results[dataset_name]['encoding_time']
            df_time_4.loc['Sprintz (All)', dataset_abbr] = sprintz_n2_results[dataset_name]['encoding_time']
            df_time_4.loc['Sprintz (Prune-RMQ)', dataset_abbr] = sprintz_search_results[dataset_name]['encoding_time']
            df_time_4.loc['Sprintz (Prune)', dataset_abbr] = sprintz_only_prune_results[dataset_name]['encoding_time']
            df_time_4.loc['Sprintz', dataset_abbr] = sprintz_results[dataset_name]['encoding_time']
            df_time_4.loc['Sprintz (Prune Plus)', dataset_abbr] = sprintz_only_prune_plus_results[dataset_name]['encoding_time']
            df_time_4.loc['BP', dataset_abbr] = bp_results[dataset_name]['encoding_time']
            if dataset_name in bp_learn_results:
                df_time_4.loc['BP (learn)', dataset_abbr] = bp_learn_results[dataset_name]['encoding_time']


    df_time_4 = df_time_4.loc[:, ~df_time_4.columns.isin(exclude_datasets)]
    df_time_4 = df_time_4.loc[~df_time_4.index.str.contains("TSDIFF\\+Subcolumn", na=False)]
    df_time_4 = df_time_4.loc[~df_time_4.index.str.contains("Sprintz\\+Subcolumn", na=False)]
    df_time_4 = df_time_4.loc[~df_time_4.index.str.contains("Subcolumn", na=False)]
    df_time_4.to_excel("compression_time.xlsx")

    df_decompression_time = pd.read_excel("./compare_camel/camel_decompression_rate.xlsx", index_col=0)
    for dataset_name, dataset_abbr in dataset_mapping.items():
        if dataset_name in rl_results:
            df_decompression_time.loc['BP-RF', dataset_abbr] = bprl_results[dataset_name]['decoding_time']
            df_decompression_time.loc['BP (All)', dataset_abbr] = bpn2_results[dataset_name]['decoding_time']
            df_decompression_time.loc['BP (RMQ)', dataset_abbr] = bprmq_results[dataset_name]['decoding_time']
            df_decompression_time.loc['BP (Prune)', dataset_abbr] = bp_only_prune_results[dataset_name]['decoding_time']
            df_decompression_time.loc['BP (Prune-RMQ)', dataset_abbr] = bp_tri_search_results[dataset_name]['decoding_time']
            df_decompression_time.loc['BP (Prune Plus)', dataset_abbr] = bp_only_prune_plus_results[dataset_name]['decoding_time']

            df_decompression_time.loc['BP', dataset_abbr] = bp_results[dataset_name]['decoding_time']
            if dataset_name in bp_learn_results:
                df_decompression_time.loc['BP (learn)', dataset_abbr] = bp_learn_results[dataset_name]['decoding_time']
            df_decompression_time.loc['Sprintz (RMQ)', dataset_abbr] = sprintz_rmq_results[dataset_name]['decoding_time']
            df_decompression_time.loc['Sprintz (All)', dataset_abbr] = sprintz_n2_results[dataset_name]['decoding_time']
            df_decompression_time.loc['Sprintz (Prune-RMQ)', dataset_abbr] = sprintz_search_results[dataset_name]['decoding_time']
            df_decompression_time.loc['Sprintz (Prune)', dataset_abbr] = sprintz_only_prune_results[dataset_name]['decoding_time']
            df_decompression_time.loc['Sprintz (Prune Plus)', dataset_abbr] = sprintz_only_prune_plus_results[dataset_name]['decoding_time']
            df_decompression_time.loc['Sprintz', dataset_abbr] = sprintz_results[dataset_name]['decoding_time']

        

    df_decompression_time = df_decompression_time.loc[:, ~df_decompression_time.columns.isin(exclude_datasets)]
    df_decompression_time = df_decompression_time.loc[~df_decompression_time.index.str.contains("TSDIFF\\+Subcolumn", na=False)]
    df_decompression_time = df_decompression_time.loc[~df_decompression_time.index.str.contains("Sprintz\\+Subcolumn", na=False)]
    df_decompression_time = df_decompression_time.loc[~df_decompression_time.index.str.contains("Subcolumn", na=False)]
    df_decompression_time.to_excel("decompression_time.xlsx")

if __name__ == "__main__":
    csv_dir = "/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/packsize_cost_analysis"
    output_dir = "./packsize_cost_plots"
    
    analysis_data()