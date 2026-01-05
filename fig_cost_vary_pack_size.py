import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# set default font size to 12 for all plots
plt.rcParams.update({'font.size': 20})

def create_simple_plots(csv_dir, output_dir):
    """
    为每个CSV文件创建简单的packsize-cost折线图
    
    Args:
        csv_dir: CSV文件目录
        output_dir: 输出图片目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_cost.csv')]
    
    for csv_file in csv_files:
        print(f"处理: {csv_file}")
        
        # 读取数据
        df = pd.read_csv(os.path.join(csv_dir, csv_file))
        
        if 'packsize' not in df.columns or 'cost' not in df.columns:
            continue
        
        # 按packsize分组计算均值
        grouped = df.groupby('packsize')['cost'].mean().reset_index()
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        plt.plot(grouped['packsize'], grouped['cost'], 
                linewidth=2, marker='o', markersize=4)
        
        # 标记最小值
        min_idx = grouped['cost'].idxmin()
        min_packsize = grouped.loc[min_idx, 'packsize']
        min_cost = grouped.loc[min_idx, 'cost']
        
        plt.scatter([min_packsize], [min_cost], 
                   color='red', s=100, zorder=5,
                   label=f'最优: packsize={min_packsize}, cost={min_cost:.0f}')
        
        plt.axvline(x=min_packsize, color='red', linestyle='--', alpha=0.5)
        
        # 设置图表属性
        plt.xlabel('Packsize')
        plt.ylabel('Average Cost (bits)')
        plt.title(f'{csv_file.replace("_cost.csv", "")}\nPacksize vs Cost')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 调整布局并保存
        plt.tight_layout()
        
        output_file = csv_file.replace('.csv', '.png')
        plt.savefig(os.path.join(output_dir, output_file), dpi=150)
        plt.close()
        
        print(f"  图表已保存: {output_file}")
        print(f"  最优packsize: {min_packsize}, 最小cost: {min_cost:.0f}")

def fig_of_cost_values_bitwidth_in_chunk(csv_dir, output_dir, chunk_size=1024):
    """
    For each CSV in csv_dir (matching *_cost.csv), split rows into chunks of `chunk_size`
    and draw a line plot for each chunk (packsize vs cost). Save plots into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)



    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_cost.csv')]

    for csv_file in csv_files:
        # print(f"Chunk plotting: {csv_file}")
        if not csv_file.startswith('PM10-dust'):
            continue
        path = os.path.join(csv_dir, csv_file)
        print(f"Chunk plotting: {csv_file}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  Failed to read {csv_file}: {e}")
            continue

        if 'pack size' not in df.columns or 'cost' not in df.columns:
            print(f"  Skipping {csv_file}: missing 'packsize' or 'cost' columns")
            continue

        n = len(df)
        # num_chunks = (n + chunk_size - 1) // chunk_size
        num_chunks = 1

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(n, (i + 1) * chunk_size)
            sub = df.iloc[start:end]
            if sub.empty:
                continue

            # Prepare x/y (sort by packsize for clearer line)
            sub_sorted = sub.sort_values(by='pack size')
            x = sub_sorted['pack size'].values
            y = sub_sorted['cost'].values
            y1 = sub_sorted['bitwidth_cost'].values
            y2 = sub_sorted['value_cost'].values

            # 原始图
            fontsize = 16
            ax =plt.figure(figsize=(8, 6))
            colors = [
                "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500", "#800080",
                "#008000", "#FFC0CB", "#A52A2A", "#808080", "#000000", "#FFD700", "#ADFF2F", "#FF4500",
                "#DA70D6", "#1E90FF", "#FF6347", "#7CFC00", "#8A2BE2", "#DC143C", "#FFFFFF"
                "#9932CC", "#8B0000", "#2E8B57", "#DAA520", "#4B0082", "#808000"
            ]
            # 第一个子图：原始数据
            # plt.subplot(2, 1, 1)
            plt.plot(x, y, linestyle='-', marker='o', markersize=3,color=colors[0], label='Total storage cost')

            # 得到y最小值及对应的x值
            min_idx = np.nanargmin(y)
            min_x = x[min_idx]
            print(f"  最小总成本点: pack size={min_x}, cost={y[min_idx]:.2f}, bitwidth cost={y1[min_idx]:.2f}, value cost={y2[min_idx]:.2f}")
            plt.plot(x, y1, linestyle='--', marker='x', markersize=3,color=colors[1], label='Bit width cost')
            plt.plot(x, y2, linestyle='--', marker='s', markersize=3,color=colors[2], label='Value cost')

            # add two guiding straight lines through the global minimum of (x,y):
            # 1) an increasing line passing through the min point that stays strictly
            #    below all points to the right of the min;
            # 2) a decreasing line passing through the min point that stays strictly
            #    below all points to the left of the min.
            # try:
            #     # x and y are numpy arrays sorted by x
            #     idx_min = int(np.nanargmin(y))
            #     x_min = float(x[idx_min])
            #     y_min = float(y[idx_min])

            #     # right side (x > x_min): slope must be < min( (y_i - y_min)/(x_i - x_min) )
            #     right_mask = x > x_min
            #     right_mask[idx_min] = False
            #     if np.any(right_mask):
            #         dx_right = x[right_mask] - x_min
            #         dy_right = y[right_mask] - y_min
            #         # avoid division by zero
            #         valid = dx_right != 0
            #         if np.any(valid):
            #             ratios_right = dy_right[valid] / dx_right[valid]
            #             min_ratio_right = np.min(ratios_right)
            #             # subtract a tiny epsilon to ensure strict inequality
            #             eps = max(1e-6, abs(min_ratio_right) * 1e-6)
            #             slope_right = min_ratio_right - eps
            #             xr = np.linspace(x_min, float(x.max()), 100)
            #             yr = y_min + slope_right * (xr - x_min)
            #             plt.plot(xr, yr, linestyle='--', color='k', linewidth=1.5)

            #     # left side (x < x_min): slope must be > max( (y_i - y_min)/(x_i - x_min) )
            #     left_mask = x < x_min
            #     left_mask[idx_min] = False
            #     if np.any(left_mask):
            #         dx_left = x[left_mask] - x_min
            #         dy_left = y[left_mask] - y_min
            #         valid = dx_left != 0
            #         if np.any(valid):
            #             ratios_left = dy_left[valid] / dx_left[valid]
            #             max_ratio_left = np.max(ratios_left)
            #             eps = max(1e-6, abs(max_ratio_left) * 1e-6)
            #             slope_left = max_ratio_left + eps
            #             xl = np.linspace(float(x.min()), x_min, 100)
            #             yl = y_min + slope_left * (xl - x_min)
            #             plt.plot(xl, yl, linestyle='--', color='k', linewidth=1.5)

            #     # highlight the minimum point
            #     plt.scatter([x_min], [y_min], color='black', s=40, zorder=6)
            # except Exception:
            #     # if something goes wrong, skip drawing the guide lines
            #     pass
            plt.xlabel(r'Pack size $s$', fontsize=fontsize)
            plt.ylabel('Cost (bits)', fontsize=fontsize)
            # plt.title(f"{csv_file.split('-')[0]}", fontsize=fontsize)  # rows {start+1}-{end}
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            # plt.grid(alpha=0.3)
            
            # 第二个子图：分组序列图（使用 Axes 对象）
            # ax2 = plt.subplot(2, 1, 2)
            ax.legend(loc='upper center',
                                bbox_to_anchor=(0.5, 1.0),
                                ncol=3, 
                                fontsize=fontsize,
                                labelspacing=0.1,
                                handletextpad=0.1,
                                columnspacing=0.1)


            # 调整布局
            # plt.tight_layout()
            
            outname = f"{os.path.splitext(csv_file)[0]}_rows_{start+1}_{end}_value_and_bit_width.png"
            outpath = os.path.join(output_dir, outname)

            outpath_eps = os.path.join(output_dir,f"{os.path.splitext(csv_file)[0]}_rows_{start+1}_{end}_value_and_bit_width.eps")
            plt.savefig(outpath_eps, dpi=150, bbox_inches='tight', format='eps')
            plt.savefig(outpath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved chunk plot: {outname}")

def create_chunk_plots(csv_dir, output_dir, chunk_size=1024):
    """
    For each CSV in csv_dir (matching *_cost.csv), split rows into chunks of `chunk_size`
    and draw a line plot for each chunk (packsize vs cost). Save plots into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 定义分组序列
    sequences = [
        # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],  # 2的幂次序列
        [4, 12, 36, 108, 324, 972],           # 3 * 2的幂次
        [9, 18, 36, 72, 144, 288, 576]
        # [5, 10, 20, 40, 80, 160, 320, 640],              # 5 * 2的幂次
        # [7, 14, 28, 56, 112, 224, 448, 896],             # 7 * 2的幂次
        # [9, 18, 36, 72, 144, 288, 576],                  # 9 * 2的幂次
        # [11, 22, 44, 88, 176, 352, 704],                 # 11 * 2的幂次
        # [13, 26, 52, 104, 208, 416, 832],                # 13 * 2的幂次
        # [15, 30, 60, 120, 240, 480, 960],                 # 15 * 2的幂次
        # [17, 34, 68, 136, 272, 544],                     # 17 * 2的幂次
        # [19, 38, 76, 152, 304, 608],
        # [21, 42, 84, 168, 336, 672],
        # [23, 46, 92, 184, 368, 736],
        # [25, 50, 100, 200, 400, 800],
        # [27, 54, 108, 216, 432, 864],
        # [29, 58, 116, 232, 464, 928],
        # [31, 62, 124, 248, 496, 992],
        # [33, 66, 132, 264, 528], 
        # [35, 70, 140, 280, 560], 
        # [37, 74, 148, 296, 592], 
        # [39, 78, 156, 312, 624], 
        # [41, 82, 164, 328, 656], 
        # [43, 86, 172, 344, 688], 
        # [45, 90, 180, 360, 720], 
        # [47, 94, 188, 376, 752], 
        # [49, 98, 196, 392, 784], 
        # [51, 102, 204, 408, 816], 
        # [53, 106, 212, 424, 848], 
        # [55, 110, 220, 440, 880], 
        # [57, 114, 228, 456, 912], 
        # [59, 118, 236, 472, 944], 
        # # [61, 122, 244, 488, 976]
        # [15,30,60,120,240,480,960],
        # [20,60,180,540],
        # [12,60,300],
    ]
    
    # 序列名称
    seq_names = [
        # r"2*$2^k$ (1,2,4,...,1024)",
        r"4*$3^\beta$ (4,12,...,972)",
        r"9*$2^\beta$ (9,18,...,576)",
        
        # r"5*$2^k$ (5,10,...,640)",
        # r"7*$2^k$ (7,14,...,896)",
        # r"9*$2^k$ (9,18,...,576)",
        # r"11*$2^k$ (11,22,...,704)",
        # r"13*$2^k$ (13,26,...,832)",
        # r"15*$2^k$ (15,30,...,960)",
        # r"17*$2^k$ (17,34,...,544)",
        # r"19*$2^k$ (19,38,...,608)",
        # r"21*$2^k$ (21,42,...,672)",
        # r"23*$2^k$ (23,46,...,736)",
        # r"25*$2^k$ (25,50,...,800)",
        # r"27*$2^k$ (27,54,...,864)",
        # r"29*$2^k$ (29,58,...,928)",
        # r"31*$2^k$ (31,62,...,992)",
        # r"33*$2^k$ (33,66,...,528)",
        # r"35*$2^k$ (35,70,...,560)",
        # r"37*$2^k$ (37,74,...,592)",
        # r"39*$2^k$ (39,78,...,624)",
        # r"41*$2^k$ (41,82,...,656)",
        # r"43*$2^k$ (43,86,...,688)",
        # r"45*$2^k$ (45,90,...,720)",
        # r"47*$2^k$ (47,94,...,752)",
        # r"49*$2^k$ (49,98,...,784)",
        # r"51*$2^k$ (51,102,...,816)",
        # r"53*$2^k$ (53,106,...,848)",
        # r"55*$2^k$ (55,110,...,880)",
        # r"57*$2^k$ (57,114,...,912)",
        # r"59*$2^k$ (59,118,...,944)",
        # r"61*$2^k$ (61,122,244,...,976)"
        # r"15*$2^\beta$ (15,30,60,...,960)",
        # r"20*$3^\beta$ (20,60,180,540)",
        # r"12*$5^\beta$ (12,60,300)",
    ]

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_cost.csv')]
    fontsize = 14
    for csv_file in csv_files:
        # print(f"Chunk plotting: {csv_file}")
        if not csv_file.startswith('PM10-dust'):
            continue
        path = os.path.join(csv_dir, csv_file)
        print(f"Chunk plotting: {csv_file}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  Failed to read {csv_file}: {e}")
            continue

        if 'pack size' not in df.columns or 'cost' not in df.columns:
            print(f"  Skipping {csv_file}: missing 'packsize' or 'cost' columns")
            continue

        n = len(df)
        # num_chunks = (n + chunk_size - 1) // chunk_size
        num_chunks = 1

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(n, (i + 1) * chunk_size)
            sub = df.iloc[start:end]
            if sub.empty:
                continue

            # Prepare x/y (sort by packsize for clearer line)
            sub_sorted = sub.sort_values(by='pack size')
            x = sub_sorted['pack size'].values
            y = sub_sorted['cost'].values

            # 原始图
            plt.figure(figsize=(6, 4))
            
            # # 第一个子图：原始数据
            # plt.subplot(2, 1, 1)
            # plt.plot(x, y, linestyle='-', marker='.', markersize=3)
            # plt.xticks(fontsize=fontsize)
            # plt.yticks(fontsize=fontsize)
            # plt.xlabel(r'pack size $p$', fontsize=fontsize)
            # plt.ylabel('Cost (bits)', fontsize=fontsize)
            # plt.title(f"(a) Cost varying pack size in {csv_file.split('-')[0]}", x=0.4, fontsize=fontsize)  # rows {start+1}-{end}
            # # plt.grid(alpha=0.3)
            
            # 第二个子图：分组序列图（使用 Axes 对象）
            ax2 = plt.subplot(1, 1, 1)

            colors = [
                "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500", "#800080",
                "#008000", "#FFC0CB", "#A52A2A", "#808080", "#000000", "#FFD700", "#ADFF2F", "#FF4500",
                "#DA70D6", "#1E90FF", "#FF6347", "#7CFC00", "#8A2BE2", "#DC143C", "#00CED1", "#FF8C00",
                "#9932CC", "#8B0000", "#2E8B57", "#DAA520", "#4B0082", "#808000"
            ]

            # 收集所有序列中的packsize，用于识别剩余值
            all_seq_packsizes = set()
            for seq in sequences:
                all_seq_packsizes.update(seq)

            # 获取当前chunk中所有的packsize
            current_packsizes = set(sub['pack size'].unique())

            # 绘制每个序列
            for idx, (seq, seq_name, color) in enumerate(zip(sequences, seq_names, colors)):
                # 找出当前序列中在当前chunk中存在的值
                seq_in_data = [ps for ps in seq if ps in current_packsizes]
                if seq_in_data:
                    # 获取这些packsize对应的cost（平均值）
                    seq_data = []
                    for ps in seq_in_data:
                        avg_cost = sub[sub['pack size'] == ps]['cost'].mean()
                        seq_data.append((ps, avg_cost))

                    # 按packsize排序
                    seq_data.sort(key=lambda x: x[0])
                    seq_x = [item[0] for item in seq_data]
                    seq_y = [item[1] for item in seq_data]

                    ax2.plot(seq_x, seq_y, linestyle='-', marker='o', markersize=5,
                             color=color, linewidth=2, label=seq_name)

            ax2.set_xlabel(r'Pack size $s$', fontsize=fontsize)
            # xtick和ytick字体大小设置为fontsize
            ax2.tick_params(axis='both', labelsize=fontsize)
            # ax2.set_xticksl(fontsize=fontsize)
            ax2.set_ylabel('Cost (bits)', fontsize=fontsize)
            ax2.vlines(x=36, ymin=0, ymax=max(y)*1.1, colors='gray', linestyles='dashed', linewidth=1)
            # ax2.set_title(f"Cost of different arrays varying s in {csv_file.split('-')[0]}", x=0.4, fontsize=fontsize)

            # 放置图例在第二个子图的正下方（xlabel 下方）
            # 使用 bbox_to_anchor 的负 y 值将图例放在轴外部，下方
            legend = ax2.legend(loc='upper center',
                                bbox_to_anchor=(0.4, 1.25),
                                ncol=2, 
                                fontsize=fontsize,
                                labelspacing=0.1,
                                handletextpad=0.1,
                                columnspacing=0.1)

            # 调整图像底部留白，防止图例被裁剪
            # fig = plt.gcf()
            # fig.subplots_adjust(bottom=0.10)
            
            # 调整布局
            # plt.tight_layout()
            
            outname = f"{os.path.splitext(csv_file)[0]}_rows_{start+1}_{end}_grouped_multiple.png"
            outpath = os.path.join(output_dir, outname)
            plt.savefig(outpath, dpi=300, bbox_inches='tight')
            outname = f"{os.path.splitext(csv_file)[0]}_rows_{start+1}_{end}_grouped_multiple.eps"
            outpath = os.path.join(output_dir, outname)
            plt.savefig(outpath, dpi=300,format='eps', bbox_inches='tight')
            plt.close()
            print(f"  Saved chunk plot: {outname}")


def create_chunk_vary_3_plots(csv_dir, output_dir, chunk_size=1024):
    """
    For each CSV in csv_dir (matching *_cost.csv), split rows into chunks of `chunk_size`
    and draw a line plot for each chunk (packsize vs cost). Save plots into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 定义分组序列
    sequences = [
        # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],  # 2的幂次序列
        [3, 6, 12, 24, 48, 96, 192, 384, 768],           # 3 * 2的幂次
        # [5, 10, 20, 40, 80, 160, 320, 640],              # 5 * 2的幂次
        # [7, 14, 28, 56, 112, 224, 448, 896],             # 7 * 2的幂次
        # [9, 18, 36, 72, 144, 288, 576],                  # 9 * 2的幂次
        # [11, 22, 44, 88, 176, 352, 704],                 # 11 * 2的幂次
        # [13, 26, 52, 104, 208, 416, 832],                # 13 * 2的幂次
        # [15, 30, 60, 120, 240, 480, 960],                 # 15 * 2的幂次
        # [17, 34, 68, 136, 272, 544],                     # 17 * 2的幂次
        # [19, 38, 76, 152, 304, 608],
        # [21, 42, 84, 168, 336, 672],
        # [23, 46, 92, 184, 368, 736],
        # [25, 50, 100, 200, 400, 800],
        # [27, 54, 108, 216, 432, 864],
        # [29, 58, 116, 232, 464, 928],
        # [31, 62, 124, 248, 496, 992],
        # [33, 66, 132, 264, 528], 
        # [35, 70, 140, 280, 560], 
        # [37, 74, 148, 296, 592], 
        # [39, 78, 156, 312, 624], 
        # [41, 82, 164, 328, 656], 
        # [43, 86, 172, 344, 688], 
        # [45, 90, 180, 360, 720], 
        # [47, 94, 188, 376, 752], 
        # [49, 98, 196, 392, 784], 
        # [51, 102, 204, 408, 816], 
        # [53, 106, 212, 424, 848], 
        # [55, 110, 220, 440, 880], 
        # [57, 114, 228, 456, 912], 
        # [59, 118, 236, 472, 944], 
        # [61, 122, 244, 488, 976]
    ]
    
    # 序列名称
    seq_names = [
        # r"2*$2^k$ (1,2,4,...,1024)",
        r"3*$2^\beta$ (3,6,...,768)",
        # r"5*$2^k$ (5,10,...,640)",
        # r"7*$2^k$ (7,14,...,896)",
        # r"9*$2^k$ (9,18,...,576)",
        # r"11*$2^k$ (11,22,...,704)",
        # r"13*$2^k$ (13,26,...,832)",
        # r"15*$2^k$ (15,30,...,960)",
        # r"17*$2^k$ (17,34,...,544)",
        # r"19*$2^k$ (19,38,...,608)",
        # r"21*$2^k$ (21,42,...,672)",
        # r"23*$2^k$ (23,46,...,736)",
        # r"25*$2^k$ (25,50,...,800)",
        # r"27*$2^k$ (27,54,...,864)",
        # r"29*$2^k$ (29,58,...,928)",
        # r"31*$2^k$ (31,62,...,992)",
        # r"33*$2^k$ (33,66,...,528)",
        # r"35*$2^k$ (35,70,...,560)",
        # r"37*$2^k$ (37,74,...,592)",
        # r"39*$2^k$ (39,78,...,624)",
        # r"41*$2^k$ (41,82,...,656)",
        # r"43*$2^k$ (43,86,...,688)",
        # r"45*$2^k$ (45,90,...,720)",
        # r"47*$2^k$ (47,94,...,752)",
        # r"49*$2^k$ (49,98,...,784)",
        # r"51*$2^k$ (51,102,...,816)",
        # r"53*$2^k$ (53,106,...,848)",
        # r"55*$2^k$ (55,110,...,880)",
        # r"57*$2^k$ (57,114,...,912)",
        # r"59*$2^k$ (59,118,...,944)",
        # r"61*$2^k$ (61,122,244,...,976)"
    ]

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_cost.csv')]

    for csv_file in csv_files:
        # print(f"Chunk plotting: {csv_file}")
        if not csv_file.startswith('PM10-dust'):
            continue
        path = os.path.join(csv_dir, csv_file)
        print(f"Chunk plotting: {csv_file}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  Failed to read {csv_file}: {e}")
            continue

        if 'pack size' not in df.columns or 'cost' not in df.columns:
            print(f"  Skipping {csv_file}: missing 'packsize' or 'cost' columns")
            continue

        n = len(df)
        # num_chunks = (n + chunk_size - 1) // chunk_size
        num_chunks = 1

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(n, (i + 1) * chunk_size)
            sub = df.iloc[start:end]
            if sub.empty:
                continue

            # Prepare x/y (sort by packsize for clearer line)
            sub_sorted = sub.sort_values(by='pack size')
            x = sub_sorted['pack size'].values
            y_1 = sub_sorted['bitwidth_cost'].values
            y_2 = sub_sorted['value_cost'].values
            y = sub_sorted['cost'].values

            # 原始图
            plt.figure(figsize=(6,4))
            
            # # 第一个子图：原始数据
            # plt.subplot(2, 1, 1)
            # plt.plot(x, y, linestyle='-', marker='.', markersize=3)
            # plt.xlabel('Packsize')
            # plt.ylabel('Cost (bits)')
            # plt.title(f"{csv_file}")  # rows {start+1}-{end}
            # # plt.grid(alpha=0.3)
            
            # 第二个子图：分组序列图（使用 Axes 对象）
            ax2 = plt.subplot(1, 1, 1)

            colors = [
                "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500", "#800080",
                "#008000", "#FFC0CB", "#A52A2A", "#808080", "#000000", "#FFD700", "#ADFF2F", "#FF4500",
                "#DA70D6", "#1E90FF", "#FF6347", "#7CFC00", "#8A2BE2", "#DC143C", "#00CED1", "#FF8C00",
                "#9932CC", "#8B0000", "#2E8B57", "#DAA520", "#4B0082", "#808000"
            ]

            # 收集所有序列中的packsize，用于识别剩余值
            all_seq_packsizes = set()
            for seq in sequences:
                all_seq_packsizes.update(seq)

            # 获取当前chunk中所有的packsize
            current_packsizes = set(sub['pack size'].unique())
            fontsize = 14

            # 绘制每个序列
            for idx, (seq, seq_name, color) in enumerate(zip(sequences, seq_names, colors)):
                # 找出当前序列中在当前chunk中存在的值
                seq_in_data = [ps for ps in seq if ps in current_packsizes]
                if seq_in_data:
                    # 获取这些packsize对应的cost（平均值）
                    seq_data = []
                    for ps in seq_in_data:
                        avg_cost = sub[sub['pack size'] == ps]['cost'].mean()
                        avg_bitwidth_cost = sub[sub['pack size'] == ps]['bitwidth_cost'].mean()
                        avg_value_cost = sub[sub['pack size'] == ps]['value_cost'].mean()
                        seq_data.append((ps, avg_cost, avg_bitwidth_cost, avg_value_cost))

                    # 按packsize排序
                    seq_data.sort(key=lambda x: x[0])
                    seq_x = [item[0] for item in seq_data]
                    seq_y = [item[1] for item in seq_data]
                    seq_y1 = [item[2] for item in seq_data]
                    seq_y2 = [item[3] for item in seq_data]

                    seq_name = seq_name.split('(')[0].strip()
                    ax2.plot(seq_x, seq_y, linestyle='-', marker='o', markersize=5,
                             color=colors[0], linewidth=2, label=f"Total stoarge cost of {seq_name} ")
                    
                    ax2.plot(seq_x, seq_y1, linestyle='--', marker='x', markersize=5,
                                color=colors[1], linewidth=1, label=f"Bit width cost of {seq_name} ")
                    ax2.plot(seq_x, seq_y2, linestyle='--', marker='s', markersize=5,
                                color=colors[2], linewidth=1, label=f"Value cost of {seq_name}")
            ax2.set_xlabel('Pack size s',fontsize=fontsize)
            ax2.set_ylabel('Cost (bits)',fontsize=fontsize)
            ax2.tick_params(axis='both', labelsize=fontsize)
            str_title = r"Cost of $s$ with $\alpha$ = 3"
            # ax2.set_title(f"{str_title}", x=0.4,fontsize=fontsize) #{csv_file.split('-')[0]}

            # 放置图例在第二个子图的正下方（xlabel 下方）
            # 使用 bbox_to_anchor 的负 y 值将图例放在轴外部，下方
            legend = ax2.legend(loc='upper center',
                                bbox_to_anchor=(0.4, 1.25),
                                ncol=2, 
                                fontsize=fontsize,
                                labelspacing=0.1,
                                handletextpad=0.1,
                                columnspacing=0.1)

            # 调整图像底部留白，防止图例被裁剪
            # fig = plt.gcf()
            # fig.subplots_adjust(bottom=0.20)
            
            # 调整布局
            # plt.tight_layout()
            
            outname = f"{os.path.splitext(csv_file)[0]}_rows_{start+1}_{end}_grouped_by_3.png"
            outpath = os.path.join(output_dir, outname)
            plt.savefig(outpath, dpi=300, bbox_inches='tight')
            outname = f"{os.path.splitext(csv_file)[0]}_rows_{start+1}_{end}_grouped_by_3.eps"
            outpath = os.path.join(output_dir, outname)
            plt.savefig(outpath, dpi=300, bbox_inches='tight', format='eps')
            plt.close()
            print(f"  Saved chunk plot: {outname}")


def create_each_chunk_plots(csv_dir, output_dir):
    """
    为每个CSV文件创建简单的packsize-cost折线图
    
    Args:
        csv_dir: CSV文件目录
        output_dir: 输出图片目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_cost.csv')]
    
    # create one plot per chunk of rows for each CSV
    for csv_file in csv_files:
        if not csv_file.startswith('Bitcoin'):
            continue
        print(f"处理: {csv_file}")
        path = os.path.join(csv_dir, csv_file)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  Failed to read {csv_file}: {e}")
            continue

        if 'pack size' not in df.columns or 'cost' not in df.columns:
            print(f"  Skipping {csv_file}: missing 'pack size' or 'cost'")
            continue

        n = len(df)
        chunk_size = 1024
        num_chunks = (n + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(n, (i + 1) * chunk_size)
            sub = df.iloc[start:end]
            if sub.empty:
                continue

            # group by packsize within the chunk
            grouped = sub.groupby('pack size')['cost'].mean().reset_index()

            # create plot for this chunk
            fontsize = 16
            plt.figure(figsize=(10, 6))
            plt.plot(grouped['pack size'], grouped['cost'], linewidth=2, marker='o', markersize=4)

            # mark minimum
            min_idx = grouped['cost'].idxmin()
            min_packsize = grouped.loc[min_idx, 'pack size']
            min_cost = grouped.loc[min_idx, 'cost']
            plt.scatter([min_packsize], [min_cost], color='red', s=100, zorder=5,
                       label=f'Optimal: packsize={min_packsize}, cost={min_cost:.0f}')
            plt.axvline(x=min_packsize, color='red', linestyle='--', alpha=0.5)
            plt.axvline(x=512, color='red', linestyle='--', alpha=0.5)

            plt.xlabel('Pack size', fontsize=fontsize)
            plt.ylabel('Average Cost (bits)', fontsize=fontsize)
            plt.title(f"{csv_file.replace('_cost.csv','')} rows {start+1}-{end}", fontsize=fontsize)
            plt.legend(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            outname = f"{os.path.splitext(csv_file)[0]}_rows_{start+1}_{end}_each_chunk.png"
            outpath = os.path.join(output_dir, outname)
            plt.savefig(outpath, dpi=150)
            plt.close()
            print(f"  Saved chunk plot: {outname}")

# 使用示例
if __name__ == "__main__":
    csv_dir = "/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/packsize_cost_analysis"
    output_dir = "/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/simple_plots"
    
    # create_simple_plots(csv_dir, output_dir)

    # chunk_output = "/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/simple_plots"
    # create_chunk_vary_3_plots(csv_dir, chunk_output, chunk_size=1024)
    # chunk_output = "/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/simple_plots"
    # fig_of_cost_values_bitwidth_in_chunk(csv_dir, chunk_output, chunk_size=1024)
    chunk_output = "/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/simple_plots"
    create_chunk_plots(csv_dir, chunk_output, chunk_size=1024)
    # chunk_output = "/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/simple_plots/each_chunk_plots"
    # create_each_chunk_plots(csv_dir, chunk_output)