# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# """
# 整个程序是将图片分成两个y轴,一个是显示模式45对比模式1所用的uplink数量,一个是收敛率
# dt1就是dt模式下没有针对p2p的做法
# df4和df5都是无obstacle的dt模式
# """
# # 加载数据集
# df1 = pd.read_csv('01DTdf.csv')
# df4 = np.load('01-04-LoopLoopUpLinkNumber-Combined_Final_Matrix.npy')
# df5 = np.load('01-05-LoopUpLinkNumber-Combined_Final_Matrix.npy')

# # 过滤出 'Uplink' 的数据
# uplink_data = df1[df1['Group'] == 'Uplink Number']

# # 按照 agent 分组，并计算每组的平均值
# dt1 = uplink_data.groupby('X')['Y'].mean()

# #将数据变成numpy可用的
# dt1_sorted = dt1.sort_index()
# dt1_mean =dt1_sorted.values

# #将dt4和dt5求平均
# # 按行求均值
# dt4_mean = np.mean(df4, axis=1)  # 结果形状是 (10,)
# dt5_mean = np.mean(df5, axis=1)  # 结果形状是 (10,)

# #第一个y轴的数据计算
# normalized_mode4 = dt4_mean / dt1_mean  # shape (10,)
# normalized_mode5 = dt5_mean / dt1_mean  # shape (10,)

# # 横轴：agent的组别
# x_labels = np.array([10,20,30,40,50,60,70,80,90,100])  # 共10个
# x = np.arange(len(x_labels))  # 也就是 0,1,2,3,...9

# Cov4 = np.load('01-04-LoopCovRate-Combined_Final_Matrix.npy')
# Cov5 = np.load('01-05-LoopCovRate-Combined_Final_Matrix.npy')
# # Cov4.shape

# # 均值（用于折线）
# Cov4_mean = np.mean(Cov4, axis=1)
# Cov5_mean = np.mean(Cov5, axis=1)

# # 25/75 百分位误差棒
# Cov4_25 = np.percentile(Cov4, 25, axis=1)
# Cov4_75 = np.percentile(Cov4, 75, axis=1)
# Cov5_25 = np.percentile(Cov5, 25, axis=1)
# Cov5_75 = np.percentile(Cov5, 75, axis=1)

# # 误差范围（用于 errorbar）
# Cov4_err = [Cov4_mean - Cov4_25, Cov4_75 - Cov4_mean]
# Cov5_err = [Cov5_mean - Cov5_25, Cov5_75 - Cov5_mean]

# fig, ax1 = plt.subplots(figsize=(12, 6))

# # ----------- 左侧 Y 轴（柱状图） -------------
# bar_width = 0.25

# # bar1 = ax1.bar(x - bar_width, normalized_mode3, width=bar_width, label='P2P', color='lightgreen', edgecolor='black')
# bar2 = ax1.bar(x, normalized_mode4, width=bar_width, label='Random Walk4', color='skyblue', edgecolor='black')
# bar3 = ax1.bar(x + bar_width, normalized_mode5, width=bar_width, label='Random Walk5', color='orange', edgecolor='black')
# ax1.set_ylabel('Comparison of Uplink Amount  (Normalized by dt mode 1)', fontsize=12)
# ax1.set_xlabel('Number of Agents', fontsize=12)
# ax1.set_xticks(x)
# ax1.set_xticklabels(x_labels)

# # 网格 & 边界处理
# ax1.grid(axis='y', linestyle='--', alpha=0.6)
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)

# # ----------- 右侧 Y 轴（折线图） -------------
# ax2 = ax1.twinx()

# # 画 Cov4 的 errorbar 折线
# ax2.errorbar(x, Cov4_mean, yerr=Cov4_err, fmt='-o', color='orange', label='Random Walk4', capsize=4)

# # 画 Cov5 的 errorbar 折线
# ax2.errorbar(x, Cov5_mean, yerr=Cov5_err, fmt='-o', color='purple', label='Random Walk5', capsize=4)

# # 画水平线（90%）
# ax2.axhline(0.9, color='red', linestyle='--', label='dt mode1')

# ax2.set_ylabel('Convergence Rate', fontsize=12)
# ax2.set_ylim(0, 1.05)  # 保证 0.9 在线内

# # ----------- 图例 -------------
# # 合并图例（bar + line）
# lines_labels = ax1.get_legend_handles_labels()
# lines2_labels = ax2.get_legend_handles_labels()
# all_lines = lines_labels[0] + lines2_labels[0]
# all_labels = lines_labels[1] + lines2_labels[1]
# ax2.legend(all_lines, all_labels, loc='upper right', fontsize=11)

# # plt.title('Normalized Uplink Amount and Convergence Rate', fontsize=14)
# # plt.tight_layout()
# # plt.show()
# plt.title('Normalized Uplink Amount and Convergence Rate', fontsize=14)
# plt.tight_layout()
# plt.savefig('normalized_uplink_and_convergence.png', dpi=300)  # 保存图片
# plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""
整个程序是将图片分成两个y轴,一个是显示模式45对比模式1所用的uplink数量,一个是收敛率
dt1就是dt模式下没有针对p2p的做法
df4和df5都是无obstacle的dt模式
"""

# 定义自定义颜色 (RGB转换为0-1范围)
colors = {
    'bar_green': (168/255, 224/255, 146/255),  # 柱状图绿色
    'bar_orange': (248/255, 216/255, 147/255), # 柱状图橙色
    'line_blue': (0, 0.4, 0.8),               # 折线蓝色 - 对比度强烈
    'line_red': (0.8, 0.2, 0.2),              # 折线红色 - 对比度强烈
    'reference_gray': (0.5, 0.5, 0.5)         # 参考线灰色
}

# 加载数据集
df1 = pd.read_csv('01DTdf.csv')
df4 = np.load('01-04-LoopLoopUpLinkNumber-Combined_Final_Matrix.npy')
df5 = np.load('01-05-LoopUpLinkNumber-Combined_Final_Matrix.npy')

# 过滤出 'Uplink' 的数据
uplink_data = df1[df1['Group'] == 'Uplink Number']

# 按照 agent 分组,并计算每组的平均值
dt1 = uplink_data.groupby('X')['Y'].mean()

#将数据变成numpy可用的
dt1_sorted = dt1.sort_index()
dt1_mean = dt1_sorted.values

#将dt4和dt5求平均
# 按行求均值
dt4_mean = np.mean(df4, axis=1)  # 结果形状是 (10,)
dt5_mean = np.mean(df5, axis=1)  # 结果形状是 (10,)

#第一个y轴的数据计算
normalized_mode4 = dt4_mean / dt1_mean  # shape (10,)
normalized_mode5 = dt5_mean / dt1_mean  # shape (10,)

# 横轴:agent的组别
x_labels = np.array([10,20,30,40,50,60,70,80,90,100])  # 共10个
x = np.arange(len(x_labels))  # 也就是 0,1,2,3,...9

Cov4 = np.load('01-04-LoopCovRate-Combined_Final_Matrix.npy')
Cov5 = np.load('01-05-LoopCovRate-Combined_Final_Matrix.npy')

# Cov4.shape
# 均值(用于折线)
Cov4_mean = np.mean(Cov4, axis=1)
Cov5_mean = np.mean(Cov5, axis=1)

# 25/75 百分位误差棒
Cov4_25 = np.percentile(Cov4, 25, axis=1)
Cov4_75 = np.percentile(Cov4, 75, axis=1)
Cov5_25 = np.percentile(Cov5, 25, axis=1)
Cov5_75 = np.percentile(Cov5, 75, axis=1)

# 误差范围(用于 errorbar)
Cov4_err = [Cov4_mean - Cov4_25, Cov4_75 - Cov4_mean]
Cov5_err = [Cov5_mean - Cov5_25, Cov5_75 - Cov5_mean]

fig, ax1 = plt.subplots(figsize=(12, 6))

# ----------- 左侧 Y 轴(柱状图) -------------
bar_width = 0.25

# 使用自定义颜色
bar2 = ax1.bar(x, normalized_mode4, width=bar_width, label='Random Walk mode 1',
               color=colors['bar_green'], edgecolor='black')
bar3 = ax1.bar(x + bar_width, normalized_mode5, width=bar_width, label='Random Walk mode 2',
               color=colors['bar_orange'], edgecolor='black')

ax1.set_ylabel('Uplink Consumption Ratio (DT mode 1 = 1)', fontsize=12)
ax1.set_xlabel('Number of Agents', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)

# 网格 & 边界处理
ax1.grid(axis='y', linestyle='--', alpha=0.6)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ----------- 右侧 Y 轴(折线图) -------------
ax2 = ax1.twinx()

# 画 Cov4 的 errorbar 折线 - 使用对比度强烈的蓝色
ax2.errorbar(x, Cov4_mean, yerr=Cov4_err, fmt='-o', color=colors['line_blue'],
             label='Random Walk mode 1', capsize=4, linewidth=2, markersize=6)

# 画 Cov5 的 errorbar 折线 - 使用对比度强烈的红色
# ax2.errorbar(x, Cov5_mean, yerr=Cov5_err, fmt='-o', color=colors['line_red'],
#              label='Random Walk mode 2', capsize=4, linewidth=2, markersize=6)

ax2.errorbar(x + bar_width, Cov5_mean, yerr=Cov5_err, fmt='-o', color=colors['line_red'],
             label='Random Walk mode 2', capsize=4, linewidth=2, markersize=6)

# 画水平线(90%) - 使用灰色
ax2.axhline(0.9, color=colors['reference_gray'], linestyle='--', label='DT mode 1', linewidth=2)

ax2.set_ylabel('Convergence Rate', fontsize=12)
ax2.set_ylim(0, 1.05)  # 保证 0.9 在线内

# ----------- 图例 -------------
# 合并图例(bar + line)
lines_labels = ax1.get_legend_handles_labels()
lines2_labels = ax2.get_legend_handles_labels()
all_lines = lines_labels[0] + lines2_labels[0]
all_labels = lines_labels[1] + lines2_labels[1]
ax2.legend(all_lines, all_labels, loc='upper right', fontsize=11)

plt.title('Normalized Uplink Consumption and Convergence Rate', fontsize=14)
plt.tight_layout()
plt.savefig('ConvergenceRate.png', dpi=300)  # 保存图片
plt.show()