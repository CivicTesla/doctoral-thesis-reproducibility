import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义异常值处理函数
def remove_outliers(data):
    Q1 = np.percentile(data, 15)
    Q3 = np.percentile(data, 95)
    IQR = Q3 - Q1
    upper_bound = Q3
    lower_bound = Q1
    return data[(data >= lower_bound) & (data <= upper_bound)]

# 加载 .npy 文件
data1 = np.load('01DTLoopCovMoves.npy')
data2 = np.load('02DTLoopCovMoves.npy')

#Loopi File
# data1 = np.load('01DTLoopi.npy')
# data2 = np.load('02DTLoopi.npy')

# 将数据转换为 DataFrame
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.read_csv('P2PLoopCovMoves.csv')
# df3 = pd.read_csv('P2PLoopi.csv')

# 添加标签列
labels1 = np.arange(10, 101, 10)# 从10到100，间隔为10的标签数组
labels2 = np.arange(10, 101, 10) 
df1['label'] = labels1[:len(df1)]   # 将标签数组与 DataFrame 的行数对应起来
df2['label'] = labels2[:len(df2)]


                                         
                                         
# 转置 DataFrame
df1_transposed = df1.T
df2_transposed = df2.T
df3_transposed = df3.T

# 重新设置列名为标签值
df1_transposed.columns = df1_transposed.iloc[-1]
df1_transposed = df1_transposed.iloc[:-1]
df2_transposed.columns = df2_transposed.iloc[-1]
df2_transposed = df2_transposed.iloc[:-1]
df3_transposed.columns = df3_transposed.iloc[-1]
df3_transposed = df3_transposed.iloc[:-1]


# # 去除异常值
df1_transposed_cleaned = df1_transposed.apply(remove_outliers, axis=0)
df2_transposed_cleaned = df2_transposed.apply(remove_outliers, axis=0)
df3_transposed_cleaned = df3_transposed.apply(remove_outliers, axis=0)


# 添加分类列
df1_transposed_cleaned['dataset'] = 'DT mode 01'
df2_transposed_cleaned['dataset'] = 'DT mode 02'
df3_transposed_cleaned['dataset'] = 'P2P'


# # 绘制小提琴图
# plt.figure(figsize=(21, 14))
# sns.violinplot(data=df1_transposed_cleaned, palette='Set1', inner="quart", linewidth=1)
# sns.violinplot(data=df2_transposed_cleaned, palette='Set2', inner="quart", linewidth=1)

color1 = (168/255,224/255,146/255)
color2 = (248/255,216/255,147/255)
color3 = (210/255,210/255,210/255)
# 绘制小提琴图
plt.figure(figsize=(21, 14))
sns.violinplot(data=df1_transposed_cleaned, color=color1, inner="quart", linewidth=1)
sns.violinplot(data=df2_transposed_cleaned, color=color2, inner="quart", linewidth=1)
sns.violinplot(data=df3_transposed_cleaned, color=color3, inner="quart", linewidth=1)

#添加网格
plt.grid(True, axis='y', which='major', linestyle='-', linewidth=0.5, alpha=0.7)
plt.yscale('log')
plt.xlabel('Number of agents', fontsize=14)
plt.ylabel('Total moves until 90% convergence', fontsize=14)
# plt.ylabel('Rounds until 90% convergence', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
# plt.title('DT & P2P')

# 手动创建图例
import matplotlib.patches as mpatches
legend_elements = [mpatches.Patch(color=color1, label='DT mode 1'),
                   mpatches.Patch(color=color2, label='DT mode 2'),
                   mpatches.Patch(color=color3, label='P2P')]
plt.legend(handles=legend_elements, loc='upper left')  # 设置图例位置为左上角
plt.tight_layout(rect=[0, 0, 1, 0.9])  # 放大图例的比例


# plt.savefig('LoopiAll.png')
plt.savefig('LoopCovMovesAll1.png')

plt.show()