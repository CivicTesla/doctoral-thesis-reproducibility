import numpy as np
import pandas as pd

# 读取.npy文件并将数据存储到字典中
data_dict = {}

for i in range(1, 31):
    for j in range(10, 110, 10):
        file_path = f"{i:02d}-Agents{j}-LoopCovMoves.npy"
#         file_path =f"DT-Agents{j:02d}-LoopiNumber.npy"

        try:
            data = np.load(file_path)
            if j not in data_dict:
                data_dict[j] = []
            data_dict[j].append(data)
        except FileNotFoundError:
            print(f"File {file_path} not found.")

        
# 合并数据列表中的所有数组
for agents, data_list in data_dict.items():
    combined_data = np.concatenate(data_list, axis=0)
    data_dict[agents] = combined_data




# 将 data_dict 转换为 DataFrame
df = pd.DataFrame()

for agents, data in data_dict.items():
    # 创建 DataFrame 的一行，将 data_dict 的键值作为最后一列
    row = pd.Series(data.tolist() + [agents], name=f"Row_{agents}")
    # 将行添加到 DataFrame
    df = df.append(row)

# 重命名列并添加 label 列
df.rename(columns=lambda x: f"Column_{x}" if x != len(df.columns) - 1 else "label", inplace=True)

# 将 label 列设置为最后一列
cols = df.columns.tolist()
cols = cols[:-1] + [cols[-1]]
df = df[cols]
# 保存为 CSV 文件
df.to_csv('CombinedFile/P2PLoopCovMoves.csv', index=False)