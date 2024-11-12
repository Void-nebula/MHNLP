import pandas as pd
import glob

# 获取所有 CSV 文件的路径
file_paths = glob.glob('/home/yran1/NLP/proposal/all_the_data/*.csv')  # 请替换为您的文件路径

# 用于存储所有数据和多余的列
dataframes = []
extra_columns = set()

# 读取并合并所有文件
for file_path in file_paths:
    df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)  # 添加 encoding 参数
    dataframes.append(df)
    
    # 检查是否有额外的列
    if len(dataframes) > 1:
        current_columns = set(df.columns)
        reference_columns = set(dataframes[0].columns)
        extra_columns.update(current_columns.symmetric_difference(reference_columns))

# 合并数据
merged_data = pd.concat(dataframes, ignore_index=True)

# 如果有多余的列，将它们单独提取
if extra_columns:
    extra_data = merged_data[list(extra_columns)]
    merged_data = merged_data.drop(columns=extra_columns)  # 去除多余列后的主要数据

    # 将合并后的数据和多余的列分别保存到本地 CSV 文件
    merged_data.to_csv('merged_data.csv', index=False)
    extra_data.to_csv('extra_columns_data.csv', index=False)

    print("合并的数据已保存到 merged_data.csv")
    print("多余的列已保存到 extra_columns_data.csv")
else:
    # 如果没有多余列，直接保存合并的数据
    merged_data.to_csv('merged_data.csv', index=False)
    print("合并的数据已保存到 merged_data.csv")
