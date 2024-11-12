import pandas as pd

# 读取 CSV 文件
file_path = "total_train_data.csv"  # 请替换为您的文件路径
df = pd.read_csv(file_path)

# 定义每个组合的最大样本数
sample_size = 300

# 按组合分组并取样
balanced_df = df.groupby(['disorder', 'depression_state', 'anxiety_state'], group_keys=False).apply(lambda x: x.sample(min(len(x), sample_size)))

# 输出平衡后的数据
print(balanced_df)

# 保存为新的 CSV 文件（可选）
balanced_df.to_csv("balanced_total_train_data.csv", index=False)
