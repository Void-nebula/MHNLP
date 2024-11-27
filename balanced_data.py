import pandas as pd
from sklearn.utils import resample

# 读取 CSV 文件
df = pd.read_csv('2016-2017.csv')

# 设置目标总样本数
target_total_size = 10000

# 打印每个标签的分布情况
print("Disorder分布: ", df['disorder'].value_counts())
print("Depression_state分布: ", df['depression_state'].value_counts())
print("Anxiety_state分布: ", df['anxiety_state'].value_counts())

# 计算每个标签组合的出现频次
# 这里通过三列标签组合成一个新的列，方便观察每种组合的分布
df['label_combo'] = df.apply(lambda row: f"{row['disorder']}_{row['depression_state']}_{row['anxiety_state']}", axis=1)

# 统计组合标签的出现次数
combo_counts = df['label_combo'].value_counts()
print("标签组合的分布: \n", combo_counts)

# 计算每个组合需要的样本数
# 使用欠采样，确保每个组合标签样本数不会超过总数
downsampled_df = pd.DataFrame()
for label_combo in combo_counts.index:
    subset = df[df['label_combo'] == label_combo]
    # 对每个组合进行欠采样
    sampled_subset = resample(subset, replace=False, n_samples=min(len(subset), target_total_size // len(combo_counts)), random_state=42)
    downsampled_df = pd.concat([downsampled_df, sampled_subset])

# 如果下采样的结果仍然超过了 1000 行，则再一次随机抽样
if len(downsampled_df) > target_total_size:
    downsampled_df = resample(downsampled_df, replace=False, n_samples=target_total_size, random_state=42)

# 删除辅助列 'label_combo'
downsampled_df = downsampled_df.drop(columns=['label_combo'])

# 检查平衡后的标签分布
print("平衡后的标签分布:")
print(downsampled_df['disorder'].value_counts())
print(downsampled_df['depression_state'].value_counts())
print(downsampled_df['anxiety_state'].value_counts())

# 保存平衡后的数据集
downsampled_df.to_csv('balanced_train_data.csv', index=False)
print(f"平衡后的数据已保存，总行数为: {len(downsampled_df)}")
