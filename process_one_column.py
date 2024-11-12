import pandas as pd

# 读取数据集
df = pd.read_csv("/home/yran1/NLP/proposal/HMS_2016-2023_Depression_Anxiety.csv")

# 删除 'disorder' 列
df.drop(columns=['depression_state','anxiety_state'], inplace=True)

# 将处理后的数据保存到新的CSV文件
df.to_csv("processed_dataset_train.csv", index=False)

# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Load your dataset
# data = pd.read_csv("/home/yran1/NLP/proposal/processed_dataset.csv")  # Replace 'your_dataset.csv' with your actual file path

# # Perform an 80-20 split for training and test sets
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# # Save the split datasets to CSV files
# train_data.to_csv('train_data.csv', index=False)
# test_data.to_csv('test_data.csv', index=False)

# print("Data split complete. Training and test sets have been saved as 'train_data.csv' and 'test_data.csv'.")

