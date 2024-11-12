import json
from collections import Counter

# # 读取 JSON 文件
# with open('/home/yran1/NLP/proposal/train_data/train_total.json', 'r', encoding='utf-8') as file:  # 请将路径替换为您的 JSON 文件路径
#     data = json.load(file)

# # 提取 "output" 字段
# outputs = [entry["output"] for entry in data["train"]]

# # 统计每个唯一 output 的出现次数
# output_counts = Counter(outputs)

# # 打印每个答案种类和出现的次数
# for output, count in output_counts.items():
#     print(f"{output}: {count}")

import json
from collections import defaultdict

# 读取 JSON 文件
with open('/home/yran1/NLP/proposal/train_data/train_total.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 按 output 进行分类
output_data = defaultdict(list)
for entry in data["train"]:
    output_data[entry["output"]].append(entry)

# 分成训练集和测试集
train_data = []
test_data = []

for output, entries in output_data.items():
    if len(entries) > 1000:

        train_data.extend(entries[:1000])
        test_data.extend(entries[1000:1200])  # 取第 101 条到第 120 条之间的数据（如果有）

# 保存训练集和测试集到不同的 JSON 文件
with open('/home/yran1/NLP/proposal/train_data/processed_train_data.json', 'w', encoding='utf-8') as train_file:
    json.dump({"train": train_data}, train_file, ensure_ascii=False, indent=4)

with open('/home/yran1/NLP/proposal/train_data/processed_test_data.json', 'w', encoding='utf-8') as test_file:
    json.dump({"test": test_data}, test_file, ensure_ascii=False, indent=4)

print("数据处理完成。训练集已保存到 'processed_train_data.json'，测试集已保存到 'processed_test_data.json'")
