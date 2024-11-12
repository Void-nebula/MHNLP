# import json
# from collections import defaultdict

# # Load the JSON data
# file_path = '/home/yran1/NLP/proposal/train_data/processed_train_data.json'
# with open(file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # Define a function to remove or replace special characters
# def clean_text(text):
#     # Replace unwanted characters
#     return text.replace("\u00e2\u0080\u009c", "")

# # Group outputs by cleaned input text
# input_to_outputs = defaultdict(set)
# for entry in data["train"]:
#     # Clean the input text
#     input_text = clean_text(entry["input"])
#     output_text = entry["output"]
#     input_to_outputs[input_text].add(output_text)

# # Find inputs with multiple distinct outputs
# duplicate_inputs = {input_text: list(outputs) for input_text, outputs in input_to_outputs.items() if len(outputs) > 1}

# # Display results
# if duplicate_inputs:
#     print("Inputs with different outputs:")
#     unique_output_count = 0
#     for input_text, outputs in duplicate_inputs.items():
#         unique_output_count += len(outputs)
#         print(f"Input: {input_text}")
#         print(f"Distinct Outputs ({len(outputs)}): {outputs}\n")

#     # Summary
#     print(f"\nSummary:")
#     print(f"Number of duplicated inputs with multiple distinct outputs: {len(duplicate_inputs)}")
#     print(f"Total number of unique outputs across these duplicated inputs: {unique_output_count}")

# else:
#     print("No inputs with different outputs found.")

# # Save the cleaned data back to a JSON file (optional)
# output_file_path = '/home/yran1/NLP/proposal/train_data/cleaned_train_data.json'
# for entry in data["train"]:
#     entry["input"] = clean_text(entry["input"])

# with open(output_file_path, 'w', encoding='utf-8') as outfile:
#     json.dump(data, outfile, ensure_ascii=False, indent=4)

# print(f"Cleaned data saved to {output_file_path}")
import json
from collections import defaultdict

# 加载 JSON 数据
file_path = '/home/yran1/NLP/proposal/test_data/processed_test_data.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 定义清理函数，移除或替换特殊字符
def clean_text(text):
    # 定义字符替换字典
    replacements = {
        "\u00e2\u0080\u009c": "",  # 替换特定字符
        "\u00e2\u0080\u009d": "",  # 另一种特殊字符替换
        "\n": " ",                # 替换换行符
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip()

# 根据清理后的 input 文本统计频率
input_counts = defaultdict(int)
for entry in data["train"]:
    input_text = clean_text(entry["input"])
    input_counts[input_text] += 1

# 找到具有多个相同 input 的条目并移除
cleaned_data = {
    "train": [entry for entry in data["train"] if input_counts[clean_text(entry["input"])] == 1]
}

# 显示结果
initial_count = len(data["train"])
final_count = len(cleaned_data["train"])
print(f"初始数据条目数: {initial_count}")
print(f"清理后数据条目数: {final_count}")
print(f"删除的重复 input 条目数: {initial_count - final_count}")

# 保存清理后的数据
output_file_path = '/home/yran1/NLP/proposal/train_data/cleaned_test_data.json'
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(cleaned_data, outfile, ensure_ascii=False, indent=4)

print(f"清理后的数据已保存至 {output_file_path}")
