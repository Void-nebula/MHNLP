import pandas as pd
import json

# 读取 CSV 文件
df = pd.read_csv("/home/yran1/NLP/proposal/processed_dataset_train.csv")

# # 定义映射字典
# depression_state_map = {
#     0: "No Depression", 1: "Major depressive disorder", 2: "Dysthymia or persistent depressive disorder",
#     3: "Premenstrual dysphoric disorder", 4: "Other Depression State", 5: "Other UnKnown Depression State"
# }
# anxiety_state_map = {
#     0: "No Anxiety", 1: "Generalized anxiety disorder", 2: "Panic disorder",
#     3: "Agoraphobia", 4: "Specific phobia (e.g., claustrophobia, arachnophobia, etc.)",
#     5: "Social anxiety disorder (or social phobia)", 6: "Other Anxiety State", 7: "Other UnKnown Anxiety State"
# }

# both : 0, depression: 1, anxiety: 2, none: 3
# 定义映射字典
disorder_state_map = {
    0: "both illness", 1: "depression", 2: "anxiety", 3: "none"
}

# 应用映射并合并列
# df["output"] = (
#     df["depression_state"].map(depression_state_map) + ", " +
#     df["anxiety_state"].map(anxiety_state_map)
# )

df["output"] = (
    df["disorder"].map(disorder_state_map)
)


# 删除多余的列并重命名
df = df.drop(columns=["idx", "disorder"])
df = df.rename(columns={"text": "input"})

# 转换为JSON格式
output_json = {
    "train": df.to_dict(orient="records")
}

# 保存为JSON文件
output_path = "/home/yran1/NLP/proposal/train_data/train_total.json"
with open(output_path, "w") as f:
    json.dump(output_json, f, indent=4)

print(f"JSON 文件已保存至 {output_path}")
