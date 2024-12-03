import pandas as pd
from sklearn.model_selection import train_test_split
import json

df = pd.read_csv('HMS_2023_processed_binary_balanced_1000.csv')

# Split the dataset into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)
train.to_csv('HMS_2023_processed_binary_balanced_1000_train.csv', index=False)
test.to_csv('HMS_2023_processed_binary_balanced_1000_test.csv', index=False)

# Construct .jsonl file for fine-tuning
jsonl_data = []
for _, row in train.iterrows():
    jsonl_data.append({
        "messages": [
            {"role": "system", "content": "You are a professional therapist. Your patient is telling you about their basic information and some answers to a mental health servey. Based on the information, please infer whether the patient has depression and/or anxiety disorders. 0: Neither, 1: Either or both. Based on this correspondence, you only need to answer the number."},
            {"role": "user", "content": row['prompt']},
            {"role": "assistant", "content": str(row['label'])}
        ]
    })

with open('fine_tuning_dataset.jsonl', 'w') as f:
    for entry in jsonl_data:
        json.dump(entry, f)
        f.write("\n")

print("JSONL file saved: fine_tuning_dataset.jsonl")

# Construct .jsonl file for validation
jsonl_data = []
for _, row in test.iterrows():
    jsonl_data.append({
        "messages": [
            {"role": "system", "content": "You are a professional therapist. Your patient is telling you about their basic information and some answers to a mental health servey. Based on the information, please infer whether the patient has depression and/or anxiety disorders. 0: Neither, 1: Either or both. Based on this correspondence, you only need to answer the number."},
            {"role": "user", "content": row['prompt']},
            {"role": "assistant", "content": str(row['label'])}
        ]
    })

with open('validation_dataset.jsonl', 'w') as f:
    for entry in jsonl_data:
        json.dump(entry, f)
        f.write("\n")

print("JSONL file saved: validation_dataset.jsonl")