import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load_test_data(test_data_path):
    # 加载测试数据集
    dataset = load_dataset("json", data_files=f"{test_data_path}/*.json", split="train", field="train")
    return dataset

def extract_response(text):
    # 查找 "Response:" 的位置并截取其后的内容
    response_start = text.find("Response:")
    if response_start != -1:
        return text[response_start + len("Response:"):].strip()
    return text.strip()  # 如果没找到 "Response:" 则返回完整文本

def calculate_accuracy(model, tokenizer, test_dataset):
    correct_predictions = 0
    total_samples = len(test_dataset)

    for example in test_dataset:
        prompt = example["input"]
        expected_output = extract_response(example["output"])

        # 模型生成
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
        generated_output = extract_response(tokenizer.decode(outputs[0], skip_special_tokens=True).strip())
        print("Generation: ", generated_output)
        print("Expectation: ", expected_output)

        # 比较生成输出与预期输出
        if generated_output == expected_output:
            correct_predictions += 1

    # 计算准确率
    accuracy = correct_predictions / total_samples
    return accuracy

if __name__ == '__main__':
    # 加载已训练好的模型和分词器
    model_path = "/home/yran1/NLP/proposal/gemma_output/final_checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

    # 加载测试数据集
    test_data_path = "/home/yran1/NLP/proposal/test_data"
    test_dataset = load_test_data(test_data_path)

    # 计算并输出测试准确率
    accuracy = calculate_accuracy(model, tokenizer, test_dataset)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
