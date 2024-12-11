import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaConfig,
    RobertaPreTrainedModel,
    RobertaModel,
    AdamW,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


# 定义数据集类
class MTGDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        text = data["text"]
        labels = torch.tensor(data["labels"], dtype=torch.float)

        # Tokenize 文本
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


class MultiClassifierRobertaModel(nn.Module):
    def __init__(self, config, num_labels):
        super(MultiClassifierRobertaModel, self).__init__()
        self.roberta = RobertaModel(config)
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(num_labels)])
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:, 0, :]  # 获取 [CLS] token 的表示
        logits = [classifier(sequence_output) for classifier in self.classifiers]

        if labels is not None:
            losses = []
            for i, logit in enumerate(logits):
                loss_fn = nn.CrossEntropyLoss()
                losses.append(loss_fn(logit, labels[:, i]))
            total_loss = torch.stack(losses).mean()
            return {"loss": total_loss, "logits": logits}
        else:
            return {"logits": logits}


# 训练模型
def train_model():
    # 加载数据集
    dataset = load_dataset("joshuasundance/mtg-coloridentity-multilabel-classification")

    # 初始化 Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # 创建数据集和数据加载器
    train_dataset = MTGDataset(dataset["train"], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 加载模型配置和模型
    config = RobertaConfig.from_pretrained("roberta-base")

    # 获取标签数量
    num_labels = len(dataset["train"][0]["labels"])  # 修改点，直接从第一条样本推断标签数
    model = MultiClassifierRobertaModel(config, num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 训练循环
    num_epochs = 2
    save_path = "./saved_model_mtg"  # 模型保存路径

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs["loss"]
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"模型已保存到路径: {save_path}")


def evaluate_model():
    dataset = load_dataset("joshuasundance/mtg-coloridentity-multilabel-classification")

    # initialize Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("./saved_model_mtg")

    # get label tokenizer
    num_labels = len(dataset["train"][0]["labels"])  # 从数据集推断标签数量

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RobertaConfig.from_pretrained("./saved_model_mtg")

    # load the pretrained model
    model = MultiLabelRobertaModel.from_pretrained("./saved_model_mtg", config=config, num_labels=num_labels)
    model.to(device)
    model.eval()

    # create validation dataset
    validation_dataset = MTGDataset(dataset["test"], tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

    # test
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            # get the model prediction
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()  # transfer to probability

            all_labels.append(labels)
            all_predictions.append(predictions)

    # merge all the batch
    all_labels = np.concatenate(all_labels, axis=0)
    print("Expected Labels: ", all_labels)
    all_predictions = np.concatenate(all_predictions, axis=0)
    print("Model Predictions: ", all_predictions)
    # calculate ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_predictions, average="macro")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")

    # calculate accuracy
    threshold = 0.5
    binary_predictions = (all_predictions >= threshold).astype(int)
    accuracy = accuracy_score(all_labels.flatten(), binary_predictions.flatten())
    print(f"Validation Accuracy: {accuracy:.4f}")




if __name__ == "__main__":
    set_seed(42)

    # print("开始训练模型...")
    # train_model()

    print("\n开始验证模型...")
    evaluate_model()
