import pandas as pd
import torch
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from torch.optim import SGD
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaConfig
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import accuracy_score, classification_report

class MentalHealthDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Extract the text and the labels from the dataset
        text = self.data.loc[index, 'text']
        symptom_label = int(self.data.loc[index, 'disorder'])  # overall disorder label (both, depression, anxiety, none)
        depression_label = int(self.data.loc[index, 'depression_state'])  # depression state (0-5)
        anxiety_label = int(self.data.loc[index, 'anxiety_state'])  # anxiety state (0-6)

        # Tokenize the text
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Return a dictionary of inputs and labels
        return {
            'input_ids': inputs['input_ids'].squeeze(),  # remove the batch dimension
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(symptom_label, dtype=torch.long),
            'depression_labels': torch.tensor(depression_label, dtype=torch.long),
            'anxiety_labels': torch.tensor(anxiety_label, dtype=torch.long)
        }

class SymptomClassifier(nn.Module):
    """Classifier head for symptom classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 全连接层，输入维度与隐藏维度相同
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 设置 dropout 值，优先使用 config.classifier_dropout，否则使用 config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出层，输出维度为症状分类的标签数
        self.out_proj = nn.Linear(config.hidden_size, 4)  # 这里假设 symptom 任务有 4 个类别

    def forward(self, features, **kwargs):
        # 提取 [CLS] token 的特征
        x = features[:, 0, :]
        # 第一次 dropout
        x = self.dropout(x)
        # 通过 dense 层
        x = self.dense(x)
        # 使用 tanh 激活
        x = torch.tanh(x)
        # 第二次 dropout
        x = self.dropout(x)
        # 输出层
        x = self.out_proj(x)
        return x

class depression_classifier(nn.Module):
    """Classifier head for symptom classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 全连接层，输入维度与隐藏维度相同
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 设置 dropout 值，优先使用 config.classifier_dropout，否则使用 config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出层，输出维度为症状分类的标签数
        self.out_proj = nn.Linear(config.hidden_size, 6)  # 这里假设 symptom 任务有 4 个类别

    def forward(self, features, **kwargs):
        # 提取 [CLS] token 的特征
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class anxiety_classifier(nn.Module):
    """Classifier head for symptom classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 8)  # 这里假设 symptom 任务有 4 个类别

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MultiTaskRobertaModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.symptom_classifier = SymptomClassifier(config)
        
        self.depression_classifier = depression_classifier(config)
        
        self.anxiety_classifier = anxiety_classifier(config)

        # Uncertainty Weight Parameters
        self.sigma_symptom = nn.Parameter(torch.ones(1))
        self.sigma_depression = nn.Parameter(torch.ones(1))
        self.sigma_anxiety = nn.Parameter(torch.ones(1))

        # Initialize the loss weights as trainable parameters
        self.symptom_loss_weight = nn.Parameter(torch.tensor(1.0))
        self.depression_loss_weight = nn.Parameter(torch.tensor(1.0))
        self.anxiety_loss_weight = nn.Parameter(torch.tensor(1.0))

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, depression_labels=None, anxiety_labels=None):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        symptom_logits = self.symptom_classifier(outputs.last_hidden_state)

        depression_logits = self.depression_classifier(outputs.last_hidden_state)

        anxiety_logits = self.anxiety_classifier(outputs.last_hidden_state)

        loss = None
        if labels is not None and depression_labels is not None and anxiety_labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            symptom_loss = loss_fct(symptom_logits, labels)
            depression_loss = loss_fct(depression_logits, depression_labels)
            anxiety_loss = loss_fct(anxiety_logits, anxiety_labels)

            symptom_loss_weight = torch.clamp(self.symptom_loss_weight, min=0.1)
            depression_loss_weight = torch.clamp(self.depression_loss_weight, min=0.1)
            anxiety_loss_weight = torch.clamp(self.anxiety_loss_weight, min=0.1)

            # Uncertainty Weighted Loss
            loss = (
            (1 / (2 * self.sigma_symptom ** 2)) * symptom_loss +
            (1 / (2 * self.sigma_depression ** 2)) * depression_loss +
            (1 / (2 * self.sigma_anxiety ** 2)) * anxiety_loss +
            torch.log(self.sigma_symptom * self.sigma_depression * self.sigma_anxiety)
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=(symptom_logits, depression_logits, anxiety_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


def evaluate_model(model, data_loader, device):
    model.eval()  # 设置模型为评估模式
    all_symptom_preds = []
    all_depression_preds = []
    all_anxiety_preds = []
    
    all_symptom_labels = []
    all_depression_labels = []
    all_anxiety_labels = []

    with torch.no_grad():  # 禁用梯度计算，加速推理过程
        for batch in data_loader:
            # 将数据加载到指定的设备上 (GPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            symptom_labels = batch['labels'].to(device)
            depression_labels = batch['depression_labels'].to(device)
            anxiety_labels = batch['anxiety_labels'].to(device)

            # 模型前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 获取预测结果
            symptom_logits = outputs.logits[0]
            depression_logits = outputs.logits[1]
            anxiety_logits = outputs.logits[2]

            # 获取每个任务的预测类别
            _, symptom_preds = torch.max(symptom_logits, dim=1)
            _, depression_preds = torch.max(depression_logits, dim=1)
            _, anxiety_preds = torch.max(anxiety_logits, dim=1)

            # 存储预测结果和真实标签
            all_symptom_preds.append(symptom_preds)
            all_depression_preds.append(depression_preds)
            all_anxiety_preds.append(anxiety_preds)
            
            all_symptom_labels.append(symptom_labels)
            all_depression_labels.append(depression_labels)
            all_anxiety_labels.append(anxiety_labels)

    # 拼接所有批次的预测和标签，保持在 GPU 上
    all_symptom_preds = torch.cat(all_symptom_preds).to(device)
    all_depression_preds = torch.cat(all_depression_preds).to(device)
    all_anxiety_preds = torch.cat(all_anxiety_preds).to(device)
    
    all_symptom_labels = torch.cat(all_symptom_labels).to(device)
    all_depression_labels = torch.cat(all_depression_labels).to(device)
    all_anxiety_labels = torch.cat(all_anxiety_labels).to(device)

    return all_symptom_preds, all_depression_preds, all_anxiety_preds, \
           all_symptom_labels, all_depression_labels, all_anxiety_labels

# 加载模型和 tokenizer
model = MultiTaskRobertaModel.from_pretrained('./saved_model_SGD')
tokenizer = RobertaTokenizer.from_pretrained('./saved_model_SGD')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 初始化模型和测试数据集的 DataLoader
test_dataset = MentalHealthDataset('test.csv', tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 评估模型
symptom_preds, depression_preds, anxiety_preds, \
symptom_labels, depression_labels, anxiety_labels = evaluate_model(model, test_loader, device)

# 打印或分析结果
print(f"Symptom Predictions: {symptom_preds}")
print(f"Depression Predictions: {depression_preds}")
print(f"Anxiety Predictions: {anxiety_preds}")


# 计算准确率
# 计算准确率（在 GPU 上进行）
symptom_accuracy = torch.mean((symptom_preds == symptom_labels).float()).item()
depression_accuracy = torch.mean((depression_preds == depression_labels).float()).item()
anxiety_accuracy = torch.mean((anxiety_preds == anxiety_labels).float()).item()

print(f"Symptom Accuracy: {symptom_accuracy}")
print(f"Depression Accuracy: {depression_accuracy}")
print(f"Anxiety Accuracy: {anxiety_accuracy}")

# # 获取详细的分类报告
# print("Symptom Classification Report:")
# print(classification_report(symptom_labels, symptom_preds))

# print("Depression Classification Report:")
# print(classification_report(depression_labels, depression_preds))

# print("Anxiety Classification Report:")
# print(classification_report(anxiety_labels, anxiety_preds))