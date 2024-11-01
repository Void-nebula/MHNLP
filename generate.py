import torch
from transformers import RobertaConfig, RobertaTokenizer
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn

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
        self.out_proj = nn.Linear(config.hidden_size, 7)  # 这里假设 symptom 任务有 4 个类别

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

            loss = (
                symptom_loss_weight * symptom_loss +
                depression_loss_weight * depression_loss +
                anxiety_loss_weight * anxiety_loss
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=(symptom_logits, depression_logits, anxiety_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

# 加载模型和 tokenizer
model = MultiTaskRobertaModel.from_pretrained('./saved_model_SGD')
tokenizer = RobertaTokenizer.from_pretrained('./saved_model_SGD')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 将模型设置为评估模式
model.eval()

# 准备一些测试文本
test_text="The Conversation between doctor and participant, discussing barriers to mental health services: In the past 12 months, which of the following factors have caused you to receive fewer services (counseling, therapy, or medications) for your mental or emotional health than you would have otherwise received? (Select all that apply): Not enough time, Prefer to deal with issues on my own or with support from family/friends. In the past 12 months, which of the following explain why you have not received medication or therapy for your mental or emotional health? (Select all that apply): no idea. Instructions for this item: âThis question asks about ways you may have hurt yourself on purpose, without intending to kill yourself.â In the past year, have you ever done any of the following intentionally? (Select all that apply): Cut myself, Burned myself, Pulled my hair, Carved words or symbols into skin, Rubbed sharp objects into skin, I'm not hurt myself.. If you were experiencing serious emotional distress, whom would you talk to about this? (Select all that apply): Professional clinician (e.g., psychologist, counselor, or psychiatrist), Friend (who is not a roommate), Family member. In the past 12 months, have you received support for your mental or emotional health from any of the following sources? (Select all that apply): Friend (who is not a roommate), Family member. Based on the respondent's answers, is the participant depressed or anxious?"

num_tokens = len(tokenizer.tokenize(test_text))
print(f'Total tokens: {num_tokens}')

# 对文本进行 tokenization
inputs = tokenizer(
    test_text,
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length=512
)

# 将输入传入模型（假设模型在 GPU 上）
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

# 获取模型输出
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# 打印输出 logits
symptom_logits = outputs.logits[0]  # Symptom classification logits
depression_logits = outputs.logits[1]  # Depression state logits
anxiety_logits = outputs.logits[2]  # Anxiety state logits

print("Symptom Logits:", symptom_logits)
print("Depression State Logits:", depression_logits)
print("Anxiety State Logits:", anxiety_logits)

# 可以根据需要对 logits 进行进一步处理，例如获取预测结果
_, symptom_prediction = torch.max(symptom_logits, dim=1)
_, depression_prediction = torch.max(depression_logits, dim=1)
_, anxiety_prediction = torch.max(anxiety_logits, dim=1)

print(f"Predicted Symptom: {symptom_prediction.item()}")
print(f"Predicted Depression State: {depression_prediction.item()}")
print(f"Predicted Anxiety State: {anxiety_prediction.item()}")
