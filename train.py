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

class SymptomClassifier(nn.Module):
    """Classifier head for symptom classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 4)  

    def forward(self, features, **kwargs):

        x = features[:, 0, :]

        x = self.dropout(x)

        x = self.dense(x)

        x = torch.tanh(x)

        x = self.dropout(x)

        x = self.out_proj(x)
        return x

class depression_classifier(nn.Module):
    """Classifier head for symptom classification tasks."""

    def __init__(self, config):
        super().__init__()
   
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.out_proj = nn.Linear(config.hidden_size, 6)  

    def forward(self, features, **kwargs):

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
        self.out_proj = nn.Linear(config.hidden_size, 8)

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

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, symptom_labels=None, depression_labels=None, anxiety_labels=None):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        symptom_logits = self.symptom_classifier(outputs.last_hidden_state)

        depression_logits = self.depression_classifier(outputs.last_hidden_state)

        anxiety_logits = self.anxiety_classifier(outputs.last_hidden_state)

        loss = None

        if symptom_labels is not None and depression_labels is not None and anxiety_labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            symptom_loss = loss_fct(symptom_logits, symptom_labels)
            depression_loss = loss_fct(depression_logits, depression_labels)
            anxiety_loss = loss_fct(anxiety_logits, anxiety_labels)

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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    set_seed(3407)
    # Define the dataset class to handle the CSV
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

    # Initialize the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Create datasets
    train_dataset = MentalHealthDataset('train.csv', tokenizer)
    # val_dataset = MentalHealthDataset('val_data_customize_hybrid_class_classification_depression.csv', tokenizer)
    test_dataset = MentalHealthDataset('balanced_test_data.csv', tokenizer)

    # Define batch size
    batch_size = 32

    # Create DataLoader for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load configuration for Roberta
    config = RobertaConfig.from_pretrained('roberta-large')

    # Initialize your multi-task classification model
    model = MultiTaskRobertaModel(config)

    # Define optimizer and scheduler
    # optimizer = AdamW(model.parameters(), lr=5e-5)
    optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9) 

    # Move model to GPU if available
    model.to(device)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            depression_labels = batch['depression_labels'].to(device)
            anxiety_labels = batch['anxiety_labels'].to(device)

            optimizer.zero_grad()

            # forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, symptom_labels=labels, depression_labels=depression_labels, anxiety_labels=anxiety_labels)

            # compute loss
            loss = outputs.loss
            total_loss += loss.item()

            # backward pass
            loss.backward()
            optimizer.step()

            if step % 2 == 0:
                print(f"Step {step}/{len(train_loader)} - Current Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

    # Save the model and tokenizer
    model.save_pretrained("./saved_model_SGD")
    tokenizer.save_pretrained("./saved_model_SGD")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()
    # evaluate_model()