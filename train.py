import pandas as pd
import torch
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaConfig
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn

class MultiTaskRobertaModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Roberta base model without pooling layer
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        
        # Classifier for overall symptom presence: 4 labels (both, depression, anxiety, none)
        self.symptom_classifier = nn.Linear(config.hidden_size, 4)
        
        # Classifier for depression states: 6 labels (0-5)
        self.depression_classifier = nn.Linear(config.hidden_size, 6)
        
        # Classifier for anxiety states: 7 labels (0-6)
        self.anxiety_classifier = nn.Linear(config.hidden_size, 7)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize weights
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, depression_labels=None, anxiety_labels=None):
        # Get Roberta outputs
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get pooled [CLS] token output
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)

        # Overall symptom classification logits
        symptom_logits = self.symptom_classifier(pooled_output)
        
        # Depression state classification logits
        depression_logits = self.depression_classifier(pooled_output)

        # Anxiety state classification logits
        anxiety_logits = self.anxiety_classifier(pooled_output)

        loss = None
        if labels is not None and depression_labels is not None and anxiety_labels is not None:
            # Loss function
            loss_fct = nn.CrossEntropyLoss()
            # Calculate individual losses for each task
            symptom_loss = loss_fct(symptom_logits, labels)
            depression_loss = loss_fct(depression_logits, depression_labels)
            anxiety_loss = loss_fct(anxiety_logits, anxiety_labels)

            # Combine losses
            loss = symptom_loss + depression_loss + anxiety_loss
        return SequenceClassifierOutput(
            loss=loss,
            logits=(symptom_logits, depression_logits, anxiety_logits),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

def train():
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
    train_dataset = MentalHealthDataset('train_data_customize_hybrid_class_classification_depression.csv', tokenizer)
    val_dataset = MentalHealthDataset('val_data_customize_hybrid_class_classification_depression.csv', tokenizer)
    test_dataset = MentalHealthDataset('test_data_customize_hybrid_class_classification_depression.csv', tokenizer)

    # Define batch size
    batch_size = 16

    # Create DataLoader for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load configuration for Roberta
    config = RobertaConfig.from_pretrained('roberta-base')

    # Initialize your multi-task classification model
    model = MultiTaskRobertaModel(config)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Move model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 1

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            depression_labels = batch['depression_labels'].to(device)
            anxiety_labels = batch['anxiety_labels'].to(device)

            optimizer.zero_grad()

            # forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels, depression_labels=depression_labels, anxiety_labels=anxiety_labels)

            # compute loss
            loss = outputs.loss
            total_loss += loss.item()

            # backward pass
            loss.backward()
            optimizer.step()

            if step%2 == 0:
                print(f"Step {step}/{len(train_loader)} - Current Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
    torch.save(model, "multi_task_roberta_full_model.pth")
    # Evaluate on validation set
    evaluate_model(model, val_loader)

    # Evaluate on test set
    evaluate_model(model, test_loader)

def evaluate_model(model, data_loader):
    model.eval()
    total_symptom_correct = 0
    total_depression_correct = 0
    total_anxiety_correct = 0

    total_items = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # Symptom classification labels
            depression_labels = batch['depression_labels'].to(device)  # Depression state labels
            anxiety_labels = batch['anxiety_labels'].to(device)  # Anxiety state labels

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Unpack logits
            symptom_logits = outputs.logits[0]  # Symptom classification logits
            depression_logits = outputs.logits[1]  # Depression state logits
            anxiety_logits = outputs.logits[2]  # Anxiety state logits

            # Symptom classification predictions
            _, symptom_preds = torch.max(symptom_logits, dim=1)
            total_symptom_correct += torch.sum(symptom_preds == labels)

            # Depression state classification predictions (only if predicted as depression or both)
            _, depression_preds = torch.max(depression_logits, dim=1)
            valid_depression_indices = (labels == 1) | (labels == 0)  # Labels 1 (depression) and 0 (both)
            total_depression_correct += torch.sum(depression_preds[valid_depression_indices] == depression_labels[valid_depression_indices])

            # Anxiety state classification predictions (only if predicted as anxiety or both)
            _, anxiety_preds = torch.max(anxiety_logits, dim=1)
            valid_anxiety_indices = (labels == 2) | (labels == 0)  # Labels 2 (anxiety) and 0 (both)
            total_anxiety_correct += torch.sum(anxiety_preds[valid_anxiety_indices] == anxiety_labels[valid_anxiety_indices])

            # Count total items
            total_items += labels.size(0)

    # Compute accuracies
    symptom_accuracy = total_symptom_correct.double() / total_items
    depression_accuracy = total_depression_correct.double() / valid_depression_indices.sum().double()
    anxiety_accuracy = total_anxiety_correct.double() / valid_anxiety_indices.sum().double()

    print(f'Symptom Classification Accuracy: {symptom_accuracy:.4f}')
    print(f'Depression State Accuracy: {depression_accuracy:.4f}')
    print(f'Anxiety State Accuracy: {anxiety_accuracy:.4f}')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()