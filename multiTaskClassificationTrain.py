import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaConfig,
    RobertaModel,
    AdamW,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

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
        labels = torch.tensor(data["labels"], dtype=torch.long)

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
        self.classifiers = nn.ModuleList(
            [nn.Linear(config.hidden_size, 2) for _ in range(num_labels)]
        )
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:, 0, :]  
        logits = torch.stack([classifier(sequence_output) for classifier in self.classifiers], dim=1)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            losses = [loss_fn(logits[:, i, :], labels[:, i]) for i in range(self.num_labels)]
            total_loss = torch.mean(torch.stack(losses))
            return {"loss": total_loss, "logits": logits}
        else:
            return {"logits": logits}

def train_model():

    dataset = load_dataset("joshuasundance/mtg-coloridentity-multilabel-classification")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset = MTGDataset(dataset["train"], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    config = RobertaConfig.from_pretrained("roberta-base")
    num_labels = len(dataset["train"][0]["labels"])  
    model = MultiClassifierRobertaModel(config, num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 2
    save_path = "./saved_model_mtg"

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
    print(f"saved model path: {save_path}")


def evaluate_model():
    dataset = load_dataset("joshuasundance/mtg-coloridentity-multilabel-classification")
    tokenizer = RobertaTokenizer.from_pretrained("./saved_model_mtg")
    num_labels = len(dataset["train"][0]["labels"])
    config = RobertaConfig.from_pretrained("./saved_model_mtg")
    model = MultiClassifierRobertaModel(config, num_labels)
    model.load_state_dict(torch.load("./saved_model_mtg/pytorch_model.bin"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    validation_dataset = MTGDataset(dataset["test"], tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()

            all_labels.append(labels)
            all_predictions.append(predictions)

    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    roc_auc = roc_auc_score(all_labels, all_predictions[:, :, 1], average="macro")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")

    threshold = 0.5
    binary_predictions = (all_predictions[:, :, 1] >= threshold).astype(int)
    accuracy = accuracy_score(all_labels.flatten(), binary_predictions.flatten())
    print(f"Validation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_model()
    evaluate_model()
