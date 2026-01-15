import pandas as pd
import torch
from torch.utils.data import Dataset
import csv


class NewsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length: int = 256):
        df = pd.read_csv(csv_path)

        self.texts = df["content"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = NewsDataset("data/processed/news.csv", tokenizer)

    sample = dataset[0]
    for k, v in sample.items():
        print(k, v.shape)
