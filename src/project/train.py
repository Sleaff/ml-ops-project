import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer

from project.data import MyDataset
from project.dataset import NewsDataset
from project.model import Model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    if not os.path.exists("data/processed/news.csv"):
        dataset = MyDataset(Path("src/project/data/"))
        dataset.preprocess(Path("data/processed/"))

    dataset = NewsDataset("data/processed/news.csv", tokenizer)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, _ = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    model = Model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

    print("Training step completed")


if __name__ == "__main__":
    train()
