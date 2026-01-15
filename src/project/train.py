import argparse
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


def train(checkpoint_path=None, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    if not os.path.exists("src/project/data/processed/news.csv"):
        dataset = MyDataset(Path("src/project/data/"))
        dataset.preprocess(Path("src/project/data/processed/"))

    dataset = NewsDataset("src/project/data/processed/news.csv", tokenizer)
    save_dir = Path("src/project/models")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, _ = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    model = Model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    best_loss = float("inf")
    if checkpoint_path != None:
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_loss = checkpoint.get("best_loss", float("inf"))
            print("Checkpoint loaded successfully!")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Starting from scratch.")

    model.train()
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*50}")
        total_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Training completed - Average Loss: {avg_loss:.4f}")

        latest_checkpoint_path = save_dir / f"latest_checkpoint_{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_size": train_size,
                "val_size": val_size,
                "loss": avg_loss,
                "best_loss": best_loss,
            },
            latest_checkpoint_path,
        )
        print(f"Latest checkpoint saved to {latest_checkpoint_path}")

        if avg_loss < best_loss:
            print(f"Loss improved from {best_loss:.4f} to {avg_loss:.4f}! Saving best model...")
            best_loss = avg_loss

            # Save best model state dict
            best_model_path = save_dir / f"best_model_{epoch}.pt"
            torch.save(model.state_dict(), best_model_path)

            # Save best checkpoint
            best_checkpoint_path = save_dir / f"best_checkpoint_{epoch}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_size": train_size,
                    "val_size": val_size,
                    "loss": avg_loss,
                    "best_loss": best_loss,
                },
                best_checkpoint_path,
            )
            print(f"Best model saved to {best_model_path}")
            print(f"Best checkpoint saved to {best_checkpoint_path}")
        else:
            print(f"Loss did not improve (current: {avg_loss:.4f}, best: {best_loss:.4f})")

    print(f"\n{'='*50}")
    print(f"Training completed! Total epochs: {epochs}")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train news classification model")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to checkpoint file to load (default: src/project/models/latest_checkpoint.pt)",
    )
    args = parser.parse_args()

    train(checkpoint_path=args.checkpoint_path)
