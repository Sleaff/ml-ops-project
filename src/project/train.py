import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from project.data import MyDataset
from project.dataset import NewsDataset
from project.model import Model


def train(checkpoint_path: Path | None = None, epochs: int = 10, train_amount: float = 1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    if not os.path.exists("src/project/data/processed/news.csv"):
        dataset = MyDataset(Path("src/project/data/"))
        dataset.preprocess(Path("src/project/data/processed/"))

    dataset = NewsDataset("src/project/data/processed/news.csv", tokenizer)
    save_dir = Path("src/project/models")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=True)

    model = Model().to(device)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="best_model_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        limit_train_batches=train_amount,
        callbacks=[checkpoint_callback],
        default_root_dir=save_dir,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    if checkpoint_path is not None:
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, train_loader, val_loader)

    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train news classification model")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to checkpoint file to load (default: src/project/models/latest_checkpoint.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--training-size",
        type=float,
        default=1.0,
        help="amount of training data to use",
    )
    args = parser.parse_args()

    train(
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        train_amount=args.training_size,
    )
