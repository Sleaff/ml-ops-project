import os
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Union

import pytorch_lightning as pl
import torch
from fastapi import FastAPI, Form
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from project.dataset import NewsDataset
from project.model import Model


class NewsItem(BaseModel):
    title: str
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, trainer, ckpt, tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Model().to(device)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    ckpt = Path("models/best_model_epoch=09_val_loss=0.1466")

    if not os.path.exists(ckpt):
        ckpt = None

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    yield

    print("cleaning up")
    del model, tokenizer, ckpt, trainer, device


app = FastAPI(lifespan=lifespan)


@app.post("/news/")
async def create_news(news: NewsItem):
    content = news.title + " " + news.text

    csv_data = StringIO("content,label\n" f"{content},0\n")

    dataset = NewsDataset(csv_data, tokenizer)
    data_loader = DataLoader(dataset, num_workers=1)
    out = trainer.predict(model, dataloaders=data_loader, ckpt_path=ckpt)
    pred = int(out[0].item())

    if pred == 1:
        prediction = "FAKE NEWS"
    else:
        prediction = "REAL NEWS"

    return f"This is {prediction}"
