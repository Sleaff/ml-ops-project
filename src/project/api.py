import os
import time
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path

import pytorch_lightning as pl
import torch
from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from project.dataset import NewsDataset
from project.model import Model

# Prometheus metrics
REQUEST_COUNT = Counter("api_requests_total", "Total number of API requests", ["endpoint", "method"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"])
PREDICTION_COUNT = Counter("api_predictions_total", "Total predictions by class", ["prediction"])


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

    ckpt = Path("models/best_model_epoch=09_val_loss=0.1466.ckpt")

    if not os.path.exists(ckpt):
        ckpt = None
        print("Warning: No model checkpoint found, running without checkpoint")

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


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/")
async def get_model():
    REQUEST_COUNT.labels(endpoint="/model/", method="GET").inc()
    return ckpt


@app.post("/news/")
async def create_news(news: NewsItem):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/news/", method="POST").inc()

    content = news.title + " " + news.text

    csv_data = StringIO("content,label\n" f"{content},0\n")

    dataset = NewsDataset(csv_data, tokenizer)
    data_loader = DataLoader(dataset, num_workers=1, persistent_workers=True)
    out = trainer.predict(model, dataloaders=data_loader, ckpt_path=ckpt)
    pred = int(out[0].item())

    if pred == 1:
        prediction = "REAL NEWS"
    else:
        prediction = "FAKE NEWS"

    # Record metrics
    PREDICTION_COUNT.labels(prediction=prediction).inc()
    REQUEST_LATENCY.labels(endpoint="/news/").observe(time.time() - start_time)

    return f"This is {prediction}"
