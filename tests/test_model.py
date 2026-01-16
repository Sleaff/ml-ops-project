import copy
import csv
from io import StringIO

import pandas as pd
import pytorch_lightning as pl
import torch
from project.dataset import NewsDataset
from project.model import Model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def test_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = Model()

    csv_data = StringIO("content,label\n" "this is a test,0\n")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = NewsDataset(csv_data, tokenizer)
    data_loader = DataLoader(dataset)

    assert torch.any(data_loader.dataset[0]["input_ids"]), "Data error"
    assert data_loader.batch_size == 1, "Batch size error"

    initial_state = copy.deepcopy(model.state_dict())

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, data_loader)

    changed = False
    for k in initial_state:
        if not torch.equal(initial_state[k], model.state_dict()[k]):
            changed = True
            break

    assert changed, "Model parameters did not change during training"
