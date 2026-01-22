from pathlib import Path

import torch
from project.model import Model
from transformers import AutoTokenizer


def test_model_inference():
    model = Model.load_from_checkpoint("models/best_model_epoch=09_val_loss=0.1466.ckpt")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    text = "Breaking news: this is a test article"
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    with torch.no_grad():
        output = model(encoding["input_ids"], encoding["attention_mask"])
        prediction = (torch.sigmoid(output) > 0.5).float()

    assert output.shape == torch.Size([1]), "Model output shape incorrect"
    assert prediction.item() in [0.0, 1.0], "Prediction should be binary"
