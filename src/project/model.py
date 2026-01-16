import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoModel


class Model(pl.LightningModule):
    """Binary text classification model."""

    def __init__(self, model_name: str = "distilbert-base-uncased", lr: float = 2e-5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        for param in self.encoder.parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_embedding)
        return logits.squeeze(-1)

    def training_step(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        logits = self(input_ids, attention_mask)

        # Loss and accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == labels).float().mean()
        loss = self.loss_fn(logits, labels)

        self.log("loss/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("acc/train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch) -> None:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy for binary classification
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == labels).float().mean()

        self.log("loss/val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("acc/val_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.classifier.parameters(), lr=self.lr)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = Model()

    batch = tokenizer(
        ["this is a test"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    out = model(batch["input_ids"], batch["attention_mask"])
    print(out.shape)
