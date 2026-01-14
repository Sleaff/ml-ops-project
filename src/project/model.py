import torch
from torch import nn
from transformers import AutoModel


class Model(nn.Module):
    """Binary text classification model."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        for param in self.encoder.parameters():
            param.requires_grad = False
        
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_embedding)
        return logits.squeeze(-1)


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
