import os
from io import StringIO
from pathlib import Path
from typing import Any, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from project.dataset import NewsDataset
from project.model import Model


def predict(in_data: str) -> Union[list[Any], list[list[Any]], None]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Model().to(device)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    csv_data = StringIO("content,label\n" f"{in_data},0\n")

    dataset = NewsDataset(csv_data, tokenizer)
    data_loader = DataLoader(dataset, num_workers=11, persistent_workers=True)

    ckpt = Path("src/project/models/best_model_epoch=09_val_loss=0.1466")

    if not os.path.exists(ckpt):
        ckpt = None

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    out = trainer.predict(model, dataloaders=data_loader, ckpt_path=ckpt)

    return out


if __name__ == "__main__":
    true_news = "Mexico to review need for tax changes after U.S. reform-document MEXICO CITY (Reuters) - Mexico’s finance ministry will predict whether to make fiscal changes in response to the U.S. tax reform, according to a document seen by Reuters on Friday. In the document, the ministry said Mexico would not make changes that left it with a higher public sector deficit. “Nevertheless, there will be an assessment of whether modifications should be made to Mexico’s fiscal framework,” the document said. "
    fake_news = "The Netherlands Just TROLLED Trump In His Own Words And It’s Hilariously BRILLIANT (VIDEO) While Donald Trump likes to go around saying he s only going to put America first, the rest of the world has its own opinion about that.One of the nations speaking out against Trump just absolutely and hilariously satirized him using his own words   or rather, words he s notoriously been known to use with his figures of speech. That nation is The Netherlands.The satire show Zondag Met Lubach, much like our own The Daily Show, put together a fake tourism commercial using a voice that sounded just like Trump s, and using language that he would use, including some policy proposals.Needless to say, they re not huge fans of Trump whatsoever.Dutch comedian Arjen Lubach says, as he introduces the pretend advertisement: He had a clear message to the rest of the world:  I will screw you over big time.  Well okay, he used slightly different words:  From this day forward. It s going to be only America first .And we realize it s better for us to get along [so] we decided to introduce our tiny country to him in a way that will probably appeal to him the most. From there, it was absolutely comic genius.Watch the hilarious video here:Featured image via screen capture from embedded video"

    print(predict(true_news))

    print(predict(fake_news))
