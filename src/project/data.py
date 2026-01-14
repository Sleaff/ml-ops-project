from pathlib import Path

import pandas as pd
import typer
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Fake vs real news dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

        fake = pd.read_csv(self.data_path / "Fake.csv")
        real = pd.read_csv(self.data_path / "True.csv")

        fake["label"] = 0
        real["label"] = 1

        df = pd.concat([fake, real], ignore_index=True)

        df["content"] = df["title"] + " " + df["text"]
        self.data = df[["content", "label"]].dropna().reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        return {
            "text": row["content"],
            "label": int(row["label"]),
        }

    def preprocess(self, output_folder: Path) -> None:
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / "news.csv"
        self.data.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")



def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
