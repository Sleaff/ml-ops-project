from pathlib import Path

from project.dataset import NewsDataset
from transformers import AutoTokenizer


def test_dvc_data_exists():
    data_dir = Path("data/processed")

    assert data_dir.exists(), "data/processed directory doesn't exist"

    news_csv = data_dir / "news.csv"
    assert news_csv.exists(), "news.csv not found - DVC pull may have failed"
    assert news_csv.stat().st_size > 0, "news.csv is empty"


def test_real_data_can_be_loaded():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    with open("data/processed/news.csv") as f:
        dataset = NewsDataset(f, tokenizer, max_length=128)

    assert len(dataset) > 1000, "Dataset seems too small"

    sample = dataset[0]
    assert sample["input_ids"].shape == (128,)
    assert sample["label"] in [0.0, 1.0]
