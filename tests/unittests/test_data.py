from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import torch
from project.dataset import NewsDataset


def test_dataset_can_be_created():
    """Test that NewsDataset can be instantiated."""
    mock_tokenizer = MagicMock()
    csv_data = StringIO("content,label\ntest,0\n")
    dataset = NewsDataset(csv_data, mock_tokenizer)

    assert dataset is not None
    assert hasattr(dataset, "tokenizer")


def test_dataset_stores_data():
    """Test that dataset reads CSV data."""
    mock_tokenizer = MagicMock()
    csv_data = StringIO("content,label\nnews1,0\nnews2,1\n")
    dataset = NewsDataset(csv_data, mock_tokenizer)

    assert len(dataset) == 2
    assert len(dataset.texts) == 2
    assert len(dataset.labels) == 2


def test_real_news_data():
    """Test loading and properties of real news dataset."""
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "news.csv"

    if not data_path.exists():
        return

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.randint(0, 1000, (1, 128)), "attention_mask": torch.ones(1, 128)}

    with open(data_path) as f:
        dataset = NewsDataset(f, mock_tokenizer, max_length=128)

    assert len(dataset) > 0

    unique_labels = set(dataset.labels)
    assert 0 in unique_labels
    assert 1 in unique_labels
