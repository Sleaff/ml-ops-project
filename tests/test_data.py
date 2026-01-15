from project.dataset import NewsDataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def test_news_dataset():
    """Test the NewsDataset class."""

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = NewsDataset("data/processed/news.csv", tokenizer)
    assert isinstance(dataset, Dataset)
