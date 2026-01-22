# tests/unittests/test_api.py
import pytest
from project.api import NewsItem


def test_news_item_valid():
    """Test NewsItem accepts valid input."""
    item = NewsItem(title="Test", text="Content")
    assert item.title == "Test"
    assert item.text == "Content"


def test_news_item_requires_title():
    """Test NewsItem requires title field."""
    with pytest.raises(Exception):
        NewsItem(text="Only text")


def test_news_item_requires_text():
    """Test NewsItem requires text field."""
    with pytest.raises(Exception):
        NewsItem(title="Only title")
