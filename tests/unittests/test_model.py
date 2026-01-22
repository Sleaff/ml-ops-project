from unittest.mock import MagicMock, patch

import pytest
import torch


def test_model_has_correct_loss_function():
    """Test model uses BCEWithLogitsLoss."""
    with patch("project.model.AutoModel"):
        from project.model import Model

        model = Model()
        assert isinstance(model.loss_fn, torch.nn.BCEWithLogitsLoss)


def test_model_has_correct_learning_rate():
    """Test model initializes with correct learning rate."""
    with patch("project.model.AutoModel"):
        from project.model import Model

        model = Model(lr=1e-4)
        assert model.lr == 1e-4


def test_model_classifier_output_size():
    """Test classifier outputs single value for binary classification."""
    with patch("project.model.AutoModel") as mock_auto:
        mock_encoder = MagicMock()
        mock_encoder.config.hidden_size = 768
        mock_encoder.parameters.return_value = []
        mock_auto.from_pretrained.return_value = mock_encoder

        from project.model import Model

        model = Model()

        assert model.classifier.out_features == 1


def test_predictions_are_binary():
    """Test that sigmoid + threshold produces binary outputs."""
    logits = torch.tensor([2.0, -2.0, 0.5, -0.5])
    preds = (torch.sigmoid(logits) > 0.5).float()
    assert torch.all((preds == 0) | (preds == 1))
