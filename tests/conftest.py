"""Shared pytest fixtures and mock helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

EMBEDDING_DIM = 1024  # RN50 output dimension


# ---------------------------------------------------------------------------
# Mock open_clip model, preprocess and tokenizer
# ---------------------------------------------------------------------------


def _make_mock_open_clip_model(embedding_dim: int = EMBEDDING_DIM) -> MagicMock:
    """Return a MagicMock that behaves like open_clip's model object."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model

    def encode_image(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.randn(batch, embedding_dim, dtype=torch.float16)

    def encode_text(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.randn(batch, embedding_dim, dtype=torch.float16)

    mock_model.encode_image.side_effect = encode_image
    mock_model.encode_text.side_effect = encode_text
    mock_model.visual.output_dim = embedding_dim
    return mock_model


def _make_mock_preprocess() -> MagicMock:
    """Return a callable mock that converts a PIL Image to a float tensor."""

    def preprocess(img: Image.Image) -> torch.Tensor:
        return torch.zeros(3, 224, 224)

    return preprocess


def _make_mock_tokenizer() -> MagicMock:
    """Return a callable mock that converts texts to token tensors."""

    def tokenizer(texts):
        n = len(texts) if isinstance(texts, list) else 1
        return torch.zeros(n, 77, dtype=torch.long)

    return tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_clip(monkeypatch):
    """Patch open_clip.create_model_and_transforms and get_tokenizer."""
    mock_model = _make_mock_open_clip_model()
    mock_preprocess = _make_mock_preprocess()
    mock_tokenizer = _make_mock_tokenizer()

    with (
        patch(
            "open_clip.create_model_and_transforms",
            return_value=(mock_model, None, mock_preprocess),
        ) as mock_create,
        patch(
            "open_clip.get_tokenizer",
            return_value=mock_tokenizer,
        ) as mock_get_tokenizer,
    ):
        yield {
            "create": mock_create,
            "get_tokenizer": mock_get_tokenizer,
            "model": mock_model,
            "preprocess": mock_preprocess,
            "tokenizer": mock_tokenizer,
        }


@pytest.fixture()
def clip_model(mock_clip):
    """Return an initialised CLIPResNet50Model backed by the mock."""
    from storesim.model import CLIPResNet50Model

    return CLIPResNet50Model(device="cpu")


@pytest.fixture()
def dummy_image() -> Image.Image:
    """Return a small dummy RGB PIL image."""
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture()
def random_embeddings() -> torch.Tensor:
    """Return a normalised (10, 1024) float tensor."""
    embs = torch.randn(10, EMBEDDING_DIM)
    return torch.nn.functional.normalize(embs, dim=-1)
