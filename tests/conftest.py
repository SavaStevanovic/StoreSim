"""Shared pytest fixtures and mock helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import MagicMock, patch


EMBEDDING_DIM = 1024  # RN50 output dimension


# ---------------------------------------------------------------------------
# Mock CLIP model & preprocess
# ---------------------------------------------------------------------------

def _make_mock_clip_model(embedding_dim: int = EMBEDDING_DIM) -> MagicMock:
    """Return a MagicMock that behaves like openai CLIP's model object."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model

    # Simulate encode_image / encode_text returning random float16 tensors
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_clip(monkeypatch):
    """Patch clip.load and clip.tokenize globally for all tests that use it."""
    mock_model = _make_mock_clip_model()
    mock_preprocess = _make_mock_preprocess()

    with patch("clip.load", return_value=(mock_model, mock_preprocess)) as mock_load, \
         patch("clip.tokenize", side_effect=lambda texts, **kw: torch.zeros(len(texts), 77, dtype=torch.long)) as mock_tokenize:
        yield {
            "load": mock_load,
            "tokenize": mock_tokenize,
            "model": mock_model,
            "preprocess": mock_preprocess,
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
