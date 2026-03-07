"""Tests for CLIPResNet50Model."""

from __future__ import annotations

import torch
import pytest

from tests.conftest import EMBEDDING_DIM


class TestCLIPResNet50Model:
    def test_model_loads_rn50(self, mock_clip):
        """Model should request 'RN50' from clip.load."""
        from storesim.model import CLIPResNet50Model

        model = CLIPResNet50Model(device="cpu")
        mock_clip["load"].assert_called_once_with("RN50", device="cpu")

    def test_default_device_is_cpu_or_cuda(self, mock_clip):
        """Device defaults to 'cpu' when CUDA is unavailable."""
        import storesim.model as m
        model = m.CLIPResNet50Model(device="cpu")
        assert model.device == "cpu"

    def test_embedding_dim_property(self, clip_model):
        """embedding_dim should equal EMBEDDING_DIM."""
        assert clip_model.embedding_dim == EMBEDDING_DIM

    def test_encode_image_returns_1d_tensor(self, clip_model, dummy_image):
        """encode_image with a PIL image returns a 1-D float tensor."""
        emb = clip_model.encode_image(dummy_image)
        assert emb.ndim == 1
        assert emb.dtype == torch.float32

    def test_encode_image_from_path(self, clip_model, tmp_path, dummy_image):
        """encode_image should accept a file path string."""
        img_path = tmp_path / "product.png"
        dummy_image.save(img_path)
        emb = clip_model.encode_image(str(img_path))
        assert emb.shape == (EMBEDDING_DIM,)

    def test_encode_image_normalised(self, clip_model, dummy_image):
        """Normalised embedding should have unit L2 norm (within float tolerance)."""
        emb = clip_model.encode_image(dummy_image, normalize=True)
        assert abs(emb.norm().item() - 1.0) < 1e-4

    def test_encode_image_not_normalised(self, clip_model, dummy_image):
        """When normalize=False the embedding is not forced to unit norm."""
        # The mock returns random values; just verify shape/dtype without norm check
        emb = clip_model.encode_image(dummy_image, normalize=False)
        assert emb.shape == (EMBEDDING_DIM,)

    def test_encode_single_text_returns_1d(self, clip_model):
        """A single string should yield a 1-D tensor."""
        emb = clip_model.encode_text("blue running shoes")
        assert emb.ndim == 1
        assert emb.shape == (EMBEDDING_DIM,)

    def test_encode_text_list_returns_2d(self, clip_model):
        """A list of strings should yield a 2-D tensor."""
        texts = ["blue shoes", "red hat", "green jacket"]
        embs = clip_model.encode_text(texts)
        assert embs.ndim == 2
        assert embs.shape == (3, EMBEDDING_DIM)

    def test_encode_text_normalised(self, clip_model):
        """Normalised text embeddings should have unit norms."""
        emb = clip_model.encode_text("a product", normalize=True)
        assert abs(emb.norm().item() - 1.0) < 1e-4

    def test_encode_text_and_image_same_dim(self, clip_model, dummy_image):
        """Image and text embeddings must share the same dimensionality."""
        img_emb = clip_model.encode_image(dummy_image)
        txt_emb = clip_model.encode_text("a product")
        assert img_emb.shape == txt_emb.shape

    def test_model_eval_mode(self, clip_model):
        """The underlying CLIP model should be set to eval mode."""
        clip_model.model.eval.assert_called()

    def test_encode_image_output_on_cpu(self, clip_model, dummy_image):
        """Result tensor should be on CPU regardless of model device."""
        emb = clip_model.encode_image(dummy_image)
        assert emb.device.type == "cpu"

    def test_encode_text_output_on_cpu(self, clip_model):
        """Result tensor should be on CPU regardless of model device."""
        emb = clip_model.encode_text("shoes")
        assert emb.device.type == "cpu"
