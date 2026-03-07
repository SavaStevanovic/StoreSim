"""Tests for EmbeddingExtractor."""

from __future__ import annotations

import torch

from tests.conftest import EMBEDDING_DIM


class TestEmbeddingExtractor:
    def test_extract_from_single_image(self, clip_model, dummy_image):
        """Extracting from one image should return shape (1, embedding_dim)."""
        from storesim.embeddings import EmbeddingExtractor

        extractor = EmbeddingExtractor(model=clip_model, batch_size=8)
        result = extractor.extract_from_images([dummy_image], show_progress=False)
        assert result.shape == (1, EMBEDDING_DIM)

    def test_extract_from_multiple_images(self, clip_model, dummy_image):
        """Extracting from N images should return shape (N, embedding_dim)."""
        from storesim.embeddings import EmbeddingExtractor

        extractor = EmbeddingExtractor(model=clip_model, batch_size=4)
        images = [dummy_image] * 9  # tests batch boundary (2 full + 1 partial)
        result = extractor.extract_from_images(images, show_progress=False)
        assert result.shape == (9, EMBEDDING_DIM)

    def test_extract_from_images_returns_float32(self, clip_model, dummy_image):
        """Output tensor should be float32."""
        from storesim.embeddings import EmbeddingExtractor

        extractor = EmbeddingExtractor(model=clip_model)
        result = extractor.extract_from_images([dummy_image], show_progress=False)
        assert result.dtype == torch.float32

    def test_extract_from_images_normalised(self, clip_model, dummy_image):
        """Each embedding row should have L2 norm ≈ 1."""
        from storesim.embeddings import EmbeddingExtractor

        extractor = EmbeddingExtractor(model=clip_model)
        result = extractor.extract_from_images([dummy_image] * 4, show_progress=False)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-4)

    def test_extract_from_images_on_cpu(self, clip_model, dummy_image):
        """Result should always be on CPU."""
        from storesim.embeddings import EmbeddingExtractor

        extractor = EmbeddingExtractor(model=clip_model)
        result = extractor.extract_from_images([dummy_image], show_progress=False)
        assert result.device.type == "cpu"

    def test_extract_from_images_from_path(self, clip_model, dummy_image, tmp_path):
        """Extractor should accept file path strings."""
        from storesim.embeddings import EmbeddingExtractor

        img_path = tmp_path / "img.png"
        dummy_image.save(img_path)
        extractor = EmbeddingExtractor(model=clip_model)
        result = extractor.extract_from_images([str(img_path)], show_progress=False)
        assert result.shape == (1, EMBEDDING_DIM)

    def test_extract_from_single_text(self, clip_model, mock_clip):
        """Extracting from one text should return shape (1, embedding_dim)."""
        from storesim.embeddings import EmbeddingExtractor

        extractor = EmbeddingExtractor(model=clip_model)
        result = extractor.extract_from_texts(["blue shoes"], show_progress=False)
        assert result.shape == (1, EMBEDDING_DIM)

    def test_extract_from_multiple_texts(self, clip_model, mock_clip):
        """Extracting from N texts should return shape (N, embedding_dim)."""
        from storesim.embeddings import EmbeddingExtractor

        extractor = EmbeddingExtractor(model=clip_model, batch_size=3)
        texts = ["a", "b", "c", "d", "e"]
        result = extractor.extract_from_texts(texts, show_progress=False)
        assert result.shape == (5, EMBEDDING_DIM)

    def test_extract_texts_normalised(self, clip_model, mock_clip):
        """Each text embedding row should have L2 norm ≈ 1."""
        from storesim.embeddings import EmbeddingExtractor

        extractor = EmbeddingExtractor(model=clip_model)
        texts = ["shoes", "hat", "jacket"]
        result = extractor.extract_from_texts(texts, show_progress=False)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-4)

    def test_default_model_created_when_none(self, mock_clip):
        """EmbeddingExtractor creates a CLIPResNet50Model if none is supplied."""
        from storesim.embeddings import EmbeddingExtractor

        extractor = EmbeddingExtractor()
        assert extractor.model is not None

    def test_batch_size_respected(self, clip_model, dummy_image):
        """The model's encode_image should be called in correct batch chunks."""
        from storesim.embeddings import EmbeddingExtractor

        batch_size = 3
        n_images = 7
        extractor = EmbeddingExtractor(model=clip_model, batch_size=batch_size)
        _ = extractor.extract_from_images([dummy_image] * n_images, show_progress=False)

        expected_calls = (n_images + batch_size - 1) // batch_size
        assert clip_model.model.encode_image.call_count == expected_calls
