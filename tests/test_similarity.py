"""Tests for SimilaritySearch."""

from __future__ import annotations

import pytest
import torch

from tests.conftest import EMBEDDING_DIM


@pytest.fixture()
def searcher(random_embeddings):
    from storesim.similarity import SimilaritySearch

    labels = [f"product_{i}" for i in range(len(random_embeddings))]
    return SimilaritySearch(embeddings=random_embeddings, labels=labels)


@pytest.fixture()
def searcher_no_labels(random_embeddings):
    from storesim.similarity import SimilaritySearch

    return SimilaritySearch(embeddings=random_embeddings)


class TestSimilaritySearchInit:
    def test_init_stores_embeddings(self, random_embeddings):
        from storesim.similarity import SimilaritySearch

        s = SimilaritySearch(random_embeddings)
        assert s.embeddings.shape == random_embeddings.shape

    def test_init_rejects_1d_tensor(self):
        from storesim.similarity import SimilaritySearch

        with pytest.raises(ValueError, match="2-D"):
            SimilaritySearch(torch.randn(EMBEDDING_DIM))

    def test_init_rejects_3d_tensor(self):
        from storesim.similarity import SimilaritySearch

        with pytest.raises(ValueError, match="2-D"):
            SimilaritySearch(torch.randn(2, 3, EMBEDDING_DIM))

    def test_labels_stored(self, random_embeddings):
        from storesim.similarity import SimilaritySearch

        labels = [str(i) for i in range(len(random_embeddings))]
        s = SimilaritySearch(random_embeddings, labels=labels)
        assert s.labels == labels

    def test_embeddings_cast_to_float32(self, random_embeddings):
        from storesim.similarity import SimilaritySearch

        s = SimilaritySearch(random_embeddings.half())
        assert s.embeddings.dtype == torch.float32


class TestSimilaritySearchQuery:
    def test_query_returns_top_k(self, searcher, random_embeddings):
        query = random_embeddings[0]
        result = searcher.query(query, top_k=3)
        assert result["scores"].shape == (3,)
        assert result["indices"].shape == (3,)
        assert len(result["labels"]) == 3

    def test_query_scores_descending(self, searcher, random_embeddings):
        """Scores should be sorted in descending order."""
        query = random_embeddings[0]
        result = searcher.query(query, top_k=5)
        scores = result["scores"]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_query_exact_match_is_first(self, random_embeddings):
        """Querying with an item from the index should return itself as top-1."""
        from storesim.similarity import SimilaritySearch

        s = SimilaritySearch(random_embeddings)
        query = random_embeddings[4]
        result = s.query(query, top_k=1)
        assert result["indices"][0].item() == 4

    def test_query_scores_in_valid_range(self, searcher, random_embeddings):
        """Cosine similarities must be in [-1, 1]."""
        query = random_embeddings[0]
        result = searcher.query(query, top_k=5)
        assert result["scores"].min().item() >= -1.0 - 1e-5
        assert result["scores"].max().item() <= 1.0 + 1e-5

    def test_query_labels_returned(self, searcher, random_embeddings):
        query = random_embeddings[0]
        result = searcher.query(query, top_k=3)
        for label in result["labels"]:
            assert label.startswith("product_")

    def test_query_no_labels_returns_empty_list(self, searcher_no_labels, random_embeddings):
        query = random_embeddings[0]
        result = searcher_no_labels.query(query, top_k=3)
        assert result["labels"] == []

    def test_query_top_k_capped_at_index_size(self, searcher, random_embeddings):
        """top_k larger than the index should not raise; clamp to index size."""
        query = random_embeddings[0]
        result = searcher.query(query, top_k=1000)
        assert result["scores"].shape[0] == len(random_embeddings)

    def test_query_accepts_2d_input(self, searcher, random_embeddings):
        """Query embedding shaped (1, dim) should also work."""
        query = random_embeddings[0].unsqueeze(0)
        result = searcher.query(query, top_k=3)
        assert result["scores"].shape == (3,)


class TestSimilaritySearchBatch:
    def test_query_batch_length(self, searcher, random_embeddings):
        queries = random_embeddings[:3]
        results = searcher.query_batch(queries, top_k=2)
        assert len(results) == 3

    def test_query_batch_each_has_correct_shape(self, searcher, random_embeddings):
        queries = random_embeddings[:4]
        results = searcher.query_batch(queries, top_k=3)
        for r in results:
            assert r["scores"].shape == (3,)

    def test_query_batch_top1_contains_self(self, random_embeddings):
        """Each item should have itself as the top-1 result."""
        from storesim.similarity import SimilaritySearch

        s = SimilaritySearch(random_embeddings)
        results = s.query_batch(random_embeddings, top_k=1)
        for i, r in enumerate(results):
            assert r["indices"][0].item() == i


class TestSimilaritySearchPersistence:
    def test_save_and_load_roundtrip(self, random_embeddings, tmp_path):
        from storesim.similarity import SimilaritySearch

        labels = [f"item_{i}" for i in range(len(random_embeddings))]
        s = SimilaritySearch(random_embeddings, labels=labels)
        index_path = str(tmp_path / "index.pt")
        s.save(index_path)

        loaded = SimilaritySearch.load(index_path)
        assert torch.allclose(s.embeddings, loaded.embeddings)
        assert loaded.labels == labels

    def test_save_and_load_no_labels(self, random_embeddings, tmp_path):
        from storesim.similarity import SimilaritySearch

        s = SimilaritySearch(random_embeddings)
        index_path = str(tmp_path / "index.pt")
        s.save(index_path)

        loaded = SimilaritySearch.load(index_path)
        assert loaded.labels is None

    def test_saved_index_query_consistent(self, random_embeddings, tmp_path):
        """Querying a saved/loaded index should give the same results."""
        from storesim.similarity import SimilaritySearch

        s = SimilaritySearch(random_embeddings)
        index_path = str(tmp_path / "index.pt")
        s.save(index_path)
        loaded = SimilaritySearch.load(index_path)

        query = random_embeddings[0]
        original = s.query(query, top_k=5)
        restored = loaded.query(query, top_k=5)
        assert torch.allclose(original["scores"], restored["scores"])
        assert torch.equal(original["indices"], restored["indices"])
