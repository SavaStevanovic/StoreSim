"""Cosine-similarity-based product search."""

from __future__ import annotations

import torch


class SimilaritySearch:
    """Nearest-neighbour search over a pre-built embedding index.

    Args:
        embeddings: Tensor of shape ``(N, embedding_dim)``.  The rows should
                    already be L2-normalised.
        labels: Optional list of labels / identifiers aligned with *embeddings*.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: list[str] | None = None,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D, got shape {tuple(embeddings.shape)}"
            )
        self.embeddings = embeddings.float()
        self.labels = labels

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> dict[str, torch.Tensor | list[str]]:
        """Return the top-k most similar items.

        Args:
            query_embedding: 1-D tensor of shape ``(embedding_dim,)``.
            top_k: Number of results to return.

        Returns:
            Dictionary with keys:
            - ``"scores"``: cosine similarity scores tensor of shape ``(top_k,)``.
            - ``"indices"``: integer indices of the top-k matches.
            - ``"labels"``: list of labels if provided during construction,
              otherwise an empty list.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)

        query_embedding = torch.nn.functional.normalize(
            query_embedding.float(), dim=-1
        )
        scores = (query_embedding @ self.embeddings.T).squeeze(0)

        top_k = min(top_k, scores.shape[0])
        scores_topk, indices = torch.topk(scores, top_k)

        result_labels: list[str] = []
        if self.labels is not None:
            result_labels = [self.labels[i.item()] for i in indices]

        return {
            "scores": scores_topk,
            "indices": indices,
            "labels": result_labels,
        }

    def query_batch(
        self,
        query_embeddings: torch.Tensor,
        top_k: int = 5,
    ) -> list[dict[str, torch.Tensor | list[str]]]:
        """Run :meth:`query` for each row of *query_embeddings*.

        Args:
            query_embeddings: Tensor of shape ``(B, embedding_dim)``.
            top_k: Number of results per query.

        Returns:
            List of result dicts, one per query.
        """
        return [self.query(q, top_k=top_k) for q in query_embeddings]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the index (embeddings + labels) to disk.

        Args:
            path: Output file path (``*.pt``).
        """
        torch.save({"embeddings": self.embeddings, "labels": self.labels}, path)

    @classmethod
    def load(cls, path: str) -> "SimilaritySearch":
        """Load an index previously saved with :meth:`save`.

        Args:
            path: Path to the saved ``.pt`` file.

        Returns:
            Initialised :class:`SimilaritySearch` instance.
        """
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(embeddings=data["embeddings"], labels=data["labels"])
