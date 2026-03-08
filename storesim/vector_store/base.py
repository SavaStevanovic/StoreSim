"""Abstract base class for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class VectorStore(ABC):
    """Common interface for vector store backends.

    Each record stored has:
        - ``id``           - unique INT64 primary key for the vector row
        - ``product_id``   - foreign key back to the products table
        - ``image_vector`` - CLIP image embedding
        - ``text_vector``  - CLIP text embedding
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @abstractmethod
    def insert(self, records: list[dict[str, Any]]) -> None:
        """Insert pre-computed embedding records.

        Each dict must contain:
            id (int), product_id (int),
            image_vector (list[float] | torch.Tensor),
            text_vector  (list[float] | torch.Tensor)
        """

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @abstractmethod
    def get_vector(self, id: int) -> dict[str, Any] | None:
        """Return the stored record for *id*, or ``None`` if not found."""

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @abstractmethod
    def search_by_text(
        self,
        text: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Encode *text* and return the *top_k* nearest neighbours."""

    @abstractmethod
    def search_by_image(
        self,
        image: Image.Image,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Encode *image* and return the *top_k* nearest neighbours."""

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    @abstractmethod
    def count(self) -> int:
        """Return the total number of vectors stored."""

    @abstractmethod
    def drop(self) -> None:
        """Drop and recreate the collection (destructive)."""
