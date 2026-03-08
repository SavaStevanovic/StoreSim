"""Milvus (local-file) vector store backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image
from pymilvus import DataType, MilvusClient

from storesim.embedding_model.base import Embedder
from storesim.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class MilvusStore(VectorStore):
    def __init__(
        self,
        path: str | Path,
        model: Embedder,
    ) -> None:
        self._path = str(path)
        self._model = model
        self._collection_name = str(self._model)
        self._dim = model.embedding_dim
        self._text_embedding_name = "text_vector"
        self._image_embedding_name = "image_vector"
        self._client = MilvusClient(self._path)
        """Create the collection + indexes if they do not yet exist."""
        if not self._client.has_collection(self._collection_name):
            schema = self._client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            schema.add_field(
                field_name="product_id",
                datatype=DataType.INT64,
                is_primary=True,
                index_type="AUTOINDEX",
            )
            schema.add_field(
                field_name=self._image_embedding_name,
                datatype=DataType.FLOAT_VECTOR,
                dim=self._dim,
                index_type="AUTOINDEX",
            )
            schema.add_field(
                field_name=self._text_embedding_name,
                datatype=DataType.FLOAT_VECTOR,
                dim=self._dim,
                index_type="AUTOINDEX",
            )

            index_params = self._client.prepare_index_params()
            for vec_field in (self._image_embedding_name, self._text_embedding_name):
                index_params.add_index(
                    field_name=vec_field,
                )

            self._client.create_collection(
                collection_name=self._collection_name,
                schema=schema,
                index_params=index_params,
            )
        logger.info("Created Milvus collection '%s'", self._collection_name)

    def _search(
        self,
        vector: list[float],
        field: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        results = self._client.search(
            collection_name=self._collection_name,
            data=[vector],
            anns_field=field,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["product_id"],
        )
        hits: list[dict[str, Any]] = []
        for hit in results[0]:
            hits.append(
                {
                    "product_id": hit["entity"]["product_id"],
                    "score": float(hit["distance"]),
                }
            )
        return hits

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def insert(self, records: list[dict[str, Any]]) -> None:
        self._client.insert(collection_name=self._collection_name, data=records)

    def get_vector(self, id: int) -> dict[str, Any] | None:
        rows = self._client.get(
            collection_name=self._collection_name,
            ids=[id],
            output_fields=["product_id", self._image_embedding_name, self._text_embedding_name],
        )
        return rows[0] if rows else None

    def search_by_text(
        self,
        text: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        vec = self._model.encode_text(text)
        return self._search(vec, self._text_embedding_name, top_k)

    def search_by_image(
        self,
        image: Image.Image,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        vec = self._model.encode_image(image)
        return self._search(vec, self._image_embedding_name, top_k)

    def count(self) -> int:
        """Return the number of vectors currently stored."""
        stats = self._client.get_collection_stats(self._collection_name)
        return int(stats["row_count"])

    def drop(self) -> None:
        """Drop the collection (destructive)."""
        self._client.drop_collection(self._collection_name)
