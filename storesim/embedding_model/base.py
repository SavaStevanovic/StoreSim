"""CLIP ResNet50 model wrapper (backed by open-clip-torch)."""

from __future__ import annotations

from abc import abstractmethod

from PIL import Image


class Embedder:
    @abstractmethod
    def encode_image(
        self,
        image: Image.Image,
    ) -> list[float]:
        pass

    @abstractmethod
    def encode_text(
        self,
        text: str,
    ) -> list[float]:
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass
