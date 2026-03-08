"""CLIP ResNet50 model wrapper (backed by open-clip-torch)."""

from __future__ import annotations

import typing
from typing import Any, cast

import open_clip
import torch
from PIL import Image

from storesim.embedding_model.base import Embedder


class CLIPResNet50Model(Embedder):
    MODEL_NAME = "RN50"
    DEFAULT_PRETRAINED = "openai"

    def __init__(
        self,
        device: str | None = None,
        pretrained: str = DEFAULT_PRETRAINED,
        model_name: str = MODEL_NAME,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pretrained = pretrained
        self._device = device
        self._model_name = model_name
        self._model, _, self.preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained, device=self._device
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self._model_name)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_image(
        self,
        image: Image.Image,
    ) -> list[float]:
        image_input = self.preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            raw = self._model.encode_image(image_input)

        features: torch.Tensor = cast(torch.Tensor, raw).squeeze(0).float()
        features = torch.nn.functional.normalize(features, dim=-1)
        features = features.cpu().numpy().tolist()

        return typing.cast(list[float], features)

    def encode_text(
        self,
        text: str,
    ) -> list[float]:
        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            raw = self._model.encode_text(tokens)

        features: torch.Tensor = cast(torch.Tensor, raw).float()
        features = torch.nn.functional.normalize(features, dim=-1)
        features = features.cpu().numpy().tolist()

        return typing.cast(list[float], features)

    @property
    def device(self) -> str:
        """Return the device this model runs on."""
        return self._device

    @property
    def model(self) -> Any:
        """Return the underlying open-clip model."""
        return self._model

    @property
    def tokenizer(self) -> Any:
        """Return the tokenizer callable."""
        return self._tokenizer

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the output embeddings (1024 for RN50)."""
        return int(self._model.visual.output_dim)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self._model_name}', "
            f"pretrained='{self._pretrained}', "
            f")"
        )
