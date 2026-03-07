"""CLIP ResNet50 model wrapper (backed by open-clip-torch)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import open_clip
import torch
from PIL import Image


class CLIPResNet50Model:
    """Wraps OpenCLIP with the RN50 (ResNet50) backbone.

    The model can encode both images and text into a shared embedding space,
    enabling cross-modal similarity search.

    Args:
        device: Target device ('cuda', 'cpu', or None for auto-detect).
        pretrained: Pre-trained weights tag accepted by open_clip, e.g.
                    ``'openai'`` (default) or ``'yfcc15m'``.
    """

    MODEL_NAME = "RN50"
    DEFAULT_PRETRAINED = "openai"

    def __init__(
        self,
        device: str | None = None,
        pretrained: str = DEFAULT_PRETRAINED,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_image(
        self,
        image: str | Path | Image.Image,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode a single image to a feature vector.

        Args:
            image: File path or a PIL Image object.
            normalize: If True, L2-normalise the output embedding.

        Returns:
            Float tensor of shape ``(embedding_dim,)`` on CPU.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw = self.model.encode_image(image_input)

        features: torch.Tensor = cast(torch.Tensor, raw).squeeze(0).float()
        if normalize:
            features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu()

    def encode_text(
        self,
        text: str | list[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode one or more text strings to feature vectors.

        Args:
            text: A single string or a list of strings.
            normalize: If True, L2-normalise the output embeddings.

        Returns:
            Float tensor of shape ``(embedding_dim,)`` for a single string or
            ``(N, embedding_dim)`` for a list, always on CPU.
        """
        single = isinstance(text, str)
        texts: list[str]
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            raw = self.model.encode_text(tokens)

        features: torch.Tensor = cast(torch.Tensor, raw).float()
        if normalize:
            features = torch.nn.functional.normalize(features, dim=-1)

        features = features.cpu()
        return features.squeeze(0) if single else features

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the output embeddings (1024 for RN50)."""
        return int(self.model.visual.output_dim)
