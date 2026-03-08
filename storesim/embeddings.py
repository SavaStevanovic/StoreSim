"""Batch embedding extraction utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from storesim.embedding_model.model import CLIPResNet50Model


class EmbeddingExtractor:
    """Extracts and caches CLIP embeddings in batch.

    Args:
        model: An initialised :class:`CLIPResNet50Model` instance.  A new one
               is created automatically when *None* is passed.
        batch_size: Number of samples processed per forward pass.
    """

    def __init__(
        self,
        model: CLIPResNet50Model | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model = model or CLIPResNet50Model()
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Image batch
    # ------------------------------------------------------------------

    def extract_from_images(
        self,
        images: Iterable[str | Path | Image.Image],
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Extract embeddings from an iterable of images.

        Args:
            images: Iterable of file paths or PIL Images.
            show_progress: Display a tqdm progress bar.

        Returns:
            Tensor of shape ``(N, embedding_dim)``.
        """
        images = list(images)
        batches = [images[i : i + self.batch_size] for i in range(0, len(images), self.batch_size)]

        all_embeddings: list[torch.Tensor] = []
        iterator = tqdm(batches, desc="Images") if show_progress else batches
        for batch in iterator:
            tensors = []
            for img in batch:
                if not isinstance(img, Image.Image):
                    img = Image.open(img).convert("RGB")
                tensors.append(self.model.preprocess(img))
            batch_tensor = torch.stack(tensors).to(self.model.device)
            with torch.no_grad():
                embs = self.model.model.encode_image(batch_tensor)
            embs = torch.nn.functional.normalize(embs.float(), dim=-1)
            all_embeddings.append(embs.cpu())

        return torch.cat(all_embeddings, dim=0)

    # ------------------------------------------------------------------
    # Text batch
    # ------------------------------------------------------------------

    def extract_from_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Extract embeddings from a list of text strings.

        Args:
            texts: List of text descriptions.
            show_progress: Display a tqdm progress bar.

        Returns:
            Tensor of shape ``(N, embedding_dim)``.
        """
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        all_embeddings: list[torch.Tensor] = []
        iterator = tqdm(batches, desc="Texts") if show_progress else batches
        for batch in iterator:
            tokens = self.model.tokenizer(batch).to(self.model.device)
            with torch.no_grad():
                embs = self.model.model.encode_text(tokens)
            embs = torch.nn.functional.normalize(embs.float(), dim=-1)
            all_embeddings.append(embs.cpu())

        return torch.cat(all_embeddings, dim=0)
