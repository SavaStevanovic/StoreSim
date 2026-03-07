# StoreSim

Store product similarity search using CLIP (ResNet50) embeddings.

## Overview

This project uses OpenAI CLIP with a ResNet50 backbone to extract visual and text embeddings from product images and descriptions, enabling similarity-based product search.

## Setup

```bash
poetry install
```

## Usage

```python
from storesim import CLIPResNet50Model

model = CLIPResNet50Model()
image_embedding = model.encode_image("product.jpg")
text_embedding = model.encode_text("blue running shoes")
```

## Dataset

Amazon Products 2023 dataset tracked via DVC.

```bash
dvc pull
```

## Testing

```bash
poetry run pytest
```
