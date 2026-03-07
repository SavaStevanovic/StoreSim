"""Extract the Amazon Products 2023 Arrow dataset to Parquet + numpy embeddings.

Outputs (written to ``data/processed/amazon_products_2023/``):
  products.parquet   - all metadata columns (no embeddings)
  embeddings.npy     - float32 matrix of shape (N, 1536)
  asin_index.json    - mapping { row_index -> parent_asin } for cross-referencing

Usage:
    python3 scripts/extract_amazon_dataset.py
    python3 scripts/extract_amazon_dataset.py --input data/raw/amazon_products_2023 \\
                                               --output data/processed/amazon_products_2023 \\
                                               --split train
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# Columns that go into the Parquet file (everything except the embedding vector)
METADATA_COLS = [
    "parent_asin",
    "title",
    "description",
    "main_category",
    "categories",
    "store",
    "price",
    "average_rating",
    "rating_number",
    "features",
    "details",
    "image",
    "filename",
    "date_first_available",
    "__index_level_0__",
]

EMBEDDING_COL = "embeddings"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Amazon Products 2023 dataset.")
    p.add_argument(
        "--input",
        default=str(ROOT / "data" / "raw" / "amazon_products_2023"),
        help="Path to the HuggingFace dataset saved with save_to_disk().",
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "data" / "processed" / "amazon_products_2023"),
        help="Output directory for Parquet + npy files.",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split to extract (default: train).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    log.info("Loading dataset from %s …", input_path)
    ds_dict = load_from_disk(str(input_path))
    split = ds_dict[args.split]
    log.info("  Split '%s': %d rows, columns: %s", args.split, len(split), split.column_names)

    # ------------------------------------------------------------------
    # 2. Embeddings → float32 numpy array
    # ------------------------------------------------------------------
    log.info("Extracting embeddings …")
    emb_matrix = np.array(split[EMBEDDING_COL], dtype=np.float32)
    log.info("  Shape: %s  dtype: %s", emb_matrix.shape, emb_matrix.dtype)

    emb_path = output_path / "embeddings.npy"
    np.save(str(emb_path), emb_matrix)
    log.info("  Saved → %s  (%.1f MB)", emb_path, emb_path.stat().st_size / 1e6)

    # ------------------------------------------------------------------
    # 3. Metadata → Parquet
    # ------------------------------------------------------------------
    log.info("Writing metadata to Parquet …")
    present_cols = [c for c in METADATA_COLS if c in split.column_names]
    meta_ds = split.select_columns(present_cols)

    parquet_path = output_path / "products.parquet"
    meta_ds.to_parquet(str(parquet_path))
    log.info("  Saved → %s  (%.1f MB)", parquet_path, parquet_path.stat().st_size / 1e6)

    # ------------------------------------------------------------------
    # 4. ASIN index → JSON  (row_index → parent_asin)
    # ------------------------------------------------------------------
    log.info("Building ASIN index …")
    asin_index = {str(i): asin for i, asin in enumerate(split["parent_asin"])}
    index_path = output_path / "asin_index.json"
    index_path.write_text(json.dumps(asin_index, ensure_ascii=False, indent=0))
    log.info("  Saved → %s", index_path)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    log.info("Done. Output files:")
    for f in sorted(output_path.iterdir()):
        log.info("  %-40s  %8.1f MB", f.name, f.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
