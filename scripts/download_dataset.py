"""Download the AMAZON Products 2023 dataset from HuggingFace Hub.

Usage:
    python scripts/download_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `storesim` can be imported
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

DATASET_REPO = "milistu/AMAZON-Products-2023"
OUTPUT_DIR = ROOT / "data" / "raw" / "amazon_products_2023"


def main() -> None:
    print(f"Downloading '{DATASET_REPO}' …")
    dataset = load_dataset(DATASET_REPO)
    print(f"  Splits: {list(dataset.keys())}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUTPUT_DIR))
    print(f"  Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
