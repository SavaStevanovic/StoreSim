import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = Path(
    os.getenv("STORESIM_DATA_DIR", ROOT / "data" / "processed" / "amazon_products_2023")
)
PARQUET = DATA_DIR / "products.parquet"
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR = Path(__file__).parent / "static"
