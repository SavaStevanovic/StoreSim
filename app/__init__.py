import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = Path(
    os.getenv("STORESIM_DATA_DIR", ROOT / "data" / "processed" / "amazon_products_2023")
)
PARQUET = DATA_DIR / "products.parquet"
STATIC_DIR = Path(__file__).parent / "static"

# Load .env from the project root (no-op when vars are already set, e.g. in Docker)
_env = ROOT / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _k, _, _v = _line.partition("=")
        if _k.strip() and _v.strip() and _k.strip() not in os.environ:
            os.environ[_k.strip()] = _v.strip()

DB_URL: str = os.environ["DATABASE_URL"]
