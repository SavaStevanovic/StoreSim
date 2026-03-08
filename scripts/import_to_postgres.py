"""Import Amazon Products parquet into PostgreSQL using the postgress.sql schema.

Supported tables   : categories, products, product_categories, ratings, images
Skipped (not in schema): asin, store, features, details, embeddings

Usage:
    uv run python scripts/import_to_postgres.py
    DATABASE_URL=postgresql://... uv run python scripts/import_to_postgres.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

ROOT = Path(__file__).parent.parent
PARQUET = ROOT / "data" / "processed" / "amazon_products_2023" / "products.parquet"

# Load DATABASE_URL from .env if not already set
_env_file = ROOT / ".env"
if _env_file.exists() and "DATABASE_URL" not in os.environ:
    for _line in _env_file.read_text().splitlines():
        if _line.startswith("DATABASE_URL="):
            os.environ["DATABASE_URL"] = _line.split("=", 1)[1].strip()

DB_URL = os.environ.get("DATABASE_URL", "postgresql://storesim:storesim@localhost:5432/storesim")
BATCH = 500


def main() -> None:
    print(f"Connecting to {DB_URL.split('@')[-1]} ...")
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    cur = conn.cursor()

    # -----------------------------------------------------------------------
    # Load parquet
    # -----------------------------------------------------------------------
    print(f"Loading {PARQUET.name} ...")
    df = pd.read_parquet(PARQUET)
    df = df.where(df.notna(), None)
    print(f"  {len(df):,} rows")

    # -----------------------------------------------------------------------
    # 1. Categories  (main_category + per-product categories list)
    # -----------------------------------------------------------------------
    print("Inserting categories ...")
    all_cats: set[str] = set()
    for mc in df["main_category"].dropna():
        all_cats.add(str(mc))
    for cats_list in df["categories"]:
        if cats_list is not None and len(cats_list) > 0:
            for c in cats_list:
                all_cats.add(str(c))

    cur.executemany(
        "INSERT INTO categories (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
        [(c,) for c in sorted(all_cats)],
    )
    conn.commit()

    cur.execute("SELECT id, name FROM categories")
    cat_map: dict[str, int] = {name: cid for cid, name in cur.fetchall()}
    print(f"  {len(cat_map)} categories")

    # -----------------------------------------------------------------------
    # 2. Products + images + product_categories + ratings  (batched)
    # -----------------------------------------------------------------------
    print(f"Inserting {len(df):,} products in batches of {BATCH} ...")
    total = 0
    t0 = time.time()

    for batch_start in range(0, len(df), BATCH):
        batch = df.iloc[batch_start : batch_start + BATCH]

        # -- products --
        product_rows = []
        for row in batch.itertuples(index=False):
            mc_id = cat_map.get(str(row.main_category)) if row.main_category else None
            price = float(row.price) if pd.notna(row.price) else None
            dfa = row.date_first_available.date() if pd.notna(row.date_first_available) else None
            product_rows.append((row.title, row.description, price, dfa, mc_id))

        inserted: list[tuple[int]] = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO products (title, description, price, \
                date_first_available, main_category_id)
               VALUES %s RETURNING id""",
            product_rows,
            fetch=True,
        )
        product_ids = [r[0] for r in inserted]

        # -- images --
        image_rows = []
        for pid, row in zip(product_ids, batch.itertuples(index=False), strict=False):
            if row.image:
                image_rows.append((pid, str(row.image)))
        if image_rows:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO images (product_id, path) VALUES %s",
                image_rows,
            )

        # -- product_categories (many sub-categories per product) --
        pc_rows = []
        for pid, row in zip(product_ids, batch.itertuples(index=False), strict=False):
            if row.categories is not None and len(row.categories) > 0:
                for c in row.categories:
                    cid = cat_map.get(str(c))
                    if cid:
                        pc_rows.append((pid, cid))
        if pc_rows:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO product_categories (product_id, categorie_id) VALUES %s \
                    ON CONFLICT DO NOTHING",
                pc_rows,
            )

        # -- ratings  (one aggregate row per product: value=avg_rating, count=rating_count) --
        rating_rows = []
        for pid, row in zip(product_ids, batch.itertuples(index=False), strict=False):
            avg = float(row.average_rating) if pd.notna(row.average_rating) else None
            cnt = int(row.rating_number) if pd.notna(row.rating_number) else None
            if avg is not None:
                rating_rows.append((pid, avg, cnt))
        if rating_rows:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO ratings (product_id, value, count) VALUES %s",
                rating_rows,
            )

        conn.commit()
        total += len(product_ids)
        elapsed = time.time() - t0
        print(f"  {total:>7,} / {len(df):,}  ({total / elapsed:.0f} rows/s)", end="\r")

    print(f"\n  Done — {total:,} products in {time.time() - t0:.1f}s")

    cur.close()
    conn.close()

    print("\nSummary:")
    print(f"  categories       : {len(cat_map):>7,}")
    print(f"  products         : {total:>7,}")
    print(f"  ratings (aggreg.): up to {total:>7,}  (avg_rating + rating_number per product)")
    print(f"  images           : up to {total:>7,}  (one URL per product)")
    print()
    print("Skipped (not in schema): asin, store, features, details, embeddings")
    print("Import complete.")


if __name__ == "__main__":
    main()
