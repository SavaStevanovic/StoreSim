"""FastAPI backend for the StoreSim product browser — PostgreSQL edition."""

from __future__ import annotations

import math
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        k, _, v = _line.partition("=")
        if k.strip() and v.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip()

DB_URL = os.environ.get("DATABASE_URL", "postgresql://storesim:storesim@localhost:5432/storesim")
STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="StoreSim Product Browser", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
@contextmanager
def _db() -> Generator[psycopg2.extensions.cursor, None, None]:
    conn = psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# CTE that joins products with ratings, images and category name.
_LATEST = """
SELECT
  p.id,
  p.title,
  p.description,
  p.price,
  p.date_first_available,
  p.main_category_id,
  p.created_at,
  r.value   AS average_rating,
  r.count   AS rating_count,
  i.path    AS image,
  c.name    AS main_category
FROM products p
LEFT JOIN ratings    r ON r.product_id = p.id
LEFT JOIN images     i ON i.product_id = p.id
LEFT JOIN categories c ON c.id = p.main_category_id
"""


def _row_to_card(r: dict[str, Any]) -> dict[str, Any]:
    avg = r["average_rating"]
    cnt = r["rating_count"]
    return {
        "id": r["id"],
        "title": r["title"],
        "image": r["image"],
        "price": float(r["price"]) if r["price"] is not None else None,
        "average_rating": round(float(avg), 1) if avg is not None else None,
        "rating_count": int(cnt) if cnt is not None else None,
        "main_category": r["main_category"],
    }


def _row_to_detail(r: dict[str, Any]) -> dict[str, Any]:
    d = _row_to_card(r)
    d.update(
        description=r["description"],
        main_category_id=r["main_category_id"],
        date_first_available=(
            r["date_first_available"].isoformat() if r["date_first_available"] else None
        ),
        created_at=r["created_at"].isoformat() if r["created_at"] else None,
    )
    return d


# ---------------------------------------------------------------------------
# Routes — categories
# ---------------------------------------------------------------------------
@app.get("/api/categories")
async def categories() -> JSONResponse:
    with _db() as cur:
        cur.execute(f"""
            SELECT main_category AS category, COUNT(*) AS count
            FROM ({_LATEST}) AS lp
            WHERE main_category IS NOT NULL
            GROUP BY main_category
            ORDER BY count DESC
        """)
        rows = cur.fetchall()
    return JSONResponse([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Routes — products list
# ---------------------------------------------------------------------------
@app.get("/api/products")
async def products(
    page: int = Query(1, ge=1),
    per_page: int = Query(24, ge=1, le=100),
    category: str = Query(""),
    search: str = Query(""),
    min_rating: float = Query(0.0, ge=0.0, le=5.0),
    max_price: float = Query(0.0, ge=0.0),
    sort: str = Query("rating_count"),
) -> JSONResponse:
    where_clauses: list[str] = []
    params: list[Any] = []

    if category:
        where_clauses.append("main_category = %s")
        params.append(category)
    if search:
        where_clauses.append("title ILIKE %s")
        params.append(f"%{search}%")
    if min_rating > 0:
        where_clauses.append("average_rating >= %s")
        params.append(min_rating)
    if max_price > 0:
        where_clauses.append("price <= %s")
        params.append(max_price)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sort_map = {
        "rating_count": "rating_count DESC NULLS LAST",
        "average_rating": "average_rating DESC NULLS LAST",
        "price": "price ASC NULLS LAST",
        "title": "title ASC",
    }
    order = sort_map.get(sort, "rating_count DESC NULLS LAST")

    base = f"FROM ({_LATEST}) AS lp {where_sql}"

    with _db() as cur:
        cur.execute(f"SELECT COUNT(*) AS n {base}", params)
        total: int = cur.fetchone()["n"]  # type: ignore[index]

        total_pages = max(1, math.ceil(total / per_page))
        page = min(page, total_pages)
        offset = (page - 1) * per_page

        cur.execute(
            f"SELECT * {base} ORDER BY {order} LIMIT %s OFFSET %s",
            [*params, per_page, offset],
        )
        rows = cur.fetchall()

    return JSONResponse(
        {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "results": [_row_to_card(dict(r)) for r in rows],
        }
    )


# ---------------------------------------------------------------------------
# Routes — single product
# ---------------------------------------------------------------------------
@app.get("/api/product/{product_id}")
async def product_detail(product_id: int) -> JSONResponse:
    with _db() as cur:
        cur.execute(f"SELECT * FROM ({_LATEST}) AS lp WHERE id = %s", [product_id])
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return JSONResponse(_row_to_detail(dict(row)))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def _run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    _run()
