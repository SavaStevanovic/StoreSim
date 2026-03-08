"""Database access layer for the StoreSim FastAPI app.

Provides a connection pool, typed query functions, and serialisers.
Route handlers should call the public ``get_*`` functions — they never
need to touch cursors or SQL directly.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypedDict

import psycopg2
import psycopg2.extras
import psycopg2.pool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection pool  (initialised by ``init_pool`` at app startup)
# ---------------------------------------------------------------------------
_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def init_pool(dsn: str, *, min_conn: int = 2, max_conn: int = 10) -> None:
    """Create the module-level connection pool. Call once at startup."""
    global _pool
    if _pool is not None:
        return
    _pool = psycopg2.pool.ThreadedConnectionPool(min_conn, max_conn, dsn)
    logger.info("DB pool opened  (%d-%d connections)", min_conn, max_conn)


def close_pool() -> None:
    """Drain the pool. Call once at shutdown."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("DB pool closed")


@contextmanager
def _cursor() -> Generator[psycopg2.extensions.cursor, None, None]:
    """Yield a RealDictCursor from the pool; auto-commit / rollback."""
    if _pool is None:
        raise RuntimeError("Database pool not initialised — call init_pool() first")
    conn = _pool.getconn()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ---------------------------------------------------------------------------
# Base SQL fragments (private)
# ---------------------------------------------------------------------------
_PRODUCTS_JOIN = """
    SELECT
      p.id,
      p.title,
      p.description,
      p.price,
      p.date_first_available,
      p.main_category_id,
      p.created_at,
      r.value  AS average_rating,
      r.count  AS rating_count,
      i.path   AS image,
      c.name   AS main_category
    FROM products p
    LEFT JOIN ratings    r ON r.product_id = p.id
    LEFT JOIN images     i ON i.product_id = p.id
    LEFT JOIN categories c ON c.id = p.main_category_id
"""

_SORT_COLUMNS: dict[str, str] = {
    "rating_count": "rating_count DESC NULLS LAST",
    "average_rating": "average_rating DESC NULLS LAST",
    "price": "price ASC NULLS LAST",
    "title": "title ASC",
}

# ---------------------------------------------------------------------------
# Filter dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProductFilters:
    """Captures every user-supplied filter / pagination / sort parameter."""

    category: str = ""
    search: str = ""
    min_rating: float = 0.0
    max_price: float = 0.0
    sort: str = "rating_count"
    page: int = 1
    per_page: int = 24

    def _where(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.category:
            clauses.append("main_category = %s")
            params.append(self.category)
        if self.search:
            clauses.append("title ILIKE %s")
            params.append(f"%{self.search}%")
        if self.min_rating > 0:
            clauses.append("average_rating >= %s")
            params.append(self.min_rating)
        if self.max_price > 0:
            clauses.append("price <= %s")
            params.append(self.max_price)
        sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return sql, params


# ---------------------------------------------------------------------------
# Typed return dicts
# ---------------------------------------------------------------------------


class CategoryRow(TypedDict):
    category: str
    count: int


class ProductCard(TypedDict):
    id: int
    title: str
    image: str | None
    price: float | None
    average_rating: float | None
    rating_count: int | None
    main_category: str | None


class ProductDetail(ProductCard, total=False):
    description: str | None
    main_category_id: int | None
    date_first_available: str | None
    created_at: str | None


class ProductPage(TypedDict):
    total: int
    page: int
    per_page: int
    total_pages: int
    results: list[ProductCard]


# ---------------------------------------------------------------------------
# Row → dict serialisers (private)
# ---------------------------------------------------------------------------


def _to_card(r: dict[str, Any]) -> ProductCard:
    avg = r["average_rating"]
    cnt = r["rating_count"]
    return ProductCard(
        id=r["id"],
        title=r["title"],
        image=r["image"],
        price=float(r["price"]) if r["price"] is not None else None,
        average_rating=round(float(avg), 1) if avg is not None else None,
        rating_count=int(cnt) if cnt is not None else None,
        main_category=r["main_category"],
    )


def _to_detail(r: dict[str, Any]) -> ProductDetail:
    d: ProductDetail = {**_to_card(r)}  # type: ignore[typeddict-item]
    d["description"] = r["description"]
    d["main_category_id"] = r["main_category_id"]
    d["date_first_available"] = (
        r["date_first_available"].isoformat() if r["date_first_available"] else None
    )
    d["created_at"] = r["created_at"].isoformat() if r["created_at"] else None
    return d


# ---------------------------------------------------------------------------
# Public query functions  (the only things routes should call)
# ---------------------------------------------------------------------------


def get_categories() -> list[CategoryRow]:
    """All categories with their product count, sorted descending."""
    with _cursor() as cur:
        cur.execute(
            f"""
            SELECT main_category AS category, COUNT(*) AS count
            FROM ({_PRODUCTS_JOIN}) AS p
            WHERE main_category IS NOT NULL
            GROUP BY main_category
            ORDER BY count DESC
            """
        )
        return [CategoryRow(category=r["category"], count=r["count"]) for r in cur.fetchall()]


def get_products(filters: ProductFilters) -> ProductPage:
    """Paginated, filtered, sorted product listing."""
    where_sql, params = filters._where()
    order = _SORT_COLUMNS.get(filters.sort, _SORT_COLUMNS["rating_count"])
    base = f"FROM ({_PRODUCTS_JOIN}) AS p {where_sql}"

    with _cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS n {base}", params)
        total: int = cur.fetchone()["n"]  # type: ignore[index]

        total_pages = max(1, math.ceil(total / filters.per_page))
        page = min(filters.page, total_pages)
        offset = (page - 1) * filters.per_page

        cur.execute(
            f"SELECT * {base} ORDER BY {order} LIMIT %s OFFSET %s",
            [*params, filters.per_page, offset],
        )
        rows = cur.fetchall()

    return ProductPage(
        total=total,
        page=page,
        per_page=filters.per_page,
        total_pages=total_pages,
        results=[_to_card(dict(r)) for r in rows],
    )


def get_product(product_id: int) -> ProductDetail | None:
    """Full detail for one product, or ``None``."""
    with _cursor() as cur:
        cur.execute(
            f"SELECT * FROM ({_PRODUCTS_JOIN}) AS p WHERE p.id = %s",
            [product_id],
        )
        row = cur.fetchone()
    return _to_detail(dict(row)) if row else None
