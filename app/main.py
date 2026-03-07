"""FastAPI backend for the StoreSim product browser."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = Path(
    os.getenv("STORESIM_DATA_DIR", ROOT / "data" / "processed" / "amazon_products_2023")
)
PARQUET = DATA_DIR / "products.parquet"
STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# App + data bootstrap
# ---------------------------------------------------------------------------
app = FastAPI(title="StoreSim Product Browser", version="0.1.0")

_df: pd.DataFrame | None = None


def _get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        if not PARQUET.exists():
            raise RuntimeError(f"Parquet not found: {PARQUET}")
        raw = pd.read_parquet(PARQUET)
        # Normalise list-type columns so they JSON-serialise cleanly
        for col in ("features", "categories"):
            if col in raw.columns:
                raw[col] = raw[col].apply(lambda v: list(v) if v is not None else [])
        # Coerce NaN strings to None for clean JSON
        for col in ("title", "description", "store", "image", "details", "main_category"):
            if col in raw.columns:
                raw[col] = raw[col].where(raw[col].notna(), None)
        _df = raw
    return _df


@app.on_event("startup")
async def _startup() -> None:
    _get_df()  # warm the cache


# ---------------------------------------------------------------------------
# Static files + root HTML
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# API - categories
# ---------------------------------------------------------------------------
@app.get("/api/categories")
async def categories() -> JSONResponse:
    df = _get_df()
    counts = (
        df["main_category"]
        .dropna()
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
        .to_dict(orient="records")
    )
    return JSONResponse(counts)


# ---------------------------------------------------------------------------
# API - products (paginated, filtered, sorted)
# ---------------------------------------------------------------------------
@app.get("/api/products")
async def products(
    page: int = Query(1, ge=1),
    per_page: int = Query(24, ge=1, le=100),
    category: str = Query(""),
    search: str = Query(""),
    min_rating: float = Query(0.0, ge=0.0, le=5.0),
    max_price: float = Query(0.0, ge=0.0),
    sort: str = Query("rating_number"),
) -> JSONResponse:
    df = _get_df().copy()

    # -- filters --
    if category:
        df = df[df["main_category"] == category]
    if search:
        q = search.lower()
        mask = df["title"].str.lower().str.contains(q, na=False)
        df = df[mask]
    if min_rating > 0:
        df = df[df["average_rating"] >= min_rating]
    if max_price > 0:
        df = df[df["price"] <= max_price]

    # -- sort --
    valid_sorts = {"rating_number", "average_rating", "price", "title"}
    if sort in valid_sorts:
        ascending = sort == "title"
        df = df.sort_values(sort, ascending=ascending, na_position="last")

    total = len(df)
    total_pages = max(1, math.ceil(total / per_page))
    page = min(page, total_pages)

    start = (page - 1) * per_page
    page_df = df.iloc[start : start + per_page]

    records: list[dict[str, Any]] = []
    for row in page_df.itertuples(index=False):
        records.append(
            {
                "asin": row.parent_asin,
                "title": row.title,
                "image": row.image,
                "price": None
                if (isinstance(row.price, float) and math.isnan(row.price))
                else row.price,
                "average_rating": (
                    None
                    if (isinstance(row.average_rating, float) and math.isnan(row.average_rating))
                    else round(float(row.average_rating), 1)
                ),
                "rating_number": (
                    None
                    if (isinstance(row.rating_number, float) and math.isnan(row.rating_number))
                    else int(row.rating_number)
                ),
                "main_category": row.main_category,
                "store": row.store,
            }
        )

    return JSONResponse(
        {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "results": records,
        }
    )


# ---------------------------------------------------------------------------
# API - single product detail
# ---------------------------------------------------------------------------
@app.get("/api/product/{asin}")
async def product_detail(asin: str) -> JSONResponse:
    df = _get_df()
    rows = df[df["parent_asin"] == asin]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    row = rows.iloc[0]

    def _safe(v: Any) -> Any:
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    return JSONResponse(
        {
            "asin": row["parent_asin"],
            "title": row["title"],
            "image": row["image"],
            "description": row["description"],
            "features": list(row["features"]) if row["features"] is not None else [],
            "categories": list(row["categories"]) if row["categories"] is not None else [],
            "main_category": row["main_category"],
            "store": row["store"],
            "price": _safe(row["price"]),
            "average_rating": _safe(row["average_rating"]),
            "rating_number": None
            if _safe(row["rating_number"]) is None
            else int(row["rating_number"]),
            "details": row["details"],
            "date_first_available": str(row["date_first_available"])
            if row["date_first_available"] is not None
            else None,
        }
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def _run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    _run()
