"""FastAPI backend for the StoreSim product browser — PostgreSQL edition."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import DB_URL, STATIC_DIR
from app.db import ProductFilters, close_pool, get_categories, get_product, get_products, init_pool


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    init_pool(DB_URL)
    yield
    close_pool()


app = FastAPI(title="StoreSim Product Browser", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Routes — categories
# ---------------------------------------------------------------------------
@app.get("/api/categories")
async def categories() -> JSONResponse:
    return JSONResponse(get_categories())


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
    filters = ProductFilters(
        category=category,
        search=search,
        min_rating=min_rating,
        max_price=max_price,
        sort=sort,
        page=page,
        per_page=per_page,
    )
    return JSONResponse(get_products(filters))


# ---------------------------------------------------------------------------
# Routes — single product
# ---------------------------------------------------------------------------
@app.get("/api/product/{product_id}")
async def product_detail(product_id: int) -> JSONResponse:
    detail = get_product(product_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return JSONResponse(detail)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
