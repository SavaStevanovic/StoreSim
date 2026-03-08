# ── Build stage: install dependencies ────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Tell uv exactly where to put the venv
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Copy dependency manifests only (better layer caching)
COPY pyproject.toml uv.lock ./

# Install runtime deps into /app/.venv — exclude dev group
RUN uv sync --frozen --no-group dev --no-install-project

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY app/      ./app/
COPY storesim/ ./storesim/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
