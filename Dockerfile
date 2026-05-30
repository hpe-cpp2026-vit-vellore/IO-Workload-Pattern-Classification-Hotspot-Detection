# syntax=docker/dockerfile:1.5

# Builder stage: compile and install Python deps once, then copy into runtime.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build tools are only needed here for native extensions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Install everything except torch first, then pull torch CPU-only wheel explicitly.
# The torch index is used only for the torch install line to avoid GPU wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v '^torch' requirements.txt > /tmp/requirements-notorch.txt \
    && pip install --no-cache-dir --prefix=/install -r /tmp/requirements-notorch.txt \
    && pip install --no-cache-dir --prefix=/install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch

# Runtime stage: slim image with only runtime deps and app code.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# curl is used for container healthchecks in docker-compose.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security.
RUN useradd -r -m -u 10001 appuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 8000 8501
