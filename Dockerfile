# syntax=docker/dockerfile:1.5

# Builder stage: compile and install Python deps once, then copy into runtime.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build tools are only needed here for native extensions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential gcc g++ nlohmann-json3-dev \
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

# Compile C++ telemetry parser
WORKDIR /build/cpp
COPY src/cpp/telemetry_parser.cpp .
RUN g++ -O2 -std=c++17 -I/usr/include -o telemetry_parser telemetry_parser.cpp \
    && strip telemetry_parser

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

WORKDIR /app

# Ensure bin directory exists
RUN mkdir -p /app/bin

COPY --from=builder /install /usr/local
COPY --chown=1000:1000 . /app

# Copy compiled binary
COPY --from=builder --chown=1000:1000 /build/cpp/telemetry_parser /app/bin/telemetry_parser

# Create a non-root user and change ownership
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501
