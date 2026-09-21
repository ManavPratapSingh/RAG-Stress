# Multi-stage Dockerfile for RAG-Stress (SPEC §13, §16, NFR-7)
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

# Install python dependencies first for layer caching
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install .

# Copy source code, config, data, and offline cassettes
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY cassettes/ ./cassettes/
COPY Makefile ./

# Reinstall editable package for non-root user
RUN pip install -e .

# Switch ownership to non-root user
RUN chown -R appuser:appgroup /app

USER appuser

# Default command: show CLI help
ENTRYPOINT ["ragstress"]
CMD ["--help"]

