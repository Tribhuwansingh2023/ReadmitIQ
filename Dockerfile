# Multi-stage production container for ReadmitIQ FastAPI Backend
FROM python:3.11-slim as backend

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    WORKERS=2 \
    READMITIQ_SERVE_MODE=full \
    CORS_ORIGINS="*"

WORKDIR /app

# Install system dependencies (build-essential for C extensions, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code, config, and models
COPY src/ ./src/
COPY configs/ ./configs/
COPY api/ ./api/
COPY models/final/ ./models/final/
COPY models/FROZEN_SELECTION.json ./models/
COPY reports/metrics/ ./reports/metrics/
COPY reports/demo/ ./reports/demo/

# Container Healthcheck verifying /health returns HTTP 200
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000

# Production uvicorn multi-worker entrypoint
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]
