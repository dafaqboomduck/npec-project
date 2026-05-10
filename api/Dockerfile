# syntax=docker/dockerfile:1.4
# 1. Dependencies
FROM python:3.11-slim AS deps

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/api.txt requirements/api.txt
COPY requirements/cv.txt requirements/cv.txt

RUN pip install --no-cache-dir \
    -r requirements/api.txt \
    -r requirements/cv.txt

COPY npec_cv/ ./npec_cv/
RUN pip install --no-cache-dir ./npec_cv

# Download models using BuildKit secret
RUN --mount=type=secret,id=github_token \
    GITHUB_TOKEN="$(cat /run/secrets/github_token)" \
    NPEC_MODELS_CACHE_DIR=/app/.cache/npec_cv/models \
    npec-cv download-models

# 2. Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps /usr/local/lib/python3.11/site-packages \
    /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY --from=deps /app/.cache/npec_cv /home/appuser/.cache/npec_cv
ENV NPEC_ARTIFACT_DIR=/home/appuser/.cache/npec_cv/models/v0.2.0

# Application source
COPY db/ ./db/
COPY npec_cv/ ./npec_cv/
COPY api/ ./api/

# Non-root user
RUN adduser --disabled-password --gecos "" --uid 1001 appuser \
    && mkdir -p /tmp/npec_masks /tmp/npec_annotation_masks \
    && chown -R appuser:appuser /tmp/npec_masks /tmp/npec_annotation_masks /home/appuser/.cache

USER appuser

# Runtime configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_MOCK_INFERENCE=false \
    USE_DB_STORAGE=true \
    PYTHONPATH=/app

EXPOSE 8000

# Liveness check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"
    
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
