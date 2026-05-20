# 1. Dependencies
FROM python:3.11-slim AS deps
WORKDIR /app

# tensorflow is ~620 MB — keep pip patient
ENV PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/api.txt requirements/api.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements/api.txt

# Install npec_cv from TestPyPI (transitive deps resolve via real PyPI)
ARG NPEC_CV_VERSION
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
"npec-cv${NPEC_CV_VERSION:+==$NPEC_CV_VERSION}"

# Download models from the matching GitHub release.
# Token comes in as a BuildKit secret so it never lands in a layer.
# Symlink the version-pinned cache dir to a stable path the runtime can rely on.
RUN --mount=type=secret,id=github_token,required=true \
    GITHUB_TOKEN="$(cat /run/secrets/github_token)" \
    NPEC_MODELS_CACHE_DIR=/opt/npec_models \
    npec-cv download-models \
 && ln -sfn "$(NPEC_MODELS_CACHE_DIR=/opt/npec_models \
        python -c 'from npec_cv.helpers.model_loader import get_models_dir; print(get_models_dir())')" \
        /opt/npec_models/current

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
COPY --from=deps /opt/npec_models /opt/npec_models

# Application source only — npec_cv is installed, no longer copied
COPY api/ ./api/

# Non-root user
RUN adduser --disabled-password --gecos "" --uid 1001 appuser \
    && mkdir -p /tmp/npec_masks /tmp/npec_annotation_masks \
    && chown -R appuser:appuser /tmp/npec_masks /tmp/npec_annotation_masks /opt/npec_models
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_MOCK_INFERENCE=false \
    NPEC_ARTIFACT_DIR=/opt/npec_models/current

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]