# =============================================================================
# ModelWatch-GCP - FastAPI prediction service
# Lightweight image suitable for Cloud Run.
# =============================================================================
FROM python:3.11-slim

# Avoid interactive prompts and keep image small.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OS dependencies kept minimal; scikit-learn wheels do not require build deps
# on python:3.11-slim for x86_64/arm64.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system app \
    && adduser --system --ingroup app --home /app app

# Install Python dependencies first so the layer caches well.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code.
COPY src ./src
COPY api ./api
COPY data/raw ./data/raw
COPY models ./models
COPY reports ./reports

# Runtime-owned directories for local CSV logging and generated drift reports.
RUN mkdir -p data/logs data/processed reports \
    && chown -R app:app /app

USER app

# Cloud Run uses PORT (default 8080); we honor it with an env-driven default.
ENV PORT=8000 \
    API_RELOAD=false
EXPOSE 8000

# Run the API. Use ``sh -c`` so the PORT variable is expanded at runtime.
CMD ["sh", "-c", "uvicorn api.fastapi_app:app --host 0.0.0.0 --port ${PORT:-8000}"]
