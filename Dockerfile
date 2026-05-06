# ── Stage 1: base — shared deps + HuggingFace model cache ────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so layer is cached when only source changes
COPY pyproject.toml .
RUN pip install -e "." --no-build-isolation

# Pre-download HuggingFace models so runtime starts in <5s.
# Layer is cached unless pyproject.toml changes.
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
models = [ \
    'ProsusAI/finbert', \
    'yiyanghkust/finbert-tone', \
    'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis', \
    'dslim/bert-base-NER', \
]; \
[AutoTokenizer.from_pretrained(m) for m in models]; \
[AutoModelForSequenceClassification.from_pretrained(m) for m in models]; \
print('Models cached')"

# Copy source last (changes most frequently)
COPY . .


# ── Stage 2: api — FastAPI + uvicorn ─────────────────────────────────────────
FROM base AS api

EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/v1/health || exit 1

CMD ["python", "-m", "src.serving.api"]


# ── Stage 3: consumer — Kafka consumer + ML inference worker ─────────────────
FROM base AS consumer

# Expose Prometheus metrics port
EXPOSE 8001

CMD ["python", "-c", "\
import logging; \
logging.basicConfig(level='INFO'); \
from src.ml.ensemble import SentimentEnsemble; \
from src.ml.feature_engineering import FinancialFeatureExtractor; \
from src.storage.timescale_writer import TimescaleWriter; \
from src.storage.redis_cache import RedisCache; \
from src.processing.kafka_consumer import SentimentConsumer; \
consumer = SentimentConsumer( \
    SentimentEnsemble(), \
    FinancialFeatureExtractor(), \
    TimescaleWriter(), \
    RedisCache(), \
); \
consumer.run()"]


# ── Stage 4: dashboard — Streamlit ───────────────────────────────────────────
FROM base AS dashboard

EXPOSE 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/dashboard/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
