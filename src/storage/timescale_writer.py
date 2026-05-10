"""TimescaleDB persistence layer.

Why TimescaleDB over plain Postgres:
  - Time-ordered queries on published_at are 10-100x faster on hypertables
  - Automatic chunk pruning / retention policies
  - Native compression on old chunks (up to 95% size reduction)
  - Continuous aggregates replace expensive GROUP BY on every dashboard load
  - Full SQL compatibility — works with SQLAlchemy, Alembic, Grafana

Schema:
  sentiment_results  — hypertable, chunked daily by published_at
  sentiment_hourly   — continuous aggregate (materialised view, refreshed automatically)
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from src.ingestion.schema import SentimentResult

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sentiment_results (
    id                   BIGSERIAL,
    published_at         TIMESTAMPTZ NOT NULL,
    processed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    article_hash         TEXT        NOT NULL,
    headline             TEXT        NOT NULL,
    url                  TEXT,
    source               TEXT        NOT NULL,
    ensemble_sentiment   TEXT        NOT NULL,
    ensemble_confidence  FLOAT       NOT NULL,
    ensemble_uncertainty FLOAT       NOT NULL,
    positive_prob        FLOAT       NOT NULL,
    negative_prob        FLOAT       NOT NULL,
    neutral_prob         FLOAT       NOT NULL,
    tickers              TEXT[],
    sectors              TEXT[],
    is_forward_looking   BOOLEAN     DEFAULT FALSE,
    has_negation         BOOLEAN     DEFAULT FALSE,
    hedge_score          FLOAT       DEFAULT 0.0,
    market_impact        TEXT        NOT NULL DEFAULT 'unknown',
    is_uncertain         BOOLEAN     DEFAULT FALSE,
    CONSTRAINT sentiment_results_pkey PRIMARY KEY (id, published_at)
);
"""

_CREATE_HYPERTABLE = """
SELECT create_hypertable(
    'sentiment_results', 'published_at',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_sr_hash     ON sentiment_results (article_hash);
CREATE INDEX IF NOT EXISTS idx_sr_source   ON sentiment_results (source, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_sr_sentiment ON sentiment_results (ensemble_sentiment, published_at DESC);
"""

_CREATE_CAGG = """
CREATE MATERIALIZED VIEW IF NOT EXISTS sentiment_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', published_at)  AS bucket,
    source,
    ensemble_sentiment,
    COUNT(*)                             AS article_count,
    AVG(ensemble_confidence)             AS avg_confidence,
    AVG(ensemble_uncertainty)            AS avg_uncertainty,
    AVG(positive_prob)                   AS avg_positive,
    AVG(negative_prob)                   AS avg_negative,
    AVG(neutral_prob)                    AS avg_neutral,
    AVG(hedge_score)                     AS avg_hedge
FROM sentiment_results
GROUP BY 1, 2, 3
WITH NO DATA;
"""

_INSERT = """
INSERT INTO sentiment_results (
    published_at, article_hash, headline, url, source,
    ensemble_sentiment, ensemble_confidence, ensemble_uncertainty,
    positive_prob, negative_prob, neutral_prob,
    tickers, sectors, is_forward_looking, has_negation,
    hedge_score, market_impact, is_uncertain
) VALUES (
    :published_at, :article_hash, :headline, :url, :source,
    :ensemble_sentiment, :ensemble_confidence, :ensemble_uncertainty,
    :positive_prob, :negative_prob, :neutral_prob,
    :tickers, :sectors, :is_forward_looking, :has_negation,
    :hedge_score, :market_impact, :is_uncertain
)
ON CONFLICT DO NOTHING;
"""


class TimescaleWriter:
    def __init__(self) -> None:
        self._engine = create_engine(
            settings.db.url,
            pool_size=settings.db.pool_size,
            max_overflow=settings.db.max_overflow,
            pool_pre_ping=True,
            echo=settings.db.echo,
        )
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        self._bootstrap()

    def _bootstrap(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(_CREATE_TABLE))
            try:
                conn.execute(text(_CREATE_HYPERTABLE))
            except Exception:
                pass  # already a hypertable
            conn.execute(text(_CREATE_INDEXES))
            try:
                conn.execute(text(_CREATE_CAGG))
            except Exception:
                pass  # already exists
        logger.info("timescaledb_schema_ready")

    def write_batch(self, results: list[SentimentResult]) -> None:
        if not results:
            return
        with self._session() as s:
            s.execute(text(_INSERT), [self._row(r) for r in results])
            s.commit()
        logger.debug("timescale_write count=%d", len(results))

    def write_one(self, result: SentimentResult) -> None:
        self.write_batch([result])

    def query_recent(self, hours: int = 24, limit: int = 500) -> list[dict[str, Any]]:
        sql = text(f"""
            SELECT * FROM sentiment_results
            WHERE published_at >= NOW() - INTERVAL '{hours} hours'
            ORDER BY published_at DESC
            LIMIT :limit
        """)
        with self._session() as s:
            return [dict(r) for r in s.execute(sql, {"limit": limit}).mappings()]

    def query_hourly_rollup(self, hours: int = 48) -> list[dict[str, Any]]:
        sql = text(f"""
            SELECT * FROM sentiment_hourly
            WHERE bucket >= NOW() - INTERVAL '{hours} hours'
            ORDER BY bucket DESC
        """)
        with self._session() as s:
            return [dict(r) for r in s.execute(sql).mappings()]

    def query_by_ticker(self, ticker: str, hours: int = 24) -> list[dict[str, Any]]:
        sql = text(f"""
            SELECT * FROM sentiment_results
            WHERE :ticker = ANY(tickers)
              AND published_at >= NOW() - INTERVAL '{hours} hours'
            ORDER BY published_at DESC
            LIMIT 200
        """)
        with self._session() as s:
            return [dict(r) for r in s.execute(sql, {"ticker": ticker}).mappings()]

    @contextmanager
    def _session(self) -> Generator[Session, None, None]:
        session: Session = self._Session()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _row(r: SentimentResult) -> dict[str, Any]:
        return {
            "published_at": r.published_at,
            "article_hash": r.article_hash,
            "headline": r.headline,
            "url": r.url,
            "source": r.source,
            "ensemble_sentiment": r.ensemble_sentiment.value,
            "ensemble_confidence": r.ensemble_confidence,
            "ensemble_uncertainty": r.ensemble_uncertainty,
            "positive_prob": r.positive_prob,
            "negative_prob": r.negative_prob,
            "neutral_prob": r.neutral_prob,
            "tickers": r.tickers,
            "sectors": r.sectors,
            "is_forward_looking": r.is_forward_looking,
            "has_negation": r.has_negation,
            "hedge_score": r.hedge_score,
            "market_impact": r.market_impact.value,
            "is_uncertain": r.is_uncertain,
        }
