"""Kafka consumer — orchestrates the full ML pipeline per batch.

Guarantees: at-least-once (offsets committed only after successful DB write).
Batching:   polls up to batch_size=16 messages before running inference.
            This amortises model forward-pass overhead across a full batch.
Shutdown:   SIGTERM/SIGINT → graceful drain → close consumer.
Metrics:    Prometheus counters + E2E latency histogram.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from datetime import UTC, datetime
from typing import Any

from confluent_kafka import Consumer, KafkaError, Message
from prometheus_client import Counter, Histogram

from config.settings import settings
from src.ingestion.schema import RawArticle, SentimentResult
from src.ml.ensemble import SentimentEnsemble
from src.ml.feature_engineering import FinancialFeatureExtractor
from src.storage.redis_cache import RedisCache
from src.storage.timescale_writer import TimescaleWriter

logger = logging.getLogger(__name__)

CONSUMED = Counter("pulsede_consumer_messages_total", "Messages consumed", ["status"])
PROCESSED = Counter("pulsede_processed_articles_total", "Articles successfully processed")
E2E_LATENCY = Histogram(
    "pulsede_e2e_latency_seconds",
    "End-to-end latency kafka→DB",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)


class SentimentConsumer:
    """Stateful consumer — models loaded once, reused across all batches."""

    def __init__(
        self,
        ensemble: SentimentEnsemble,
        feature_extractor: FinancialFeatureExtractor,
        db_writer: TimescaleWriter,
        cache: RedisCache,
        batch_size: int = 16,
        poll_timeout_s: float = 1.0,
    ) -> None:
        self._ensemble = ensemble
        self._features = feature_extractor
        self._db = db_writer
        self._cache = cache
        self._batch_size = batch_size
        self._poll_timeout = poll_timeout_s
        self._running = False

        conf: dict[str, Any] = {
            "bootstrap.servers": settings.kafka.bootstrap_servers,
            "group.id": settings.kafka.consumer_group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,  # manual commit for at-least-once
            "max.poll.interval.ms": 300_000,
            "session.timeout.ms": 30_000,
        }
        self._consumer = Consumer(conf)
        self._consumer.subscribe([settings.kafka.news_topic])
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        signal.signal(signal.SIGINT, lambda *_: self.stop())

    def run(self) -> None:
        self._running = True
        logger.info("consumer_started topic=%s", settings.kafka.news_topic)
        try:
            while self._running:
                batch = self._poll_batch()
                if batch:
                    self._process_batch(batch)
        finally:
            self._consumer.close()
            logger.info("consumer_stopped")

    def stop(self) -> None:
        self._running = False

    def _poll_batch(self) -> list[tuple[Message, RawArticle]]:
        batch: list[tuple[Message, RawArticle]] = []
        deadline = time.monotonic() + self._poll_timeout
        while len(batch) < self._batch_size and time.monotonic() < deadline:
            msg = self._consumer.poll(timeout=0.05)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error("kafka_error %s", msg.error())
                    CONSUMED.labels(status="error").inc()
                continue
            try:
                payload = json.loads(msg.value())
                batch.append((msg, self._deserialise(payload)))
                CONSUMED.labels(status="ok").inc()
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("deserialise_error %s", exc)
                CONSUMED.labels(status="deserialise_error").inc()
        return batch

    def _process_batch(self, batch: list[tuple[Message, RawArticle]]) -> None:
        msgs, articles = zip(*batch)
        texts = [(a.headline, a.body) for a in articles]
        headlines = [a.headline for a in articles]

        t0 = time.perf_counter()
        feature_vectors = self._features.extract_batch(texts)
        ensemble_outputs = self._ensemble.predict(headlines)
        elapsed = time.perf_counter() - t0
        E2E_LATENCY.observe(elapsed)

        results: list[SentimentResult] = []
        for article, fv, eo in zip(articles, feature_vectors, ensemble_outputs):
            results.append(
                SentimentResult(
                    article_hash=article.content_hash,
                    headline=article.headline,
                    url=article.url,
                    source=article.source,
                    published_at=article.published_at,
                    processed_at=datetime.now(UTC),
                    ensemble_sentiment=eo["sentiment"],
                    ensemble_confidence=eo["confidence"],
                    ensemble_uncertainty=eo["uncertainty"],
                    positive_prob=eo["positive_prob"],
                    negative_prob=eo["negative_prob"],
                    neutral_prob=eo["neutral_prob"],
                    model_predictions=eo["model_predictions"],
                    entities=fv.entities,
                    tickers=fv.tickers,
                    sectors=fv.sectors,
                    is_forward_looking=fv.is_forward_looking,
                    has_negation=fv.has_negation,
                    hedge_score=fv.hedge_score,
                    market_impact=fv.market_impact,
                    is_uncertain=eo["is_uncertain"],
                )
            )

        try:
            self._db.write_batch(results)
            self._cache.cache_results(results)
            PROCESSED.inc(len(results))
        except Exception as exc:
            logger.error("db_write_failed %s — will not commit offsets", exc)
            return  # retry on next consumer restart

        for msg in msgs:
            self._consumer.commit(message=msg, asynchronous=False)

        logger.info(
            "batch_done count=%d elapsed=%.2fs arts/s=%.1f",
            len(results),
            elapsed,
            len(results) / elapsed,
        )

    @staticmethod
    def _deserialise(p: dict[str, Any]) -> RawArticle:
        return RawArticle(
            source=p["source"],
            headline=p["headline"],
            url=p["url"],
            published_at=datetime.fromisoformat(p["published_at"]),
            body=p.get("body", ""),
            author=p.get("author", ""),
        )
