"""Kafka producer — exactly-once delivery, snappy compression, DLQ on failure.

Design: acks=all + enable.idempotence prevents duplicates within the broker.
        Tenacity handles transient errors with exponential backoff.
        Dead-letter queue receives messages that exhaust all retries.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from confluent_kafka import KafkaError, KafkaException, Producer
from prometheus_client import Counter, Histogram
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from src.ingestion.schema import RawArticle

logger = logging.getLogger(__name__)

PRODUCE_TOTAL = Counter(
    "pulsede_kafka_produce_total", "Kafka produce attempts", ["topic", "status"]
)
PRODUCE_LATENCY = Histogram(
    "pulsede_kafka_produce_latency_seconds", "Kafka produce latency", ["topic"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


class ArticleProducer:
    """Thread-safe Kafka producer for RawArticle messages."""

    def __init__(self) -> None:
        conf: dict[str, Any] = {
            "bootstrap.servers": settings.kafka.bootstrap_servers,
            "acks": settings.kafka.acks,
            "retries": settings.kafka.retries,
            "linger.ms": settings.kafka.linger_ms,
            "batch.size": settings.kafka.batch_size,
            "compression.type": settings.kafka.compression_type,
            "enable.idempotence": "true",
            "delivery.timeout.ms": 30_000,
            "request.timeout.ms": 10_000,
            "socket.keepalive.enable": "true",
        }
        self._producer = Producer(conf)
        self._topic = settings.kafka.news_topic
        self._dlq = settings.kafka.dead_letter_topic

    def publish_batch(self, articles: list[RawArticle]) -> None:
        for article in articles:
            self._publish_one(article)
        self._producer.flush(timeout=30)
        logger.info("batch_published count=%d topic=%s", len(articles), self._topic)

    @retry(
        retry=retry_if_exception_type(KafkaException),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        stop=stop_after_attempt(5),
        reraise=False,
    )
    def _publish_one(self, article: RawArticle) -> None:
        payload = json.dumps(article.to_dict()).encode()
        with PRODUCE_LATENCY.labels(topic=self._topic).time():
            try:
                self._producer.produce(
                    topic=self._topic,
                    key=article.content_hash.encode(),
                    value=payload,
                    on_delivery=self._delivery_callback,
                )
                self._producer.poll(0)
                PRODUCE_TOTAL.labels(topic=self._topic, status="enqueued").inc()
            except KafkaException as exc:
                logger.error("produce_error err=%s headline=%.80s", exc, article.headline)
                PRODUCE_TOTAL.labels(topic=self._topic, status="error").inc()
                self._send_to_dlq(article, str(exc))
                raise

    def _send_to_dlq(self, article: RawArticle, error: str) -> None:
        try:
            payload = json.dumps({**article.to_dict(), "_dlq_reason": error}).encode()
            self._producer.produce(
                topic=self._dlq,
                key=article.content_hash.encode(),
                value=payload,
            )
        except KafkaException:
            logger.critical("dlq_produce_failed hash=%s", article.content_hash)

    @staticmethod
    def _delivery_callback(err: KafkaError | None, msg: Any) -> None:
        if err:
            logger.warning("delivery_failed err=%s topic=%s", err, msg.topic())
        else:
            logger.debug("delivery_ok topic=%s partition=%d offset=%d",
                         msg.topic(), msg.partition(), msg.offset())

    def close(self) -> None:
        self._producer.flush(timeout=30)

    def __enter__(self) -> "ArticleProducer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
