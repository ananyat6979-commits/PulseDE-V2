"""Redis caching layer.

Responsibilities:
  Cache latest N results as a sorted set scored by published_at timestamp
  Per-ticker hash aggregates with 5-minute TTL
  Pub/Sub channel for WebSocket real-time push
  Sliding-window rate limiter for the FastAPI layer
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

import redis as redis_lib

from config.settings import settings
from src.ingestion.schema import SentimentResult

logger = logging.getLogger(__name__)

_LATEST_KEY = "pulsede:latest"
_TICKER_KEY_TPL = "pulsede:ticker:{ticker}"
_RATE_KEY_TPL = "pulsede:rate:{client_id}"
_PUBSUB_CHANNEL = "pulsede:realtime"


class RedisCache:
    def __init__(self) -> None:
        self._r = redis_lib.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            password=(
                settings.redis.password.get_secret_value() if settings.redis.password else None
            ),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

    def cache_results(self, results: list[SentimentResult]) -> None:
        pipe = self._r.pipeline()
        ttl = settings.redis.ttl_seconds
        for r in results:
            value = json.dumps(r.to_dict())
            pipe.zadd(_LATEST_KEY, {value: r.published_at.timestamp()})
            pipe.publish(_PUBSUB_CHANNEL, value)
        pipe.zremrangebyrank(_LATEST_KEY, 0, -1001)  # keep latest 1000
        pipe.expire(_LATEST_KEY, ttl)
        pipe.execute()
        self._update_ticker_aggregates(results)

    def get_latest(self, n: int = 100) -> list[dict[str, Any]]:
        raw = self._r.zrevrange(_LATEST_KEY, 0, n - 1)
        return [json.loads(v) for v in raw]

    def get_ticker_summary(self, ticker: str) -> dict[str, Any] | None:
        key = _TICKER_KEY_TPL.format(ticker=ticker)
        data = self._r.hgetall(key)
        if not data:
            return None
        return {
            "ticker": ticker,
            "positive_pct": float(data.get("positive_pct", 0)),
            "negative_pct": float(data.get("negative_pct", 0)),
            "neutral_pct": float(data.get("neutral_pct", 0)),
            "article_count": int(data.get("article_count", 0)),
            "avg_confidence": float(data.get("avg_confidence", 0)),
            "avg_uncertainty": float(data.get("avg_uncertainty", 0)),
        }

    def _update_ticker_aggregates(self, results: list[SentimentResult]) -> None:
        ticker_data: dict[str, list[SentimentResult]] = defaultdict(list)
        for r in results:
            for t in r.tickers:
                ticker_data[t].append(r)

        pipe = self._r.pipeline()
        for ticker, items in ticker_data.items():
            key = _TICKER_KEY_TPL.format(ticker=ticker)
            n = len(items)
            pos = sum(1 for r in items if r.ensemble_sentiment.value == "positive")
            neg = sum(1 for r in items if r.ensemble_sentiment.value == "negative")
            existing = int(self._r.hget(key, "article_count") or 0)
            pipe.hset(
                key,
                mapping={
                    "positive_pct": round(pos / n, 4),
                    "negative_pct": round(neg / n, 4),
                    "neutral_pct": round((n - pos - neg) / n, 4),
                    "article_count": existing + n,
                    "avg_confidence": round(sum(r.ensemble_confidence for r in items) / n, 4),
                    "avg_uncertainty": round(sum(r.ensemble_uncertainty for r in items) / n, 4),
                },
            )
            pipe.expire(key, 300)
        pipe.execute()

    def is_rate_limited(self, client_id: str, limit: int = 60, window_s: int = 60) -> bool:
        key = _RATE_KEY_TPL.format(client_id=client_id)
        pipe = self._r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_s)
        count, _ = pipe.execute()
        return int(count) > limit

    def get_pubsub(self) -> redis_lib.client.PubSub:
        ps = self._r.pubsub()
        ps.subscribe(_PUBSUB_CHANNEL)
        return ps

    def close(self) -> None:
        self._r.close()
