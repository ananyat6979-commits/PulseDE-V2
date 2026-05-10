"""Multi-source financial news ingestion with two-stage deduplication.

Sources: NewsAPI · RSS (FT/WSJ/Bloomberg/Yahoo) · Alpha Vantage (optional)
Dedup:   SHA-256 exact hash in Redis (24h) + SimHash Hamming < 4 for near-duplicates

Redis is used for persistent cross-session deduplication but is NOT required.
If Redis is unavailable, deduplication falls back to an in-process hash set.
This means dedup works correctly within a single run but does not persist
across restarts. Behaviour is logged clearly so the operator knows which
mode is active.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import feedparser
import httpx
import redis as redis_lib
from simhash import Simhash

from config.settings import settings
from src.ingestion.schema import RawArticle

logger = logging.getLogger(__name__)

_SIMHASH_NS = "pulsede:simhash:"
_HASH_NS = "pulsede:hash:"


@dataclass
class FetchStats:
    source: str
    fetched: int = 0
    deduplicated: int = 0
    errors: int = 0

    @property
    def yield_rate(self) -> float:
        if self.fetched == 0:
            return 0.0
        return (self.fetched - self.deduplicated) / self.fetched


class NewsFetcher:
    """Fetches and deduplicates financial news from multiple sources.

    Deduplication is two-stage:
      1. SHA-256 exact hash stored in Redis (24h TTL)
      2. SimHash Hamming distance < 4 for near-duplicates (reworded headlines)

    If Redis is unavailable at startup, both stages fall back to an
    in-process set. This is logged as a WARNING so it is always visible.
    The system continues to function correctly within a single run.
    """

    def __init__(self) -> None:
        self._http = httpx.Client(
            timeout=10.0,
            headers={"User-Agent": "PulseDE/2.0"},
        )

        # ── Redis connection (optional) ────────────────────────────────────────
        self._redis: redis_lib.Redis | None = None
        self._seen_hashes: set[str] = set()  # fallback exact dedup
        self._seen_simhashes: list[int] = []  # fallback near-dedup

        try:
            r = redis_lib.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=(
                    settings.redis.password.get_secret_value() if settings.redis.password else None
                ),
                decode_responses=True,
                socket_connect_timeout=2,  # fail fast — don't block startup
                socket_timeout=2,
            )
            r.ping()
            self._redis = r
            logger.info("Redis connected — using persistent deduplication (24h window)")
        except Exception as exc:
            logger.warning(
                "Redis unavailable (%s) — falling back to in-memory deduplication. "
                "Dedup will not persist across restarts.",
                type(exc).__name__,
            )

        self._stats: list[FetchStats] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch_all(self) -> list[RawArticle]:
        """Fetch from all sources; return deduplicated articles sorted newest-first."""
        articles: list[RawArticle] = []
        articles.extend(self._fetch_newsapi())
        articles.extend(self._fetch_rss_feeds())
        if settings.news.alpha_vantage_key:
            articles.extend(self._fetch_alpha_vantage())

        unique = list(self._deduplicate(articles))
        unique.sort(key=lambda a: a.published_at, reverse=True)

        for stat in self._stats:
            logger.info(
                "fetch_stats source=%s fetched=%d deduped=%d yield=%.1f%%",
                stat.source,
                stat.fetched,
                stat.deduplicated,
                stat.yield_rate * 100,
            )
        return unique[: settings.news.max_articles_per_fetch]

    # ── Source adapters ────────────────────────────────────────────────────────

    def _fetch_newsapi(self) -> list[RawArticle]:
        stat = FetchStats(source="newsapi")
        self._stats.append(stat)
        articles: list[RawArticle] = []
        queries = ["stock market", "earnings", "Fed rates", "inflation", "S&P 500"]
        api_key = settings.news.news_api_key.get_secret_value()

        for query in queries:
            try:
                resp = self._http.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 20,
                        "apiKey": api_key,
                    },
                )
                resp.raise_for_status()
                for item in resp.json().get("articles", []):
                    if not item.get("title") or item["title"] == "[Removed]":
                        continue
                    articles.append(
                        RawArticle(
                            source=f"newsapi:{item.get('source', {}).get('name', 'unknown')}",
                            headline=item["title"].strip(),
                            url=item.get("url", ""),
                            published_at=datetime.fromisoformat(
                                item["publishedAt"].replace("Z", "+00:00")
                            ),
                            body=item.get("content") or item.get("description") or "",
                            author=item.get("author") or "",
                        )
                    )
                    stat.fetched += 1
            except httpx.HTTPError as exc:
                logger.warning("newsapi_error query=%s err=%s", query, exc)
                stat.errors += 1
            time.sleep(0.1)
        return articles

    def _fetch_rss_feeds(self) -> list[RawArticle]:
        stat = FetchStats(source="rss")
        self._stats.append(stat)
        articles: list[RawArticle] = []

        for feed_url in settings.news.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                source_name = feed.feed.get("title", feed_url)
                for entry in feed.entries[:20]:
                    pub = entry.get("published_parsed") or entry.get("updated_parsed")
                    published_at = datetime(*pub[:6], tzinfo=UTC) if pub else datetime.now(UTC)
                    headline = (entry.get("title") or "").strip()
                    if not headline:
                        continue
                    articles.append(
                        RawArticle(
                            source=f"rss:{source_name}",
                            headline=headline,
                            url=entry.get("link") or "",
                            published_at=published_at,
                            body=entry.get("summary") or "",
                            author=entry.get("author") or "",
                        )
                    )
                    stat.fetched += 1
            except Exception as exc:
                logger.warning("rss_error feed=%s err=%s", feed_url, exc)
                stat.errors += 1
        return articles

    def _fetch_alpha_vantage(self) -> list[RawArticle]:
        stat = FetchStats(source="alpha_vantage")
        self._stats.append(stat)
        articles: list[RawArticle] = []
        try:
            resp = self._http.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "NEWS_SENTIMENT",
                    "topics": "financial_markets,earnings",
                    "limit": 50,
                    "apikey": settings.news.alpha_vantage_key.get_secret_value(),  # type: ignore[union-attr]
                },
            )
            resp.raise_for_status()
            for item in resp.json().get("feed", []):
                pub_str = item.get("time_published", "")
                try:
                    published_at = datetime.strptime(pub_str, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
                except ValueError:
                    published_at = datetime.now(UTC)
                headline = (item.get("title") or "").strip()
                if headline:
                    articles.append(
                        RawArticle(
                            source=f"av:{item.get('source', 'unknown')}",
                            headline=headline,
                            url=item.get("url") or "",
                            published_at=published_at,
                            body=item.get("summary") or "",
                            author=(item.get("authors") or [None])[0] or "",
                        )
                    )
                    stat.fetched += 1
        except Exception as exc:
            logger.warning("av_error err=%s", exc)
            stat.errors += 1
        return articles

    # ── Deduplication ──────────────────────────────────────────────────────────

    def _deduplicate(self, articles: list[RawArticle]) -> Iterator[RawArticle]:
        """Two-stage dedup: exact hash → SimHash near-duplicate.

        Uses Redis if available, falls back to in-memory sets if not.
        """
        ttl = settings.redis.dedup_window_seconds

        for article in articles:
            # ── Stage 1: exact hash ────────────────────────────────────────────
            if self._redis:
                exact_key = f"{_HASH_NS}{article.content_hash}"
                if self._redis.exists(exact_key):
                    continue
            else:
                if article.content_hash in self._seen_hashes:
                    continue

            # ── Stage 2: SimHash near-duplicate ────────────────────────────────
            sh_int = Simhash(article.headline).value
            if self._is_near_duplicate(sh_int):
                continue

            # ── Mark as seen ───────────────────────────────────────────────────
            if self._redis:
                pipe = self._redis.pipeline()
                pipe.setex(f"{_HASH_NS}{article.content_hash}", ttl, "1")
                pipe.setex(f"{_SIMHASH_NS}{sh_int}", ttl, str(sh_int))
                pipe.execute()
            else:
                self._seen_hashes.add(article.content_hash)
                self._seen_simhashes.append(sh_int)

            yield article

    def _is_near_duplicate(self, simhash_int: int) -> bool:
        if self._redis:
            for key in self._redis.scan_iter(f"{_SIMHASH_NS}*"):
                stored = self._redis.get(key)
                if stored and bin(simhash_int ^ int(stored)).count("1") < 4:
                    return True
        else:
            for stored_int in self._seen_simhashes:
                if bin(simhash_int ^ stored_int).count("1") < 4:
                    return True
        return False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._http.close()
        if self._redis:
            self._redis.close()

    def __enter__(self) -> NewsFetcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
