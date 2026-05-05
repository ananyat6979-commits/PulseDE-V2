"""FastAPI REST + WebSocket API.

GET  /v1/sentiment/latest           paginated recent results (Redis → DB fallback)
GET  /v1/sentiment/ticker/{ticker}  ticker-level aggregated summary
GET  /v1/sentiment/hourly           hourly rollup for charting
POST /v1/sentiment/analyse          synchronous on-demand inference
GET  /v1/health                     liveness + readiness probe
WS   /ws/realtime                   live stream via Redis pub/sub

Auth:       Bearer JWT (HS256)
Rate limit: Redis sliding window (60 req/min per user)
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import uvicorn
from fastapi import (
    Depends, FastAPI, HTTPException, Query, Request,
    WebSocket, WebSocketDisconnect, status,
)
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from config.settings import settings
from src.ml.ensemble import SentimentEnsemble
from src.storage.redis_cache import RedisCache
from src.storage.timescale_writer import TimescaleWriter

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PulseDE API",
    description="Real-Time Financial Sentiment Intelligence",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (loaded at startup) ────────────────────────────────────────────
_ensemble: SentimentEnsemble | None = None
_db: TimescaleWriter | None = None
_cache: RedisCache | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _ensemble, _db, _cache
    _ensemble = SentimentEnsemble()
    _db = TimescaleWriter()
    _cache = RedisCache()
    logger.info("api_startup_complete")


def get_db() -> TimescaleWriter:
    assert _db is not None; return _db

def get_cache() -> RedisCache:
    assert _cache is not None; return _cache

def get_ensemble() -> SentimentEnsemble:
    assert _ensemble is not None; return _ensemble


# ── Auth ───────────────────────────────────────────────────────────────────────
def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.api.access_token_expire_minutes
    )
    return jwt.encode(
        {"sub": subject, "exp": expire},
        settings.api.secret_key.get_secret_value(),
        algorithm=settings.api.algorithm,
    )


async def get_current_user(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    try:
        payload = jwt.decode(
            auth.removeprefix("Bearer ").strip(),
            settings.api.secret_key.get_secret_value(),
            algorithms=[settings.api.algorithm],
        )
        return str(payload["sub"])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def check_rate_limit(
    request: Request,
    cache: RedisCache = Depends(get_cache),
    client_id: str = Depends(get_current_user),
) -> None:
    if cache.is_rate_limited(client_id, limit=settings.api.rate_limit_per_minute):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Rate limit exceeded — max 60 req/min")


AuthDep = Annotated[str, Depends(get_current_user)]
RateDep = Annotated[None, Depends(check_rate_limit)]


# ── Request/response models ────────────────────────────────────────────────────
class AnalyseRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)


class HealthResponse(BaseModel):
    status: str
    db_ok: bool
    cache_ok: bool
    model_loaded: bool
    version: str = "2.0.0"


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/v1/health", response_model=HealthResponse, tags=["ops"])
async def health(
    db: TimescaleWriter = Depends(get_db),
    cache: RedisCache = Depends(get_cache),
) -> HealthResponse:
    db_ok, cache_ok = True, True
    try:
        db.query_recent(hours=0, limit=1)
    except Exception:
        db_ok = False
    try:
        cache.get_latest(n=1)
    except Exception:
        cache_ok = False
    return HealthResponse(
        status="ok" if db_ok and cache_ok else "degraded",
        db_ok=db_ok, cache_ok=cache_ok, model_loaded=_ensemble is not None,
    )


@app.get("/v1/sentiment/latest", tags=["sentiment"])
async def get_latest(
    _: RateDep,
    n: int = Query(default=50, ge=1, le=500),
    cache: RedisCache = Depends(get_cache),
    db: TimescaleWriter = Depends(get_db),
) -> list[dict[str, Any]]:
    results = cache.get_latest(n=n)
    return results if results else db.query_recent(hours=24, limit=n)


@app.get("/v1/sentiment/ticker/{ticker}", tags=["sentiment"])
async def get_ticker(
    ticker: str,
    _: RateDep,
    hours: int = Query(default=24, ge=1, le=168),
    cache: RedisCache = Depends(get_cache),
    db: TimescaleWriter = Depends(get_db),
) -> dict[str, Any]:
    cached = cache.get_ticker_summary(ticker.upper())
    if cached:
        return {**cached, "source": "cache"}
    rows = db.query_by_ticker(ticker.upper(), hours=hours)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for {ticker.upper()}")
    n = len(rows)
    pos = sum(1 for r in rows if r["ensemble_sentiment"] == "positive")
    neg = sum(1 for r in rows if r["ensemble_sentiment"] == "negative")
    return {
        "ticker": ticker.upper(), "hours": hours, "article_count": n,
        "positive_pct": round(pos / n, 4), "negative_pct": round(neg / n, 4),
        "neutral_pct": round((n - pos - neg) / n, 4),
        "avg_confidence": round(sum(r["ensemble_confidence"] for r in rows) / n, 4),
        "source": "db",
    }


@app.get("/v1/sentiment/hourly", tags=["sentiment"])
async def get_hourly(
    _: RateDep,
    hours: int = Query(default=48, ge=1, le=720),
    db: TimescaleWriter = Depends(get_db),
) -> list[dict[str, Any]]:
    return db.query_hourly_rollup(hours=hours)


@app.post("/v1/sentiment/analyse", tags=["inference"])
async def analyse(
    body: AnalyseRequest,
    _: RateDep,
    ensemble: SentimentEnsemble = Depends(get_ensemble),
) -> dict[str, Any]:
    results = ensemble.predict(body.texts)
    return {
        "results": [
            {
                "text": text,
                "sentiment": r["sentiment"].value,
                "confidence": r["confidence"],
                "uncertainty": r["uncertainty"],
                "is_uncertain": r["is_uncertain"],
                "positive_prob": r["positive_prob"],
                "negative_prob": r["negative_prob"],
                "neutral_prob": r["neutral_prob"],
            }
            for text, r in zip(body.texts, results)
        ],
        "model_count": 3,
        "temperature": ensemble._temperature,
    }


# ── WebSocket ──────────────────────────────────────────────────────────────────
_active_ws: set[WebSocket] = set()


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket) -> None:
    await websocket.accept()
    _active_ws.add(websocket)
    cache = get_cache()
    ps = cache.get_pubsub()
    try:
        while True:
            msg = ps.get_message(ignore_subscribe_messages=True, timeout=0.05)
            if msg and msg.get("data"):
                await websocket.send_text(msg["data"])
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        _active_ws.discard(websocket)
    finally:
        ps.close()


if __name__ == "__main__":
    uvicorn.run(
        "src.serving.api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.env == "development",
        workers=1 if settings.env == "development" else 4,
    )
