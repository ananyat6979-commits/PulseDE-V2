"""Streamlit dashboard — real-time financial sentiment intelligence.

Features:
  WebSocket background thread → live article feed via Redis pub/sub
  KPI cards: articles, positive %, negative %, avg confidence, avg uncertainty
  Plotly: hourly stacked bar, donut mix, violin by sentiment, uncertainty scatter
  Ticker heatmap (sentiment × article count, top 20 tickers)
  Styled live article table with colour-coded rows
  Sidebar: time window, confidence filter, sentiment filter, impact filter, ticker lookup
  30-second auto-refresh
"""

from __future__ import annotations

import ast
import json
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = "http://localhost:8080/v1"
WS_URL = "ws://localhost:8080/ws/realtime"
API_TOKEN = "dev-token"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

SENTIMENT_COLOURS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#6b7280",
}

st.set_page_config(
    page_title="PulseDE — Financial Sentiment Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "live_queue" not in st.session_state:
    st.session_state.live_queue: deque = deque(maxlen=200)
if "ws_started" not in st.session_state:
    st.session_state.ws_started = False


# ── WebSocket background listener ──────────────────────────────────────────────
def _start_ws_listener() -> None:
    """Push new results into session_state.live_queue from a daemon thread."""
    try:
        import websocket  # websocket-client

        def on_message(_ws: Any, message: str) -> None:
            try:
                st.session_state.live_queue.appendleft(json.loads(message))
            except Exception:
                pass

        def _run() -> None:
            while True:
                try:
                    ws = websocket.WebSocketApp(WS_URL, on_message=on_message)
                    ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception:
                    time.sleep(5)

        threading.Thread(target=_run, daemon=True).start()
    except ImportError:
        pass  # websocket-client not installed — live feed disabled


if not st.session_state.ws_started:
    _start_ws_listener()
    st.session_state.ws_started = True


# ── Data loaders (cached) ──────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_recent(n: int = 300) -> pd.DataFrame:
    try:
        r = requests.get(
            f"{API_BASE}/sentiment/latest", params={"n": n}, headers=HEADERS, timeout=5
        )
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_hourly(hours: int = 48) -> pd.DataFrame:
    try:
        r = requests.get(
            f"{API_BASE}/sentiment/hourly", params={"hours": hours}, headers=HEADERS, timeout=5
        )
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        return pd.DataFrame()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 PulseDE")
    st.caption("Real-Time Financial Sentiment Intelligence v2.0")
    st.divider()

    time_window = st.selectbox("Time window", ["1h", "6h", "24h", "48h", "7d"], index=2)
    hours = {"1h": 1, "6h": 6, "24h": 24, "48h": 48, "7d": 168}[time_window]

    min_confidence = st.slider("Min confidence", 0.0, 1.0, 0.5, 0.05)
    sentiment_filter = st.multiselect(
        "Sentiment",
        ["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"],
    )
    impact_filter = st.multiselect(
        "Market impact",
        ["high", "medium", "low", "unknown"],
        default=["high", "medium", "low", "unknown"],
    )
    show_uncertain = st.toggle("Show uncertain predictions", value=True)
    auto_refresh = st.toggle("Auto-refresh (30s)", value=True)

    st.divider()
    ticker_query = st.text_input("Ticker lookup", placeholder="AAPL")
    if ticker_query:
        try:
            r = requests.get(
                f"{API_BASE}/sentiment/ticker/{ticker_query.upper()}",
                headers=HEADERS,
                timeout=5,
            )
            if r.ok:
                t = r.json()
                c1, c2 = st.columns(2)
                c1.metric("Positive", f"{t['positive_pct'] * 100:.1f}%")
                c2.metric("Negative", f"{t['negative_pct'] * 100:.1f}%")
                st.metric("Articles", t["article_count"])
        except Exception:
            st.warning("Ticker not found or API offline")

    st.divider()
    st.caption("Model stack")
    st.code(
        "ProsusAI/finbert        (50%)\n"
        "yiyanghkust/finbert-tone (30%)\n"
        "distilroberta-financial  (20%)",
        language=None,
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📡 PulseDE — Financial Sentiment Intelligence")

try:
    health = requests.get(f"{API_BASE}/health", timeout=2).json()
    badge = "🟢 API Online" if health["status"] == "ok" else "🟡 API Degraded"
except Exception:
    badge = "🔴 API Offline"
st.caption(badge)

# ── Load + filter ──────────────────────────────────────────────────────────────
df_raw = load_recent(n=500)
df_hourly = load_hourly(hours=hours)

if df_raw.empty:
    st.warning("No data available — ensure the pipeline is running.")
    st.stop()

df = df_raw.copy()
df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
df["ensemble_confidence"] = df["ensemble_confidence"].astype(float)

cutoff = datetime.now(UTC) - timedelta(hours=hours)
df = df[df["published_at"] >= cutoff]
df = df[df["ensemble_confidence"] >= min_confidence]
df = df[df["ensemble_sentiment"].isin(sentiment_filter)]
if "market_impact" in df.columns:
    df = df[df["market_impact"].isin(impact_filter)]
if not show_uncertain and "is_uncertain" in df.columns:
    df = df[~df["is_uncertain"]]

# ── KPI cards ──────────────────────────────────────────────────────────────────
n = len(df)
pos_pct = (df["ensemble_sentiment"] == "positive").mean() * 100 if n else 0.0
neg_pct = (df["ensemble_sentiment"] == "negative").mean() * 100 if n else 0.0
avg_conf = df["ensemble_confidence"].mean() * 100 if n else 0.0
avg_unc = (
    df["ensemble_uncertainty"].mean() * 100 if n and "ensemble_uncertainty" in df.columns else 0.0
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Articles", f"{n:,}")
col2.metric("Positive", f"{pos_pct:.1f}%", delta=f"{pos_pct - 50:.1f}pp")
col3.metric("Negative", f"{neg_pct:.1f}%")
col4.metric("Avg confidence", f"{avg_conf:.1f}%")
col5.metric("Avg uncertainty", f"{avg_unc:.2f}%")

st.divider()

# ── Row 1: timeline + donut ────────────────────────────────────────────────────
r1c1, r1c2 = st.columns([2, 1])

with r1c1:
    st.subheader("Sentiment over time (hourly)")
    if not df_hourly.empty:
        dh = df_hourly.copy()
        dh["bucket"] = pd.to_datetime(dh["bucket"])
        pivot = dh.groupby(["bucket", "ensemble_sentiment"])["article_count"].sum().reset_index()
        fig = px.bar(
            pivot,
            x="bucket",
            y="article_count",
            color="ensemble_sentiment",
            color_discrete_map=SENTIMENT_COLOURS,
            barmode="stack",
            template="plotly_white",
            labels={"article_count": "Articles", "bucket": "", "ensemble_sentiment": "Sentiment"},
        )
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Hourly rollup accumulating — check back in ~1 hour.")

with r1c2:
    st.subheader("Sentiment mix")
    if n:
        counts = df["ensemble_sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        fig = px.pie(
            counts,
            values="count",
            names="sentiment",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLOURS,
            hole=0.55,
            template="plotly_white",
        )
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ── Row 2: violin + scatter ────────────────────────────────────────────────────
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.subheader("Confidence by sentiment")
    fig = px.violin(
        df,
        y="ensemble_confidence",
        x="ensemble_sentiment",
        color="ensemble_sentiment",
        color_discrete_map=SENTIMENT_COLOURS,
        box=True,
        points="outliers",
        template="plotly_white",
        labels={"ensemble_confidence": "Confidence", "ensemble_sentiment": "Sentiment"},
    )
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with r2c2:
    st.subheader("Uncertainty vs confidence")
    if "ensemble_uncertainty" in df.columns:
        sample = df.sample(min(len(df), 300))
        fig = px.scatter(
            sample,
            x="ensemble_confidence",
            y="ensemble_uncertainty",
            color="ensemble_sentiment",
            color_discrete_map=SENTIMENT_COLOURS,
            opacity=0.7,
            template="plotly_white",
            hover_data=["headline", "source"] if "headline" in df.columns else None,
            labels={
                "ensemble_confidence": "Confidence",
                "ensemble_uncertainty": "Uncertainty (MC dropout entropy)",
            },
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Uncertainty data not available.")

# ── Ticker heatmap ─────────────────────────────────────────────────────────────
st.subheader("Ticker sentiment heatmap (top 20)")
if "tickers" in df.columns and n:
    rows = []
    for _, row in df.iterrows():
        raw = row.get("tickers", [])
        tickers = ast.literal_eval(raw) if isinstance(raw, str) else (raw or [])
        for t in tickers:
            rows.append({"ticker": t, "sentiment": row["ensemble_sentiment"]})
    if rows:
        dt = pd.DataFrame(rows)
        pivot = dt.groupby(["ticker", "sentiment"]).size().unstack(fill_value=0)
        top = pivot.sum(axis=1).nlargest(20).index
        pivot = pivot.loc[top]
        fig = px.imshow(
            pivot,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            labels=dict(x="Sentiment", y="Ticker", color="Count"),
            template="plotly_white",
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No ticker data in this window.")

# ── Live feed table ────────────────────────────────────────────────────────────
st.subheader("Live article feed")

live = list(st.session_state.live_queue)
display_df = pd.DataFrame(live if live else df.head(50).to_dict("records"))

if not display_df.empty and "ensemble_sentiment" in display_df.columns:
    show_cols = [
        c
        for c in [
            "published_at",
            "headline",
            "source",
            "ensemble_sentiment",
            "ensemble_confidence",
            "ensemble_uncertainty",
            "market_impact",
            "tickers",
            "is_uncertain",
        ]
        if c in display_df.columns
    ]

    def _colour(row: pd.Series) -> list[str]:
        col = SENTIMENT_COLOURS.get(str(row.get("ensemble_sentiment", "")), "")
        bg = f"background-color: {col}22" if col else ""
        return [bg] * len(row)

    styled = (
        display_df[show_cols]
        .head(100)
        .style.apply(_colour, axis=1)
        .format(
            {
                "ensemble_confidence": "{:.1%}",
                "ensemble_uncertainty": "{:.3f}",
            },
            na_rep="-",
        )
    )
    st.dataframe(styled, use_container_width=True, height=420)

# ── Auto-refresh ───────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
