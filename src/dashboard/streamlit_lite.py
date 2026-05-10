"""Streamlit dashboard — lite mode, reads directly from SQLite.
No Redis, no FastAPI, no Kafka required.
"""

import json
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = "pulsede_lite.db"
SENTIMENT_COLOURS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#6b7280",
}

st.set_page_config(
    page_title="PulseDE Lite — Financial Sentiment",
    page_icon="📡",
    layout="wide",
)


@st.cache_data(ttl=10)
def load_data():
    try:
        con = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT * FROM sentiment_results ORDER BY published_at DESC LIMIT 500", con
        )
        con.close()
        if "tickers" in df.columns:
            df["tickers"] = df["tickers"].apply(lambda x: json.loads(x) if x else [])
        return df
    except Exception:
        return pd.DataFrame()


st.title("📡 PulseDE v2.0 — Financial Sentiment Intelligence")
st.caption("🟡 Lite Mode — SQLite backend (Docker-free)")

df = load_data()

if df.empty:
    st.warning("No data yet. Run `python3 run_lite.py` first.")
    st.stop()

n = len(df)
pos = (df["ensemble_sentiment"] == "positive").sum()
neg = (df["ensemble_sentiment"] == "negative").sum()
avg_conf = df["ensemble_confidence"].mean()
avg_unc = df["ensemble_uncertainty"].mean() if "ensemble_uncertainty" in df.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Articles", f"{n:,}")
c2.metric("Positive", f"{pos / n:.0%}", delta=f"{pos / n - 0.5:.0%}")
c3.metric("Negative", f"{neg / n:.0%}")
c4.metric("Avg Confidence", f"{avg_conf:.1%}")
c5.metric("Avg Uncertainty", f"{avg_unc:.4f}")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Confidence by sentiment")
    # Use box plot for small datasets (violin degenerates with < 50 points per class)
    fig = px.box(
        df,
        y="ensemble_confidence",
        x="ensemble_sentiment",
        color="ensemble_sentiment",
        color_discrete_map=SENTIMENT_COLOURS,
        points="all",  # show individual article dots
        template="plotly_white",
        labels={"ensemble_confidence": "Confidence", "ensemble_sentiment": "Sentiment"},
    )
    fig.update_layout(height=320, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("Mix")
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
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

if "ensemble_uncertainty" in df.columns:
    st.subheader("Uncertainty vs confidence (MC Dropout entropy)")
    fig = px.scatter(
        df,
        x="ensemble_confidence",
        y="ensemble_uncertainty",
        color="ensemble_sentiment",
        color_discrete_map=SENTIMENT_COLOURS,
        opacity=0.7,
        template="plotly_white",
        hover_data=["headline"] if "headline" in df.columns else None,
    )
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

st.subheader("All articles")
show_cols = [
    c
    for c in [
        "headline",
        "source",
        "ensemble_sentiment",
        "ensemble_confidence",
        "ensemble_uncertainty",
        "market_impact",
        "tickers",
        "is_uncertain",
    ]
    if c in df.columns
]


def colour_row(row):
    c = SENTIMENT_COLOURS.get(str(row.get("ensemble_sentiment", "")), "")
    bg = f"background-color: {c}22" if c else ""
    return [bg] * len(row)


styled = (
    df[show_cols]
    .style.apply(colour_row, axis=1)
    .format({"ensemble_confidence": "{:.1%}", "ensemble_uncertainty": "{:.4f}"}, na_rep="-")
)
st.dataframe(styled, width="stretch", height=500)

if st.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
