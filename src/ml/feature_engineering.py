"""Financial NLP feature engineering.

Extracted per-article:
  entities         — NER (persons, orgs, locations) via dslim/bert-base-NER
  tickers          — regex over all-caps tokens, filtered against blocklist
  sectors          — GICS sector from ticker → sector lookup table
  hedge_score      — fraction of hedging words in text ∈ [0, 1]
  has_negation     — boolean: negation word present
  is_forward_looking — boolean: FLS regex match
  market_impact    — high / medium / low / unknown (keyword heuristic + ZSC fallback)
"""
from __future__ import annotations

import re
from typing import NamedTuple

from transformers import pipeline as hf_pipeline

from config.settings import settings
from src.ingestion.schema import EntityMention, MarketImpact

_HEDGE_WORDS = frozenset([
    "may", "might", "could", "would", "should", "possibly", "potentially",
    "likely", "unlikely", "uncertain", "expects", "anticipates", "believes",
    "estimates", "approximately", "about", "around", "roughly", "seems", "appears",
])

_NEGATION_WORDS = frozenset([
    "not", "no", "never", "neither", "nor", "hardly", "barely", "scarcely",
    "failed", "fails", "miss", "missed", "despite", "however", "but", "without",
])

_FLS_REGEX = re.compile(
    r"\b(will|shall|forecast|guidance|outlook|project(?:s|ed)|plan(?:s|ned)"
    r"|expect(?:s|ed)|next\s+(?:quarter|year|month|fiscal)|fiscal\s+20\d{2}"
    r"|full[\s-]year)\b",
    re.IGNORECASE,
)

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

_TICKER_TO_SECTOR: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "INTC": "Technology", "TSM": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "WFC": "Financials", "V": "Financials", "MA": "Financials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "BTC": "Crypto", "ETH": "Crypto",
}

_TICKER_BLOCKLIST = frozenset([
    "I", "A", "AT", "BE", "BY", "FOR", "IT", "IN", "IS", "OR", "ON", "OF",
    "TO", "US", "ARE", "CEO", "CFO", "IPO", "ETF", "GDP", "FED", "SEC",
    "FTC", "FDA", "AI", "ML", "CAGR", "YOY", "QOQ", "EPS", "PE", "EV",
    "EBIT", "EBITDA", "Q1", "Q2", "Q3", "Q4",
])

_HIGH_IMPACT_KEYWORDS = [
    "crash", "collapse", "surge", "plunge", "recession", "fed rate",
    "bankruptcy", "default", "rate hike", "rate cut", "layoffs", "merger",
    "acquisition", "earnings beat", "earnings miss", "profit warning",
    "upgrade", "downgrade", "investigation", "fraud", "scandal", "sanction",
]

_MED_IMPACT_KEYWORDS = [
    "report", "quarterly", "revenue", "profit", "loss", "guidance",
    "analyst", "forecast", "dividend", "buyback",
]


class FeatureVector(NamedTuple):
    entities: list[EntityMention]
    tickers: list[str]
    sectors: list[str]
    is_forward_looking: bool
    has_negation: bool
    hedge_score: float
    market_impact: MarketImpact


class FinancialFeatureExtractor:
    """Extracts financial NLP features from headline + body text.

    NER and ZSC pipelines are lazy-loaded on first call to avoid
    loading 4 models at import time.
    """

    def __init__(self) -> None:
        self._ner: object | None = None
        self._zsc: object | None = None

    def extract(self, headline: str, body: str = "") -> FeatureVector:
        full = f"{headline} {body}".strip()
        words = full.lower().split()

        return FeatureVector(
            entities=self._extract_entities(headline),
            tickers=self._extract_tickers(full),
            sectors=self._extract_sectors(self._extract_tickers(full)),
            is_forward_looking=bool(_FLS_REGEX.search(full)),
            has_negation=any(w in _NEGATION_WORDS for w in words),
            hedge_score=self._hedge_score(words),
            market_impact=self._classify_market_impact(headline),
        )

    def extract_batch(self, texts: list[tuple[str, str]]) -> list[FeatureVector]:
        return [self.extract(h, b) for h, b in texts]

    def _get_ner(self) -> object:
        if self._ner is None:
            self._ner = hf_pipeline(
                "ner", model=settings.ml.ner_model,
                aggregation_strategy="simple", device=settings.ml.device,
            )
        return self._ner

    def _get_zsc(self) -> object:
        if self._zsc is None:
            self._zsc = hf_pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=settings.ml.device,
            )
        return self._zsc

    def _extract_entities(self, text: str) -> list[EntityMention]:
        if not text:
            return []
        try:
            raw = self._get_ner()(text)  # type: ignore[operator]
            return [
                EntityMention(
                    text=e["word"], entity_type=e["entity_group"],
                    confidence=round(float(e["score"]), 4),
                    start_char=e["start"], end_char=e["end"],
                )
                for e in raw if e.get("score", 0) > 0.80
            ]
        except Exception:
            return []

    def _extract_tickers(self, text: str) -> list[str]:
        candidates = _TICKER_RE.findall(text)
        return list(dict.fromkeys(
            t for t in candidates
            if t not in _TICKER_BLOCKLIST and len(t) >= 2
        ))

    def _extract_sectors(self, tickers: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for t in tickers:
            s = _TICKER_TO_SECTOR.get(t)
            if s:
                seen[s] = None
        return list(seen)

    def _hedge_score(self, words: list[str]) -> float:
        if not words:
            return 0.0
        return min(sum(1 for w in words if w in _HEDGE_WORDS) / len(words), 1.0)

    def _classify_market_impact(self, headline: str) -> MarketImpact:
        hl = headline.lower()
        if any(kw in hl for kw in _HIGH_IMPACT_KEYWORDS):
            return MarketImpact.HIGH
        if any(kw in hl for kw in _MED_IMPACT_KEYWORDS):
            return MarketImpact.MEDIUM
        try:
            zsc = self._get_zsc()
            result = zsc(  # type: ignore[operator]
                headline,
                candidate_labels=["high market impact", "medium market impact", "low market impact"],
                hypothesis_template="This news has {} on financial markets.",
            )
            label: str = result["labels"][0]
            if "high" in label:
                return MarketImpact.HIGH
            if "medium" in label:
                return MarketImpact.MEDIUM
            return MarketImpact.LOW
        except Exception:
            return MarketImpact.UNKNOWN
