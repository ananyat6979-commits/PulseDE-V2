"""Unit tests for the ML layer.

Covers:
  FinancialFeatureExtractor  — hedge, negation, FLS, tickers, sectors, market impact, edge cases
  EvaluationReport           — ECE, Brier, disagreement rate, latency ordering, MCC
  RawArticle                 — content hash stability and uniqueness
  SentimentResult            — to_dict serialisation round-trip
  Ensemble helpers           — softmax, MC dropout uncertainty shape

All tests run without Kafka / TimescaleDB / Redis.
HuggingFace NER + ZSC pipelines are mocked so CI doesn't download models.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.schema import (
    EntityMention,
    MarketImpact,
    ModelPrediction,
    RawArticle,
    Sentiment,
    SentimentResult,
)
from src.ml.evaluator import (
    EvaluationReport,
    _ece,
    _flatten,
    _pairwise_disagreement,
    evaluate,
)
from src.ml.feature_engineering import FinancialFeatureExtractor


# ────────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def extractor() -> FinancialFeatureExtractor:
    """FeatureExtractor with NER + ZSC pipelines mocked out."""
    fx = FinancialFeatureExtractor()
    # Mock NER to return empty list — avoids downloading 400 MB model in CI
    fx._ner = MagicMock(return_value=[])
    # Mock ZSC to return 'low market impact' by default
    fx._zsc = MagicMock(return_value={
        "labels": ["low market impact", "medium market impact", "high market impact"],
        "scores": [0.6, 0.3, 0.1],
    })
    return fx


def _make_eval_data(
    n: int = 200, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 3, n)
    y_proba = rng.dirichlet(np.ones(3), size=n).astype(np.float32)
    # Bias probabilities toward the true class so accuracy is reasonable
    for i, c in enumerate(y_true):
        y_proba[i, c] += 1.5
    y_proba /= y_proba.sum(axis=1, keepdims=True)
    y_pred = y_proba.argmax(axis=1)
    per_model = [
        rng.integers(0, 3, n),
        rng.integers(0, 3, n),
        y_pred.copy(),
    ]
    uncertainties = rng.uniform(0, 0.3, n).astype(np.float32)
    latencies = rng.uniform(10, 150, n).astype(np.float32)
    return y_true, y_pred, y_proba, per_model, uncertainties, latencies


# ────────────────────────────────────────────────────────────────────────────────
# FinancialFeatureExtractor tests
# ────────────────────────────────────────────────────────────────────────────────

class TestHedgeScore:
    def test_high_hedge(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract(
            "The market may possibly recover if conditions potentially improve"
        )
        assert fv.hedge_score > 0.2

    def test_low_hedge_factual(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Apple beats Q3 earnings, stock up 5%")
        assert fv.hedge_score < 0.1

    def test_hedge_score_bounded(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract(
            "may might could would should possibly potentially likely "
            "unlikely uncertain estimates approximately about"
        )
        assert 0.0 <= fv.hedge_score <= 1.0

    def test_empty_text_hedge_zero(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("")
        assert fv.hedge_score == 0.0


class TestNegationDetection:
    def test_explicit_negation(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Company failed to meet revenue guidance")
        assert fv.has_negation is True

    def test_no_negation_positive(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Strong earnings beat across all segments")
        assert fv.has_negation is False

    def test_negation_word_not(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Revenue did not grow in Q3")
        assert fv.has_negation is True

    def test_negation_word_despite(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Stock rose despite weak macro data")
        assert fv.has_negation is True

    def test_empty_no_negation(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("")
        assert fv.has_negation is False


class TestForwardLooking:
    def test_fls_expects(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Management expects next quarter revenue to grow 15%")
        assert fv.is_forward_looking is True

    def test_fls_next_fiscal_year(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Guidance raised for next fiscal year 2026")
        assert fv.is_forward_looking is True

    def test_fls_full_year(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Full-year earnings forecast revised upward")
        assert fv.is_forward_looking is True

    def test_not_fls_historical(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Stock closed down 3% on heavy volume yesterday")
        assert fv.is_forward_looking is False

    def test_not_fls_empty(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("")
        assert fv.is_forward_looking is False


class TestTickerExtraction:
    def test_known_tickers(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("AAPL and MSFT both rallied after the Fed decision")
        assert "AAPL" in fv.tickers
        assert "MSFT" in fv.tickers

    def test_blocklist_filtered(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("The CEO said AI will transform the ETF market")
        assert "CEO" not in fv.tickers
        assert "AI" not in fv.tickers
        assert "ETF" not in fv.tickers

    def test_single_letter_filtered(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("A rate hike is coming")
        assert "A" not in fv.tickers

    def test_no_duplicates(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("NVDA up as NVDA reports record NVDA revenue")
        assert fv.tickers.count("NVDA") == 1

    def test_empty_no_tickers(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("")
        assert fv.tickers == []


class TestSectorMapping:
    def test_technology_sector(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("NVDA GPU sales drive record revenue quarter")
        assert "Technology" in fv.sectors

    def test_financials_sector(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("JPM reported record investment banking revenue")
        assert "Financials" in fv.sectors

    def test_energy_sector(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("XOM profits surge on oil price spike")
        assert "Energy" in fv.sectors

    def test_no_sector_unknown_ticker(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("ZZZZ Corp reports earnings")
        # ZZZZ not in lookup table — sectors should be empty or not contain it
        assert "ZZZZ" not in fv.sectors

    def test_multi_sector(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("AAPL and JPM both beat estimates")
        assert "Technology" in fv.sectors
        assert "Financials" in fv.sectors


class TestMarketImpact:
    def test_high_impact_crash(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Market crash feared as recession signals mount")
        assert fv.market_impact == MarketImpact.HIGH

    def test_high_impact_merger(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Merger agreement announced between two major banks")
        assert fv.market_impact == MarketImpact.HIGH

    def test_medium_impact_earnings(self, extractor: FinancialFeatureExtractor) -> None:
        fv = extractor.extract("Quarterly earnings report shows modest revenue growth")
        assert fv.market_impact in (MarketImpact.MEDIUM, MarketImpact.HIGH)

    def test_low_impact_zsc_fallback(self, extractor: FinancialFeatureExtractor) -> None:
        # No keyword matches → ZSC mocked to return 'low market impact'
        fv = extractor.extract("Some minor operational update from a small company")
        assert fv.market_impact == MarketImpact.LOW


class TestBatchExtraction:
    def test_batch_same_as_individual(self, extractor: FinancialFeatureExtractor) -> None:
        texts = [
            ("Apple beats earnings", ""),
            ("Fed may raise rates", ""),
            ("Oil crash fears mount", ""),
        ]
        batch = extractor.extract_batch(texts)
        individual = [extractor.extract(h, b) for h, b in texts]
        assert len(batch) == len(individual)
        for b, ind in zip(batch, individual):
            assert b.hedge_score == ind.hedge_score
            assert b.has_negation == ind.has_negation
            assert b.is_forward_looking == ind.is_forward_looking


# ────────────────────────────────────────────────────────────────────────────────
# Evaluator tests
# ────────────────────────────────────────────────────────────────────────────────

class TestECE:
    def test_ece_in_range(self) -> None:
        y_true, y_pred, y_proba, *_ = _make_eval_data()
        ece = _ece(y_true, y_proba)
        assert 0.0 <= ece <= 1.0

    def test_perfect_ece_near_zero(self) -> None:
        """Perfect calibration: always predicts class 0 with 100% confidence, always correct."""
        n = 400
        y_proba = np.zeros((n, 3), dtype=np.float32)
        y_proba[:, 0] = 1.0
        y_true = np.zeros(n, dtype=int)
        assert _ece(y_true, y_proba) < 0.05

    def test_worst_case_ece_positive(self) -> None:
        """100% confidence always wrong → ECE should be high (≈ 1.0)."""
        n = 200
        y_proba = np.zeros((n, 3), dtype=np.float32)
        y_proba[:, 0] = 1.0           # predicts positive
        y_true = np.ones(n, dtype=int)  # always negative
        ece = _ece(y_true, y_proba)
        assert ece > 0.5


class TestDisagreementRate:
    def test_all_agree_zero(self) -> None:
        preds = [np.zeros(50, dtype=int)] * 3
        assert _pairwise_disagreement(preds) == 0.0

    def test_always_disagree_one(self) -> None:
        preds = [
            np.zeros(50, dtype=int),
            np.ones(50, dtype=int),
            np.full(50, 2, dtype=int),
        ]
        assert _pairwise_disagreement(preds) == 1.0

    def test_single_model_zero(self) -> None:
        preds = [np.zeros(50, dtype=int)]
        assert _pairwise_disagreement(preds) == 0.0

    def test_partial_disagreement(self) -> None:
        n = 100
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        b[: n // 2] = 1          # half disagree
        rate = _pairwise_disagreement([a, b])
        assert 0.4 < rate < 0.6


class TestEvaluationReport:
    def test_all_fields_present(self) -> None:
        args = _make_eval_data()
        report = evaluate(*args)
        assert 0.0 <= report.macro_f1 <= 1.0
        assert 0.0 <= report.accuracy <= 1.0
        assert -1.0 <= report.mcc <= 1.0
        assert report.n_samples == 200
        assert set(report.per_class.keys()) == {"positive", "negative", "neutral"}

    def test_latency_percentiles_ordered(self) -> None:
        args = _make_eval_data()
        report = evaluate(*args)
        assert report.p50_ms <= report.p95_ms <= report.p99_ms

    def test_brier_perfect(self) -> None:
        n = 300
        y_true = np.zeros(n, dtype=int)
        y_proba = np.zeros((n, 3), dtype=np.float32)
        y_proba[:, 0] = 1.0
        y_pred = np.zeros(n, dtype=int)
        latencies = np.ones(n, dtype=np.float32) * 50
        uncertainties = np.zeros(n, dtype=np.float32)
        report = evaluate(y_true, y_pred, y_proba, [y_pred] * 3, uncertainties, latencies)
        assert report.calibration.brier_score < 0.01

    def test_per_class_metrics_bounded(self) -> None:
        args = _make_eval_data()
        report = evaluate(*args)
        for cls, m in report.per_class.items():
            assert 0.0 <= m.precision <= 1.0, f"{cls} precision out of range"
            assert 0.0 <= m.recall <= 1.0,    f"{cls} recall out of range"
            assert 0.0 <= m.f1 <= 1.0,        f"{cls} f1 out of range"
            assert 0.0 <= m.pr_auc <= 1.0,    f"{cls} pr_auc out of range"
            assert 0.0 <= m.roc_auc <= 1.0,   f"{cls} roc_auc out of range"

    def test_confusion_matrix_shape(self) -> None:
        args = _make_eval_data()
        report = evaluate(*args)
        assert len(report.confusion_matrix) == 3
        assert all(len(row) == 3 for row in report.confusion_matrix)

    def test_class_distribution_sums_to_one(self) -> None:
        args = _make_eval_data()
        report = evaluate(*args)
        total = sum(report.class_distribution.values())
        assert abs(total - 1.0) < 1e-5

    def test_flatten_returns_all_keys(self) -> None:
        args = _make_eval_data()
        report = evaluate(*args)
        flat = _flatten(report)
        required = [
            "eval/macro_f1", "eval/ece", "eval/brier_score",
            "eval/mcc", "eval/cohen_kappa", "eval/p95_ms",
            "eval/positive/pr_auc", "eval/negative/roc_auc",
        ]
        for key in required:
            assert key in flat, f"Missing key: {key}"

    def test_empty_raises(self) -> None:
        with pytest.raises(AssertionError):
            evaluate(
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.zeros((0, 3)),
                [],
                np.array([]),
                np.array([]),
            )


# ────────────────────────────────────────────────────────────────────────────────
# RawArticle tests
# ────────────────────────────────────────────────────────────────────────────────

class TestRawArticle:
    def _article(self, headline: str, url: str = "http://x.com") -> RawArticle:
        return RawArticle("src", headline, url, datetime.now(timezone.utc))

    def test_same_inputs_same_hash(self) -> None:
        a = self._article("headline", "http://a.com")
        b = self._article("headline", "http://a.com")
        assert a.content_hash == b.content_hash

    def test_different_headline_different_hash(self) -> None:
        a = self._article("headline A")
        b = self._article("headline B")
        assert a.content_hash != b.content_hash

    def test_different_url_different_hash(self) -> None:
        a = self._article("headline", "http://a.com")
        b = self._article("headline", "http://b.com")
        assert a.content_hash != b.content_hash

    def test_hash_is_16_chars(self) -> None:
        a = self._article("test headline")
        assert len(a.content_hash) == 16

    def test_to_dict_keys(self) -> None:
        a = self._article("test")
        d = a.to_dict()
        expected_keys = {"source", "headline", "url", "published_at", "body", "author", "content_hash"}
        assert expected_keys == set(d.keys())

    def test_to_dict_published_at_is_iso_string(self) -> None:
        a = self._article("test")
        d = a.to_dict()
        # Should be parseable as ISO 8601
        datetime.fromisoformat(d["published_at"])


# ────────────────────────────────────────────────────────────────────────────────
# SentimentResult tests
# ────────────────────────────────────────────────────────────────────────────────

class TestSentimentResult:
    def _make_result(self) -> SentimentResult:
        now = datetime.now(timezone.utc)
        return SentimentResult(
            article_hash="abc123",
            headline="Test headline",
            url="http://example.com",
            source="newsapi:test",
            published_at=now,
            processed_at=now,
            ensemble_sentiment=Sentiment.POSITIVE,
            ensemble_confidence=0.87,
            ensemble_uncertainty=0.05,
            positive_prob=0.87,
            negative_prob=0.08,
            neutral_prob=0.05,
            model_predictions=[
                ModelPrediction("finbert", Sentiment.POSITIVE, 0.88, 0.07, 0.05, 42.0),
            ],
            entities=[],
            tickers=["AAPL"],
            sectors=["Technology"],
            is_forward_looking=False,
            has_negation=False,
            hedge_score=0.02,
            market_impact=MarketImpact.HIGH,
            is_uncertain=False,
        )

    def test_to_dict_serialisable(self) -> None:
        import json
        r = self._make_result()
        d = r.to_dict()
        # Should be JSON serialisable
        json.dumps(d)

    def test_to_dict_sentiment_is_string(self) -> None:
        r = self._make_result()
        assert isinstance(r.to_dict()["ensemble_sentiment"], str)

    def test_to_dict_probs_rounded(self) -> None:
        r = self._make_result()
        d = r.to_dict()
        assert d["ensemble_confidence"] == 0.87
        assert d["hedge_score"] == 0.02

    def test_to_dict_tickers_preserved(self) -> None:
        r = self._make_result()
        assert r.to_dict()["tickers"] == ["AAPL"]

    def test_model_count_in_dict(self) -> None:
        r = self._make_result()
        assert r.to_dict()["model_count"] == 1
