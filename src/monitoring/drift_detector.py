"""Model and data drift detection.

Checks (run hourly via Prefect):
  PSI on confidence scores        — threshold 0.2
  Chi-squared on label counts     — alert if p < 0.01
  Jensen-Shannon divergence       — alert if > 0.1

Outputs:
  Prometheus gauges (always)
  Structured log (always)
  MLflow run tag on drift alert (via orchestration/pipeline.py)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from prometheus_client import Gauge
from scipy.stats import chi2_contingency, entropy

from config.settings import settings

logger = logging.getLogger(__name__)

PSI_GAUGE = Gauge("pulsede_psi_confidence", "PSI on confidence scores")
JS_DIV_GAUGE = Gauge("pulsede_js_divergence", "JS divergence on sentiment probs")
CHI2_PVALUE_GAUGE = Gauge("pulsede_chi2_pvalue", "Chi-squared p-value label drift")


@dataclass
class DriftReport:
    timestamp: datetime
    psi_confidence: float
    js_divergence: float
    chi2_pvalue: float
    is_drifting: bool
    alerts: list[str]


class DriftDetector:
    def __init__(self) -> None:
        self._ref_confidence: np.ndarray | None = None
        self._ref_label_dist: np.ndarray | None = None

    def set_reference(self, confidence_scores: list[float], sentiment_labels: list[str]) -> None:
        self._ref_confidence = np.array(confidence_scores, dtype=np.float32)
        counts = np.array(
            [
                sum(1 for s in sentiment_labels if s == "positive"),
                sum(1 for s in sentiment_labels if s == "negative"),
                sum(1 for s in sentiment_labels if s == "neutral"),
            ],
            dtype=np.float32,
        )
        self._ref_label_dist = counts / counts.sum()
        logger.info("drift_reference_set n=%d", len(confidence_scores))

    def check(
        self,
        current_confidence: list[float],
        current_sentiments: list[str],
        current_proba: np.ndarray,
    ) -> DriftReport:
        if self._ref_confidence is None or self._ref_label_dist is None:
            raise RuntimeError("Call set_reference() first")

        alerts: list[str] = []
        cur_conf = np.array(current_confidence, dtype=np.float32)

        psi = self._psi(self._ref_confidence, cur_conf)
        PSI_GAUGE.set(psi)
        if psi > settings.monitoring.psi_threshold:
            alerts.append(f"PSI={psi:.4f} > {settings.monitoring.psi_threshold}")
            logger.warning("drift_psi_alert psi=%.4f", psi)

        cur_counts = np.array(
            [
                sum(1 for s in current_sentiments if s == "positive"),
                sum(1 for s in current_sentiments if s == "negative"),
                sum(1 for s in current_sentiments if s == "neutral"),
            ],
            dtype=np.float32,
        )
        ref_expected = self._ref_label_dist * cur_counts.sum()
        _, pvalue, _, _ = chi2_contingency(np.vstack([cur_counts, ref_expected]).clip(min=1))
        CHI2_PVALUE_GAUGE.set(float(pvalue))
        if pvalue < 0.01:
            alerts.append(f"Label drift chi2 p={pvalue:.4f}")
            logger.warning("drift_label_alert pvalue=%.4f", pvalue)

        ref_proba = self._ref_label_dist
        cur_mean_proba = current_proba.mean(axis=0)
        js = self._js(ref_proba, cur_mean_proba)
        JS_DIV_GAUGE.set(float(js))
        if js > 0.1:
            alerts.append(f"Prediction drift JS={js:.4f}")
            logger.warning("drift_prediction_alert js=%.4f", js)

        return DriftReport(
            timestamp=datetime.now(UTC),
            psi_confidence=round(float(psi), 6),
            js_divergence=round(float(js), 6),
            chi2_pvalue=round(float(pvalue), 6),
            is_drifting=bool(alerts),
            alerts=alerts,
        )

    @staticmethod
    def _psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """PSI < 0.1 stable, 0.1-0.2 moderate drift, > 0.2 significant."""
        bins = np.linspace(0, 1, n_bins + 1)
        ref_pct = (np.histogram(reference, bins=bins)[0] / len(reference)).clip(1e-6)
        cur_pct = (np.histogram(current, bins=bins)[0] / len(current)).clip(1e-6)
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    @staticmethod
    def _js(p: np.ndarray, q: np.ndarray) -> float:
        p = (p + 1e-9) / (p + 1e-9).sum()
        q = (q + 1e-9) / (q + 1e-9).sum()
        m = 0.5 * (p + q)
        return min(0.5 * float(entropy(p, m) + entropy(q, m)), 1.0)
