"""Research-grade ML evaluation for financial sentiment classification.

Metrics:
  Macro / micro / weighted F1
  Per-class PR-AUC (area under precision-recall curve)
  Per-class ROC-AUC (one-vs-rest)
  Matthews Correlation Coefficient (MCC)
  Cohen's Kappa
  Expected Calibration Error (ECE) — 15-bin
  Brier Score (proper scoring rule)
  Reliability diagram data
  Ensemble pairwise disagreement rate
  Latency percentiles (p50 / p95 / p99)
  Confusion matrix (raw + normalised)

All results are MLflow-ready: log_to_mlflow() serialises everything.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)
CLASS_NAMES = ["positive", "negative", "neutral"]


@dataclass
class PerClassMetrics:
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    support: int


@dataclass
class CalibrationMetrics:
    ece: float
    brier_score: float
    reliability_diagram: list[dict[str, float]]


@dataclass
class EvaluationReport:
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    macro_roc_auc: float
    mcc: float
    cohen_kappa: float
    accuracy: float
    per_class: dict[str, PerClassMetrics]
    calibration: CalibrationMetrics
    confusion_matrix: list[list[int]]
    confusion_matrix_normalised: list[list[float]]
    ensemble_disagreement_rate: float
    uncertain_fraction: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_samples: int
    class_distribution: dict[str, float]
    model_names: list[str] = field(default_factory=list)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    per_model_preds: list[np.ndarray],
    uncertainties: np.ndarray,
    latencies_ms: np.ndarray,
    model_names: list[str] | None = None,
) -> EvaluationReport:
    """Compute full evaluation report.

    Args:
        y_true:           (N,) ground-truth class indices 0=pos 1=neg 2=neu
        y_pred:           (N,) ensemble hard predictions
        y_proba:          (N, 3) ensemble soft probabilities
        per_model_preds:  list of (N,) per-sub-model predictions
        uncertainties:    (N,) MC Dropout entropy per sample
        latencies_ms:     (N,) per-sample inference time
        model_names:      optional labels for sub-models
    """
    n = len(y_true)
    assert n > 0, "Empty evaluation set"

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    accuracy = float((y_true == y_pred).mean())
    macro_roc_auc = float(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    )

    per_class: dict[str, PerClassMetrics] = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        bt = (y_true == i).astype(int)
        bp = (y_pred == i).astype(int)
        per_class[cls_name] = PerClassMetrics(
            precision=float(precision_score(bt, bp, zero_division=0)),
            recall=float(recall_score(bt, bp, zero_division=0)),
            f1=float(f1_score(bt, bp, zero_division=0)),
            pr_auc=float(average_precision_score(bt, y_proba[:, i])),
            roc_auc=float(roc_auc_score(bt, y_proba[:, i])),
            support=int(bt.sum()),
        )

    ece = _ece(y_true, y_proba)
    brier = float(np.mean([
        brier_score_loss((y_true == i).astype(int), y_proba[:, i]) for i in range(3)
    ]))
    frac_pos, mean_pred = calibration_curve((y_true == 0).astype(int), y_proba[:, 0], n_bins=10)
    reliability = [
        {"mean_predicted_prob": float(mp), "fraction_positive": float(fp)}
        for mp, fp in zip(mean_pred, frac_pos)
    ]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    disagreement = _pairwise_disagreement(per_model_preds)
    uncertain_frac = float((uncertainties > 0.15).mean())

    return EvaluationReport(
        macro_f1=round(macro_f1, 6),
        micro_f1=round(micro_f1, 6),
        weighted_f1=round(weighted_f1, 6),
        macro_roc_auc=round(macro_roc_auc, 6),
        mcc=round(mcc, 6),
        cohen_kappa=round(kappa, 6),
        accuracy=round(accuracy, 6),
        per_class=per_class,
        calibration=CalibrationMetrics(
            ece=round(ece, 6), brier_score=round(brier, 6),
            reliability_diagram=reliability,
        ),
        confusion_matrix=cm.tolist(),
        confusion_matrix_normalised=cm_norm.round(4).tolist(),
        ensemble_disagreement_rate=round(disagreement, 4),
        uncertain_fraction=round(uncertain_frac, 4),
        p50_ms=round(float(np.percentile(latencies_ms, 50)), 2),
        p95_ms=round(float(np.percentile(latencies_ms, 95)), 2),
        p99_ms=round(float(np.percentile(latencies_ms, 99)), 2),
        n_samples=n,
        class_distribution={
            cls: float((y_true == i).mean())
            for i, cls in enumerate(CLASS_NAMES)
        },
        model_names=model_names or [],
    )


def log_to_mlflow(report: EvaluationReport, run_id: str | None = None) -> None:
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_metrics(_flatten(report))
        cm_path = Path("/tmp/confusion_matrix.json")
        cm_path.write_text(json.dumps({
            "raw": report.confusion_matrix,
            "normalised": report.confusion_matrix_normalised,
            "class_names": CLASS_NAMES,
        }, indent=2))
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
    logger.info("mlflow_eval_logged n=%d macro_f1=%.4f", report.n_samples, report.macro_f1)


def _ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    confidences = y_proba.max(axis=1)
    correct = (y_proba.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def _pairwise_disagreement(per_model_preds: list[np.ndarray]) -> float:
    if len(per_model_preds) < 2:
        return 0.0
    stacked = np.stack(per_model_preds, axis=1)
    return float((stacked != stacked[:, 0:1]).any(axis=1).mean())


def _flatten(report: EvaluationReport) -> dict[str, float]:
    flat: dict[str, float] = {
        "eval/macro_f1": report.macro_f1,
        "eval/micro_f1": report.micro_f1,
        "eval/weighted_f1": report.weighted_f1,
        "eval/macro_roc_auc": report.macro_roc_auc,
        "eval/mcc": report.mcc,
        "eval/cohen_kappa": report.cohen_kappa,
        "eval/accuracy": report.accuracy,
        "eval/ece": report.calibration.ece,
        "eval/brier_score": report.calibration.brier_score,
        "eval/ensemble_disagreement": report.ensemble_disagreement_rate,
        "eval/uncertain_fraction": report.uncertain_fraction,
        "eval/p50_ms": report.p50_ms,
        "eval/p95_ms": report.p95_ms,
        "eval/p99_ms": report.p99_ms,
        "eval/n_samples": float(report.n_samples),
    }
    for cls, m in report.per_class.items():
        flat[f"eval/{cls}/precision"] = m.precision
        flat[f"eval/{cls}/recall"] = m.recall
        flat[f"eval/{cls}/f1"] = m.f1
        flat[f"eval/{cls}/pr_auc"] = m.pr_auc
        flat[f"eval/{cls}/roc_auc"] = m.roc_auc
        flat[f"eval/{cls}/support"] = float(m.support)
    return flat
