"""Three-model FinBERT ensemble with uncertainty quantification and calibration.

Models:
  ProsusAI/finbert           — primary, 50% weight (Araci 2019)
  yiyanghkust/finbert-tone   — tone-aware, 30% weight
  mrm8488/distilroberta-...  — speed/diversity, 20% weight

Uncertainty: Monte Carlo Dropout (Gal & Ghahramani, NeurIPS 2016)
Calibration: Temperature scaling (Guo et al., ICML 2017)
Ensemble:    Deep ensembles reduce variance by ~25% vs single model
             (Lakshminarayanan et al., NeurIPS 2017)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import entr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import settings
from src.ingestion.schema import ModelPrediction, Sentiment

logger = logging.getLogger(__name__)

_LABEL_MAPS: dict[str, dict[str, Sentiment]] = {
    settings.ml.primary_model: {
        "positive": Sentiment.POSITIVE,
        "negative": Sentiment.NEGATIVE,
        "neutral": Sentiment.NEUTRAL,
    },
    settings.ml.secondary_model: {
        "Positive": Sentiment.POSITIVE,
        "Negative": Sentiment.NEGATIVE,
        "Neutral": Sentiment.NEUTRAL,
    },
    settings.ml.tertiary_model: {
        "positive": Sentiment.POSITIVE,
        "negative": Sentiment.NEGATIVE,
        "neutral": Sentiment.NEUTRAL,
    },
}

_IDX_TO_SENTIMENT = [Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL]


class _MCDropoutModel:
    """Wraps a HuggingFace classification model for inference with optional dropout."""

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(settings.ml.device)
        self._label_map = _LABEL_MAPS[model_name]
        self._id2label: dict[int, str] = self.model.config.id2label

    @torch.no_grad()
    def predict(self, texts: list[str], enable_dropout: bool = False) -> list[dict[str, float]]:
        self.model.train() if enable_dropout else self.model.eval()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.ml.max_length,
            return_tensors="pt",
        ).to(settings.ml.device)

        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for prob_row in probs:
            row: dict[str, float] = {}
            for idx, prob in enumerate(prob_row):
                raw_label = self._id2label[idx]
                canonical = self._label_map.get(raw_label, Sentiment.NEUTRAL)
                row[canonical.value] = float(prob)
            results.append(row)
        return results


class SentimentEnsemble:
    """Production-grade ensemble classifier. Instantiate once as a singleton."""

    def __init__(self) -> None:
        logger.info("Loading ensemble models — this may take 30-60s on first run")
        self._models = [
            _MCDropoutModel(settings.ml.primary_model),
            _MCDropoutModel(settings.ml.secondary_model),
            _MCDropoutModel(settings.ml.tertiary_model),
        ]
        self._weights = np.array(settings.ml.weights, dtype=np.float32)
        self._temperature = settings.ml.temperature
        logger.info("Ensemble ready — %d models loaded", len(self._models))

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Full pipeline: ensemble + MC Dropout uncertainty + temperature scaling.

        Returns list of dicts:
            sentiment, confidence, uncertainty, is_uncertain,
            positive_prob, negative_prob, neutral_prob, model_predictions
        """
        if not texts:
            return []

        per_model_probs: list[np.ndarray] = []
        model_preds_per_text: list[list[ModelPrediction]] = [[] for _ in texts]

        for i, model in enumerate(self._models):
            t0 = time.perf_counter()
            preds = model.predict(texts, enable_dropout=False)
            latency_ms = (time.perf_counter() - t0) * 1000 / len(texts)

            probs_matrix = np.array(
                [
                    [p.get("positive", 0.0), p.get("negative", 0.0), p.get("neutral", 0.0)]
                    for p in preds
                ],
                dtype=np.float32,
            )
            per_model_probs.append(probs_matrix)

            for j, probs in enumerate(probs_matrix):
                argmax = int(probs.argmax())
                model_preds_per_text[j].append(
                    ModelPrediction(
                        model_name=model.name,
                        sentiment=_IDX_TO_SENTIMENT[argmax],
                        positive_prob=float(probs[0]),
                        negative_prob=float(probs[1]),
                        neutral_prob=float(probs[2]),
                        latency_ms=latency_ms,
                    )
                )

        # Weighted average: (n_models, n_texts, 3) → (n_texts, 3)
        stacked = np.stack(per_model_probs, axis=0)
        ensemble_probs = (stacked * self._weights[:, None, None]).sum(axis=0)

        # Temperature scaling (calibration)
        ensemble_logits = np.log(ensemble_probs + 1e-9) / self._temperature
        calibrated = self._softmax(ensemble_logits)

        # MC Dropout uncertainty (predictive entropy)
        mc_uncertainty = self._mc_dropout_uncertainty(texts)

        results = []
        for i in range(len(texts)):
            probs = calibrated[i]
            argmax = int(probs.argmax())
            confidence = float(probs[argmax])
            uncertainty = float(mc_uncertainty[i])
            results.append(
                {
                    "sentiment": _IDX_TO_SENTIMENT[argmax],
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "is_uncertain": uncertainty > settings.ml.uncertainty_threshold,
                    "positive_prob": float(probs[0]),
                    "negative_prob": float(probs[1]),
                    "neutral_prob": float(probs[2]),
                    "model_predictions": model_preds_per_text[i],
                }
            )
        return results

    def calibrate_temperature(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Learn scalar T via NLL minimisation on a calibration set (LBFGS)."""
        T = torch.nn.Parameter(torch.ones(1))
        optimiser = torch.optim.LBFGS([T], lr=lr, max_iter=n_epochs)
        logits_t = torch.from_numpy(logits).float()
        labels_t = torch.from_numpy(labels).long()

        def _closure() -> torch.Tensor:
            optimiser.zero_grad()
            loss = F.cross_entropy(logits_t / T, labels_t)
            loss.backward()
            return loss

        optimiser.step(_closure)
        self._temperature = float(T.item())
        logger.info("Temperature calibrated T=%.4f", self._temperature)
        return self._temperature

    def _mc_dropout_uncertainty(self, texts: list[str]) -> np.ndarray:
        """T stochastic passes on primary model → predictive entropy ∈ [0, 1]."""
        T = settings.ml.mc_dropout_passes
        mc_samples: list[np.ndarray] = []
        primary = self._models[0]

        for _ in range(T):
            preds = primary.predict(texts, enable_dropout=True)
            mc_samples.append(
                np.array(
                    [
                        [p.get("positive", 0.0), p.get("negative", 0.0), p.get("neutral", 0.0)]
                        for p in preds
                    ],
                    dtype=np.float32,
                )
            )

        mean_probs = np.stack(mc_samples, axis=0).mean(axis=0)
        # Normalise entropy to [0,1] by dividing by ln(3)
        return entr(mean_probs).sum(axis=1) / np.log(3)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
