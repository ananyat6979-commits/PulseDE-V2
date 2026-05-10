"""Prefect 2 orchestration — replaces the naive schedule module.

Flows:
  fetch_news_flow  every 5 min  — NewsAPI + RSS → Kafka
  drift_check_flow every 1 hour — TimescaleDB → DriftDetector → MLflow tags

Why Prefect over schedule:
  - Per-task retries with exponential backoff (not per-run)
  - Persistent run history, artifact storage, failure notifications
  - Parameterised flows: trigger ad-hoc runs via CLI or REST API
  - Concurrency limits: prevent overlapping fetch runs
  - UI at localhost:4200 for monitoring without building anything
"""

from __future__ import annotations

import logging
from datetime import timedelta

import mlflow
import numpy as np
from prefect import flow, get_run_logger, task
from prefect.tasks import task_input_hash

from config.settings import settings
from src.ingestion.kafka_producer import ArticleProducer
from src.ingestion.news_fetcher import NewsFetcher
from src.monitoring.drift_detector import DriftDetector
from src.storage.timescale_writer import TimescaleWriter

logger = logging.getLogger(__name__)


@task(
    name="fetch-and-publish-news",
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(minutes=4),
    tags=["ingestion"],
)
def fetch_and_publish_task() -> int:
    log = get_run_logger()
    with NewsFetcher() as fetcher:
        articles = fetcher.fetch_all()
    log.info(f"Fetched {len(articles)} unique articles")

    if articles:
        with ArticleProducer() as producer:
            producer.publish_batch(articles)
        log.info(f"Published {len(articles)} articles to Kafka")

    return len(articles)


@task(
    name="check-model-drift",
    retries=2,
    retry_delay_seconds=60,
    tags=["monitoring"],
)
def check_drift_task(lookback_hours: int = 1, reference_hours: int = 24) -> dict:
    log = get_run_logger()
    db = TimescaleWriter()
    detector = DriftDetector()

    reference = db.query_recent(hours=reference_hours, limit=2000)
    if len(reference) < 50:
        log.warning("Insufficient reference data — skipping drift check")
        return {}

    detector.set_reference(
        [r["ensemble_confidence"] for r in reference],
        [r["ensemble_sentiment"] for r in reference],
    )

    current = db.query_recent(hours=lookback_hours, limit=500)
    if len(current) < 10:
        log.info("Insufficient current data — skipping")
        return {}

    cur_proba = np.array(
        [[r["positive_prob"], r["negative_prob"], r["neutral_prob"]] for r in current]
    )
    report = detector.check(
        [r["ensemble_confidence"] for r in current],
        [r["ensemble_sentiment"] for r in current],
        cur_proba,
    )

    try:
        exp = mlflow.get_experiment_by_name(settings.mlflow.experiment_name)
        with mlflow.start_run(
            run_name="drift_check",
            experiment_id=exp.experiment_id if exp else None,
        ):
            mlflow.log_metrics(
                {
                    "drift/psi": report.psi_confidence,
                    "drift/js": report.js_divergence,
                    "drift/chi2_pvalue": report.chi2_pvalue,
                }
            )
            if report.is_drifting:
                mlflow.set_tag("drift_alert", "; ".join(report.alerts))
    except Exception as exc:
        log.warning(f"MLflow logging failed: {exc}")

    if report.is_drifting:
        log.warning(f"DRIFT DETECTED: {report.alerts}")

    return {
        "psi": report.psi_confidence,
        "js": report.js_divergence,
        "chi2_pvalue": report.chi2_pvalue,
        "is_drifting": report.is_drifting,
        "alerts": report.alerts,
    }


@flow(
    name="pulsede-fetch-news",
    description="Ingest financial news from all sources into Kafka",
    version="2.0.0",
    log_prints=True,
)
def fetch_news_flow() -> None:
    count = fetch_and_publish_task()
    print(f"Pipeline run complete: {count} articles")


@flow(
    name="pulsede-drift-check",
    description="Hourly model and data drift monitoring",
    version="2.0.0",
)
def drift_check_flow() -> None:
    result = check_drift_task()
    if result.get("is_drifting"):
        print(f"Drift alerts: {result['alerts']}")


if __name__ == "__main__":
    fetch_news_flow()
