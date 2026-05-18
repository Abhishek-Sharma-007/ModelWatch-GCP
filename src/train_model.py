"""Train churn prediction models.

Trains a Logistic Regression baseline and a Random Forest model, picks
the better one based on ROC-AUC, saves the full ``Pipeline`` (including
preprocessing) to disk with ``joblib`` and writes a markdown report
plus a ``metrics.json`` file under ``reports/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import SETTINGS
from src.data_loader import load_raw_data, train_test_split_data
from src.evaluate_model import compute_metrics, format_metrics_markdown
from src.preprocessing import build_preprocessor
from src.utils import ensure_dir, get_logger


logger = get_logger(__name__)


def _build_pipelines() -> Dict[str, Pipeline]:
    """Return the candidate model pipelines keyed by display name."""
    pre = build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=SETTINGS.random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", pre),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_split=5,
                        random_state=SETTINGS.random_state,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _fit_and_score(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Fit a pipeline and return its test-set metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return compute_metrics(np.asarray(y_test), y_pred, y_proba)


def train_and_select(
    df: pd.DataFrame | None = None,
) -> Tuple[Pipeline, str, Dict[str, Dict[str, Any]]]:
    """Train all candidate models and pick the best by ROC-AUC.

    Returns
    -------
    (best_pipeline, best_name, all_metrics)
    """
    if df is None:
        df = load_raw_data()

    X_train, X_test, y_train, y_test = train_test_split_data(df)
    logger.info(
        "Training set: %d rows, test set: %d rows", len(X_train), len(X_test)
    )

    pipelines = _build_pipelines()
    all_metrics: Dict[str, Dict[str, Any]] = {}
    for name, pipe in pipelines.items():
        logger.info("Training %s ...", name)
        all_metrics[name] = _fit_and_score(pipe, X_train, X_test, y_train, y_test)
        logger.info(
            "%s -> ROC-AUC=%.4f, F1=%.4f",
            name,
            all_metrics[name]["roc_auc"],
            all_metrics[name]["f1"],
        )

    best_name = max(all_metrics, key=lambda n: all_metrics[n]["roc_auc"])
    best_pipeline = pipelines[best_name]
    logger.info("Selected best model: %s", best_name)
    return best_pipeline, best_name, all_metrics


def save_artifacts(
    pipeline: Pipeline,
    best_name: str,
    all_metrics: Dict[str, Dict[str, Any]],
) -> Path:
    """Persist model artifact, JSON metrics and markdown report.

    Returns the path the model was written to.
    """
    ensure_dir(SETTINGS.model_path.parent)
    ensure_dir(SETTINGS.reports_dir)

    joblib.dump(pipeline, SETTINGS.model_path)
    logger.info("Saved model pipeline to %s", SETTINGS.model_path)

    metrics_payload = {
        "model_version": SETTINGS.model_version,
        "best_model": best_name,
        "models": all_metrics,
    }
    metrics_json_path = SETTINGS.reports_dir / "metrics.json"
    metrics_json_path.write_text(json.dumps(metrics_payload, indent=2))
    logger.info("Saved metrics JSON to %s", metrics_json_path)

    report_path = SETTINGS.reports_dir / "model_report.md"
    md = ["# Model Report", "", f"Model version: **{SETTINGS.model_version}**", ""]
    md.append(f"**Selected model:** `{best_name}`")
    md.append("")
    for name, metrics in all_metrics.items():
        md.append(format_metrics_markdown(metrics, name))
    report_path.write_text("\n".join(md))
    logger.info("Saved markdown report to %s", report_path)

    return SETTINGS.model_path


def main() -> None:  # pragma: no cover - CLI entry point
    pipeline, best_name, metrics = train_and_select()
    save_artifacts(pipeline, best_name, metrics)
    print(f"Best model: {best_name}")
    print(f"Model saved to: {SETTINGS.model_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
