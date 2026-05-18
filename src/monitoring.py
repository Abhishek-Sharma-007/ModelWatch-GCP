"""Monitoring helpers used by the Streamlit dashboard.

This module sits between raw data sources (model artifact, prediction
log CSV, training data) and the UI so the dashboard code itself stays
focused on layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import SETTINGS
from src.data_loader import load_raw_data
from src.drift_detection import (
    detect_drift,
    overall_status,
    save_drift_report,
)
from src.logger import read_prediction_logs


def load_metrics() -> Optional[Dict[str, Any]]:
    """Load the JSON metrics produced by training. Returns None if absent."""
    metrics_path = SETTINGS.reports_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return None


def load_model_card() -> Optional[str]:
    """Return the raw text of the model card markdown, if present."""
    card_path = SETTINGS.reports_dir / "model_card.md"
    if not card_path.exists():
        return None
    return card_path.read_text()


def load_reference_data() -> pd.DataFrame:
    """Reference data used as the drift baseline.

    Currently the full training CSV is used. In a real system this
    would be a snapshot stored alongside the model artifact.
    """
    return load_raw_data()


def load_recent_predictions() -> pd.DataFrame:
    """Load the prediction log CSV (may be empty)."""
    return read_prediction_logs()


def compute_drift_dashboard_data(
    n_recent: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute drift information for the dashboard.

    Parameters
    ----------
    n_recent:
        If provided, only the last ``n_recent`` prediction logs are
        used as the current distribution.

    Returns
    -------
    dict
        Keys: ``has_logs`` (bool), ``n_logs`` (int), ``results`` (list),
        ``overall`` (str), ``threshold`` (float).
    """
    ref = load_reference_data()
    cur = load_recent_predictions()
    if n_recent is not None and len(cur) > n_recent:
        cur = cur.tail(n_recent)

    has_logs = not cur.empty
    if not has_logs:
        return {
            "has_logs": False,
            "n_logs": 0,
            "results": [],
            "overall": "low",
            "threshold": SETTINGS.drift_threshold,
        }

    results = detect_drift(ref, cur)
    save_drift_report(results)

    return {
        "has_logs": True,
        "n_logs": int(len(cur)),
        "results": results,
        "overall": overall_status(results),
        "threshold": SETTINGS.drift_threshold,
    }


def summarize_predictions(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary stats over the prediction log dataframe."""
    if df.empty:
        return {
            "total_predictions": 0,
            "churn_count": 0,
            "no_churn_count": 0,
            "avg_probability": 0.0,
            "churn_rate": 0.0,
        }
    total = int(len(df))
    churn = int((df["prediction"].astype(float) == 1).sum())
    avg_prob = float(df["churn_probability"].astype(float).mean())
    return {
        "total_predictions": total,
        "churn_count": churn,
        "no_churn_count": total - churn,
        "avg_probability": round(avg_prob, 4),
        "churn_rate": round(churn / total, 4) if total else 0.0,
    }
