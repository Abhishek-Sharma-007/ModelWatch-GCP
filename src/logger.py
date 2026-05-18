"""Prediction logging.

Each prediction served by the API is appended to a CSV file under
``data/logs/``. Logs include the timestamp, input features, predicted
label and predicted probability. Logs are deliberately structured to
make downstream analysis (e.g. drift detection) trivial: each input
feature becomes its own column.

Logs never contain authentication tokens, raw API keys or any other
secret values - only the model input features the caller supplied.
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config import SETTINGS
from src.utils import ensure_dir, get_logger


logger = get_logger(__name__)

# Writes to the CSV are guarded by this lock so concurrent FastAPI
# workers in the same process don't interleave rows.
_log_lock = threading.Lock()


def _log_columns() -> list[str]:
    """Stable column ordering for the prediction log CSV."""
    feature_cols = SETTINGS.numeric_features + SETTINGS.categorical_features
    return (
        ["timestamp", "model_version"]
        + feature_cols
        + ["prediction", "prediction_label", "churn_probability"]
    )


def log_prediction(
    features: Dict[str, Any],
    result: Dict[str, Any],
    log_path: str | Path | None = None,
) -> Path:
    """Append a single prediction record to the prediction log CSV.

    Parameters
    ----------
    features:
        The exact input features used for the prediction.
    result:
        Output of :func:`src.predict.predict_one`.
    log_path:
        Optional override for the CSV path. Defaults to
        ``SETTINGS.prediction_log_path``.

    Returns
    -------
    pathlib.Path
        The path that was written to.
    """
    path = Path(log_path) if log_path is not None else SETTINGS.prediction_log_path
    ensure_dir(path.parent)

    columns = _log_columns()
    row = {col: "" for col in columns}
    row["timestamp"] = datetime.now(timezone.utc).isoformat()
    row["model_version"] = result.get("model_version", SETTINGS.model_version)
    for k, v in features.items():
        if k in row:
            row[k] = v
    row["prediction"] = result.get("prediction")
    row["prediction_label"] = result.get("prediction_label")
    row["churn_probability"] = result.get("churn_probability")

    with _log_lock:
        file_exists = path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    return path


def read_prediction_logs(log_path: str | Path | None = None) -> pd.DataFrame:
    """Read the prediction log CSV into a DataFrame.

    Returns an empty DataFrame (with the expected columns) if the file
    does not yet exist.
    """
    path = Path(log_path) if log_path is not None else SETTINGS.prediction_log_path
    columns = _log_columns()
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
