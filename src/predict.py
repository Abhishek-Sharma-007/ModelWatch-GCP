"""Prediction utilities.

Loads the trained model pipeline lazily, validates the input payload
against the expected feature schema and returns a structured
prediction result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.config import SETTINGS
from src.utils import get_logger
from src.validation import validate_feature_payload


logger = get_logger(__name__)

# Cached pipeline so the model file is only read from disk once per process.
_model_cache: Optional[object] = None
_model_path_cache: Optional[Path] = None


class ModelNotLoadedError(RuntimeError):
    """Raised when the trained model artifact cannot be found on disk."""


def reset_cache() -> None:
    """Clear the in-process model cache (useful for tests)."""
    global _model_cache, _model_path_cache
    _model_cache = None
    _model_path_cache = None


def load_model(path: str | Path | None = None) -> object:
    """Load (and cache) the trained model pipeline.

    Raises
    ------
    ModelNotLoadedError
        If the model file does not exist. Train the model first with
        ``python run_training.py``.
    """
    global _model_cache, _model_path_cache
    model_path = Path(path) if path is not None else SETTINGS.model_path

    if _model_cache is not None and _model_path_cache == model_path:
        return _model_cache

    if not model_path.exists():
        raise ModelNotLoadedError(
            f"Model artifact not found at {model_path}. "
            "Run `python run_training.py` to generate it."
        )

    logger.info("Loading model from %s", model_path)
    _model_cache = joblib.load(model_path)
    _model_path_cache = model_path
    return _model_cache


def is_model_available(path: str | Path | None = None) -> bool:
    """Return True if a model artifact is present on disk."""
    model_path = Path(path) if path is not None else SETTINGS.model_path
    return model_path.exists()


def _validate_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required features are present, typed and in expected ranges."""
    return validate_feature_payload(payload)


def predict_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Predict churn for a single customer payload.

    Parameters
    ----------
    payload:
        Dict with all numeric and categorical features (see
        :data:`src.config.SETTINGS`). Extra keys are ignored.

    Returns
    -------
    dict
        Keys: ``prediction`` (0/1), ``prediction_label``
        ("Churn"/"No Churn"), ``churn_probability`` (float in [0, 1]),
        ``model_version`` (str).
    """
    features = _validate_features(payload)
    model = load_model()

    row = pd.DataFrame([features])
    proba = float(model.predict_proba(row)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "prediction_label": "Churn" if pred == 1 else "No Churn",
        "churn_probability": round(proba, 4),
        "model_version": SETTINGS.model_version,
    }
