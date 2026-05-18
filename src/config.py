"""Project configuration.

All configurable paths and thresholds are loaded from environment
variables (with sensible defaults) so that the project can be deployed
to different environments without code changes. A ``.env`` file is
supported via ``python-dotenv`` for local development; **never** commit
``.env`` itself - only ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency at import time
    pass


# Project root is the parent of ``src/``.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def _abs(path_str: str) -> Path:
    """Return ``path_str`` resolved against the project root if relative."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    model_version: str
    model_path: Path
    prediction_log_path: Path
    raw_data_path: Path
    reports_dir: Path
    drift_threshold: float
    random_state: int

    @property
    def numeric_features(self) -> list[str]:
        return [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "support_tickets",
            "is_senior_citizen",
        ]

    @property
    def categorical_features(self) -> list[str]:
        return [
            "contract_type",
            "internet_service",
            "payment_method",
        ]

    @property
    def target(self) -> str:
        return "churn"


def get_settings() -> Settings:
    """Load settings from environment with defaults suitable for local dev."""
    return Settings(
        model_version=os.getenv("MODEL_VERSION", "v1.0.0"),
        model_path=_abs(os.getenv("MODEL_PATH", "models/churn_model.joblib")),
        prediction_log_path=_abs(
            os.getenv("PREDICTION_LOG_PATH", "data/logs/prediction_logs.csv")
        ),
        raw_data_path=_abs(
            os.getenv("RAW_DATA_PATH", "data/raw/customer_churn_sample.csv")
        ),
        reports_dir=_abs(os.getenv("REPORTS_DIR", "reports")),
        drift_threshold=float(os.getenv("DRIFT_THRESHOLD", "0.20")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
    )


# Module-level singleton for convenience.
SETTINGS: Settings = get_settings()
