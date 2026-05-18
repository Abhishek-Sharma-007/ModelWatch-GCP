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

from dotenv import load_dotenv


load_dotenv()

# Project root is the parent of ``src/``.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def _abs(path_str: str) -> Path:
    """Return ``path_str`` resolved against the project root if relative."""
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a clear error message."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}.")


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    """Parse an integer environment variable with optional lower bound."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _env_float(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse a float environment variable with optional bounds."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = float(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}.")
    return value


def _env_csv(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Parse a comma-separated environment variable into trimmed values."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


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
    api_host: str
    api_port: int
    api_reload: bool
    api_key: str | None
    cors_allow_origins: tuple[str, ...]
    log_level: str

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

    @property
    def api_key_required(self) -> bool:
        """Whether protected API endpoints require ``X-API-Key``."""
        return self.api_key is not None and self.api_key.strip() != ""


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
        drift_threshold=_env_float("DRIFT_THRESHOLD", 0.20, minimum=0.0, maximum=1.0),
        random_state=_env_int("RANDOM_STATE", 42, minimum=0),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=_env_int("API_PORT", 8000, minimum=1),
        api_reload=_env_bool("API_RELOAD", False),
        api_key=os.getenv("API_KEY") or None,
        cors_allow_origins=_env_csv("CORS_ALLOW_ORIGINS"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


# Module-level singleton for convenience.
SETTINGS: Settings = get_settings()
