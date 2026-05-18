"""Shared feature validation for API, dashboard and direct Python calls."""

from __future__ import annotations

from typing import Any, Mapping

from src.config import SETTINGS


CONTRACT_TYPES = ("Month-to-month", "One year", "Two year")
INTERNET_SERVICES = ("DSL", "Fiber optic", "No")
PAYMENT_METHODS = (
    "Electronic check",
    "Mailed check",
    "Bank transfer",
    "Credit card",
)

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "tenure_months": (0, 120),
    "monthly_charges": (0.0, 1000.0),
    "total_charges": (0.0, 100000.0),
    "support_tickets": (0, 100),
    "is_senior_citizen": (0, 1),
}

CATEGORICAL_ALLOWED: dict[str, tuple[str, ...]] = {
    "contract_type": CONTRACT_TYPES,
    "internet_service": INTERNET_SERVICES,
    "payment_method": PAYMENT_METHODS,
}


def validate_feature_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized feature dict or raise ``ValueError``.

    This validation intentionally mirrors the FastAPI/Pydantic schema so
    direct calls from Streamlit or notebooks cannot bypass basic domain
    checks.
    """
    required = SETTINGS.numeric_features + SETTINGS.categorical_features
    missing = [feature for feature in required if feature not in payload]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    normalized: dict[str, Any] = {}
    for feature in SETTINGS.numeric_features:
        value = payload[feature]
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be numeric.") from exc

        minimum, maximum = FEATURE_BOUNDS[feature]
        if not minimum <= numeric_value <= maximum:
            raise ValueError(f"{feature} must be between {minimum} and {maximum}.")

        if feature in {"tenure_months", "support_tickets", "is_senior_citizen"}:
            if not numeric_value.is_integer():
                raise ValueError(f"{feature} must be an integer.")
            normalized[feature] = int(numeric_value)
        else:
            normalized[feature] = numeric_value

    for feature, allowed in CATEGORICAL_ALLOWED.items():
        value = str(payload[feature])
        if value not in allowed:
            raise ValueError(
                f"{feature} must be one of {list(allowed)}, got {value!r}."
            )
        normalized[feature] = value

    return normalized
