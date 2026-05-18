"""FastAPI application that serves churn predictions.

Endpoints
---------
GET  /        - project name and a friendly health message
GET  /health  - JSON status + whether the model artifact is present
POST /predict - run the churn model on a single customer payload

Every successful prediction is appended to
``data/logs/prediction_logs.csv`` so it can be analyzed by the drift
detection module and the Streamlit dashboard.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from src.config import SETTINGS
from src.logger import log_prediction
from src.predict import (
    ModelNotLoadedError,
    is_model_available,
    predict_one,
)
from src.security import SecurityHeadersMiddleware, require_api_key
from src.utils import get_logger


ContractType: TypeAlias = Literal["Month-to-month", "One year", "Two year"]
InternetService: TypeAlias = Literal["DSL", "Fiber optic", "No"]
PaymentMethod: TypeAlias = Literal[
    "Electronic check", "Mailed check", "Bank transfer", "Credit card"
]


logger = get_logger("api")


class CustomerFeatures(BaseModel):
    """Pydantic schema for a single customer prediction request."""

    tenure_months: int = Field(..., ge=0, le=120, description="Months as customer")
    monthly_charges: float = Field(..., ge=0.0, le=1000.0)
    total_charges: float = Field(..., ge=0.0, le=100000.0)
    contract_type: ContractType
    internet_service: InternetService
    support_tickets: int = Field(..., ge=0, le=100)
    payment_method: PaymentMethod
    is_senior_citizen: int = Field(..., ge=0, le=1)

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "tenure_months": 12,
                "monthly_charges": 75.5,
                "total_charges": 906.0,
                "contract_type": "Month-to-month",
                "internet_service": "Fiber optic",
                "support_tickets": 3,
                "payment_method": "Electronic check",
                "is_senior_citizen": 0,
            }
        },
    }


class PredictionResponse(BaseModel):
    """Response schema for ``POST /predict``."""

    prediction: int = Field(..., ge=0, le=1)
    prediction_label: Literal["Churn", "No Churn"]
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    model_version: str


app = FastAPI(
    title="ModelWatch-GCP Churn Prediction API",
    description=(
        "Production-style MLOps service that predicts customer churn "
        "and logs every prediction for monitoring and drift detection."
    ),
    version=SETTINGS.model_version,
)

app.add_middleware(SecurityHeadersMiddleware)
if SETTINGS.cors_allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(SETTINGS.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


@app.get("/", tags=["meta"])
def root() -> dict:
    """Root endpoint with a friendly project banner."""
    return {
        "project": "ModelWatch-GCP",
        "message": "Customer churn prediction service is up.",
        "docs_url": "/docs",
        "api_key_required": SETTINGS.api_key_required,
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Lightweight health check used by load balancers and Cloud Run."""
    return {
        "status": "ok",
        "model_loaded": is_model_available(),
        "model_version": SETTINGS.model_version,
        "api_key_required": SETTINGS.api_key_required,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    dependencies=[Depends(require_api_key)],
)
def predict(features: CustomerFeatures) -> PredictionResponse:
    """Predict churn for a single customer.

    Returns a 503 if the model artifact is not available (i.e. you have
    not run ``python run_training.py`` yet). Schema validation issues are
    handled automatically by FastAPI as 422 responses. When ``API_KEY`` is
    configured, callers must pass the same value in the ``X-API-Key`` header.
    """
    payload = features.model_dump()
    try:
        result = predict_one(payload)
    except ModelNotLoadedError as exc:
        logger.error("Model not loaded: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected prediction error")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    try:
        log_prediction(payload, result)
    except Exception:  # pragma: no cover - logging must not break inference
        logger.exception("Failed to log prediction; returning prediction anyway")

    return PredictionResponse(**result)


@app.exception_handler(Exception)
async def _global_exception_handler(_request, exc: Exception):  # pragma: no cover
    """Last-resort error handler so the API never leaks tracebacks."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
