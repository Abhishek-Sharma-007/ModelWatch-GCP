"""Tests for the FastAPI prediction service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained_model_path):
    """Build a TestClient AFTER the model artifact has been trained."""
    from api.fastapi_app import app

    return TestClient(app)


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["project"] == "ModelWatch-GCP"


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "model_version" in body


def test_predict_endpoint_returns_prediction(client, sample_payload):
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in {0, 1}
    assert body["prediction_label"] in {"Churn", "No Churn"}
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_rejects_invalid_payload(client):
    bad_payload = {
        "tenure_months": -5,  # invalid (>=0)
        "monthly_charges": 75.5,
        "total_charges": 906.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "support_tickets": 3,
        "payment_method": "Electronic check",
        "is_senior_citizen": 0,
    }
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_rejects_unknown_enum(client, sample_payload):
    bad = dict(sample_payload)
    bad["contract_type"] = "Lifetime"  # not in Literal options
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
