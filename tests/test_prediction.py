"""Tests for src.predict."""

from __future__ import annotations


def test_predict_one_returns_valid_schema(trained_model_path, sample_payload):
    from src.predict import predict_one

    result = predict_one(sample_payload)
    assert set(result.keys()) == {
        "prediction",
        "prediction_label",
        "churn_probability",
        "model_version",
    }
    assert result["prediction"] in {0, 1}
    assert result["prediction_label"] in {"Churn", "No Churn"}
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_predict_one_missing_feature_raises(trained_model_path, sample_payload):
    import pytest

    from src.predict import predict_one

    bad = dict(sample_payload)
    bad.pop("monthly_charges")
    with pytest.raises(ValueError):
        predict_one(bad)


def test_predict_logs_prediction(trained_model_path, sample_payload, tmp_path):
    from src.logger import log_prediction, read_prediction_logs
    from src.predict import predict_one

    log_path = tmp_path / "logs.csv"
    result = predict_one(sample_payload)
    log_prediction(sample_payload, result, log_path=log_path)
    df = read_prediction_logs(log_path=log_path)
    assert len(df) == 1
    assert df.iloc[0]["prediction_label"] in {"Churn", "No Churn"}
    assert "timestamp" in df.columns
