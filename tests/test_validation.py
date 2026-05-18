"""Tests for shared feature validation."""

from __future__ import annotations

import pytest

from src.validation import validate_feature_payload


def test_validate_feature_payload_accepts_valid_payload(sample_payload):
    normalized = validate_feature_payload(sample_payload)
    assert normalized["tenure_months"] == 12
    assert normalized["monthly_charges"] == 75.5


def test_validate_feature_payload_rejects_unknown_category(sample_payload):
    bad = dict(sample_payload)
    bad["payment_method"] = "Crypto"
    with pytest.raises(ValueError, match="payment_method"):
        validate_feature_payload(bad)


def test_validate_feature_payload_rejects_out_of_range_number(sample_payload):
    bad = dict(sample_payload)
    bad["monthly_charges"] = 5000
    with pytest.raises(ValueError, match="monthly_charges"):
        validate_feature_payload(bad)
