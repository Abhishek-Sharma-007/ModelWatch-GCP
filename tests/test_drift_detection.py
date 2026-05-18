"""Tests for src.drift_detection."""

from __future__ import annotations

import pandas as pd

from src.data_loader import load_raw_data


def test_drift_status_low_for_identical_distributions():
    from src.drift_detection import detect_drift, overall_status

    df = load_raw_data()
    results = detect_drift(df, df.copy())
    for r in results:
        assert r["status"] == "low"
    assert overall_status(results) == "low"


def test_drift_status_high_for_shifted_numeric():
    from src.drift_detection import detect_drift, overall_status

    df = load_raw_data()
    shifted = df.copy()
    # Shift numeric distribution dramatically.
    shifted["monthly_charges"] = shifted["monthly_charges"] + 200
    results = detect_drift(df, shifted)
    statuses = {r["feature"]: r["status"] for r in results}
    assert statuses["monthly_charges"] in {"medium", "high"}
    assert overall_status(results) in {"medium", "high"}


def test_drift_status_high_for_shifted_categorical():
    from src.drift_detection import detect_drift

    df = load_raw_data()
    shifted = df.copy()
    shifted["contract_type"] = "Two year"  # collapse category distribution
    results = detect_drift(df, shifted)
    statuses = {r["feature"]: r["status"] for r in results}
    assert statuses["contract_type"] in {"medium", "high"}


def test_render_and_save_drift_report(tmp_path):
    from src.drift_detection import (
        detect_drift,
        render_drift_report,
        save_drift_report,
    )

    df = load_raw_data()
    results = detect_drift(df, df.sample(50, random_state=0))
    md = render_drift_report(results)
    assert "Drift Report" in md
    path = save_drift_report(results, report_path=tmp_path / "drift.md")
    assert path.exists()
    assert path.read_text().startswith("# Drift Report")


def test_drift_result_schema():
    from src.drift_detection import detect_drift

    df = load_raw_data()
    results = detect_drift(df, df.sample(100, random_state=0))
    assert len(results) > 0
    for r in results:
        assert "feature" in r
        assert "drift_score" in r
        assert r["status"] in {"low", "medium", "high"}
