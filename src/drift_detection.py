"""Lightweight data drift detection.

This module compares reference data (the training distribution) to
recently logged prediction inputs and produces per-feature drift
scores. The implementation deliberately avoids heavy dependencies
(e.g. evidently, alibi-detect): it relies only on pandas/numpy so it
remains easy to deploy and reason about. For a production system you
would likely swap this out for a managed drift detection service such
as Vertex AI Model Monitoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config import SETTINGS
from src.utils import ensure_dir, get_logger


logger = get_logger(__name__)


def _drift_status(score: float, threshold: float) -> str:
    """Bucket a normalized drift score into low/medium/high."""
    if score < threshold:
        return "low"
    if score < threshold * 2:
        return "medium"
    return "high"


def _numeric_drift(
    reference: pd.Series, current: pd.Series, threshold: float
) -> Dict[str, Any]:
    """Compute drift for a single numeric feature.

    Uses a normalized absolute mean difference scaled by the reference
    standard deviation. A value of 1.0 means the means differ by one
    reference-standard-deviation.
    """
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)

    if len(cur) == 0 or len(ref) == 0:
        return {
            "feature": reference.name,
            "type": "numeric",
            "drift_score": 0.0,
            "status": "low",
            "details": {
                "reference_mean": float(ref.mean()) if len(ref) else None,
                "current_mean": None,
                "reference_std": float(ref.std()) if len(ref) > 1 else None,
                "current_std": None,
                "note": "Insufficient current data",
            },
        }

    ref_mean = float(ref.mean())
    cur_mean = float(cur.mean())
    ref_std = float(ref.std()) if len(ref) > 1 else 0.0
    cur_std = float(cur.std()) if len(cur) > 1 else 0.0

    denom = ref_std if ref_std > 1e-9 else max(abs(ref_mean), 1.0)
    score = float(abs(ref_mean - cur_mean) / denom)

    return {
        "feature": reference.name,
        "type": "numeric",
        "drift_score": round(score, 4),
        "status": _drift_status(score, threshold),
        "details": {
            "reference_mean": round(ref_mean, 4),
            "current_mean": round(cur_mean, 4),
            "reference_std": round(ref_std, 4),
            "current_std": round(cur_std, 4),
            "n_reference": int(len(ref)),
            "n_current": int(len(cur)),
        },
    }


def _categorical_drift(
    reference: pd.Series, current: pd.Series, threshold: float
) -> Dict[str, Any]:
    """Compute drift for a single categorical feature.

    Score is the total variation distance between the two category
    frequency distributions: 0.5 * sum(|p_ref - p_cur|), which lies in
    [0, 1]. Categories present in either side are taken into account.
    """
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)

    if len(cur) == 0 or len(ref) == 0:
        return {
            "feature": reference.name,
            "type": "categorical",
            "drift_score": 0.0,
            "status": "low",
            "details": {"note": "Insufficient data"},
        }

    ref_freq = ref.value_counts(normalize=True)
    cur_freq = cur.value_counts(normalize=True)
    all_cats = sorted(set(ref_freq.index) | set(cur_freq.index))

    diffs = {}
    score = 0.0
    for cat in all_cats:
        p_ref = float(ref_freq.get(cat, 0.0))
        p_cur = float(cur_freq.get(cat, 0.0))
        diffs[cat] = {
            "reference": round(p_ref, 4),
            "current": round(p_cur, 4),
        }
        score += abs(p_ref - p_cur)
    score *= 0.5

    return {
        "feature": reference.name,
        "type": "categorical",
        "drift_score": round(score, 4),
        "status": _drift_status(score, threshold),
        "details": {
            "n_reference": int(len(ref)),
            "n_current": int(len(cur)),
            "category_distribution": diffs,
        },
    }


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Run drift checks on all configured features.

    Parameters
    ----------
    reference_df:
        Snapshot of the training data (or another reference window).
    current_df:
        Recent prediction inputs.
    threshold:
        Drift threshold; defaults to ``SETTINGS.drift_threshold``.

    Returns
    -------
    list of dict
        One entry per feature with ``feature``, ``drift_score`` and
        ``status``.
    """
    thr = SETTINGS.drift_threshold if threshold is None else float(threshold)
    results: List[Dict[str, Any]] = []

    for feat in SETTINGS.numeric_features:
        if feat in reference_df.columns and feat in current_df.columns:
            results.append(_numeric_drift(reference_df[feat], current_df[feat], thr))

    for feat in SETTINGS.categorical_features:
        if feat in reference_df.columns and feat in current_df.columns:
            results.append(
                _categorical_drift(reference_df[feat], current_df[feat], thr)
            )

    return results


def overall_status(drift_results: List[Dict[str, Any]]) -> str:
    """Aggregate per-feature drift statuses into a single label."""
    if not drift_results:
        return "low"
    statuses = [r["status"] for r in drift_results]
    if "high" in statuses:
        return "high"
    if "medium" in statuses:
        return "medium"
    return "low"


def render_drift_report(
    drift_results: List[Dict[str, Any]], threshold: float | None = None
) -> str:
    """Render drift results as human-readable markdown."""
    thr = SETTINGS.drift_threshold if threshold is None else float(threshold)
    lines = [
        "# Drift Report",
        "",
        f"Drift threshold: **{thr:.2f}**",
        f"Overall status: **{overall_status(drift_results).upper()}**",
        "",
        "## Per-feature drift",
        "",
        "| Feature | Type | Drift score | Status |",
        "|---------|------|-------------|--------|",
    ]
    for r in drift_results:
        lines.append(
            f"| {r['feature']} | {r['type']} | "
            f"{r['drift_score']:.4f} | {r['status']} |"
        )
    lines.append("")
    lines.append("## Details")
    lines.append("")
    for r in drift_results:
        lines.append(f"### {r['feature']} ({r['type']})")
        lines.append("")
        for key, value in r["details"].items():
            if isinstance(value, dict):
                lines.append(f"- **{key}**:")
                for k, v in value.items():
                    lines.append(f"  - `{k}`: {v}")
            else:
                lines.append(f"- **{key}**: {value}")
        lines.append("")
    return "\n".join(lines)


def save_drift_report(
    drift_results: List[Dict[str, Any]],
    report_path: str | Path | None = None,
    threshold: float | None = None,
) -> Path:
    """Write the rendered drift report to ``reports/drift_report.md``."""
    path = (
        Path(report_path)
        if report_path is not None
        else SETTINGS.reports_dir / "drift_report.md"
    )
    ensure_dir(path.parent)
    md = render_drift_report(drift_results, threshold=threshold)
    path.write_text(md)
    logger.info("Saved drift report to %s", path)
    return path
