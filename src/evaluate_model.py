"""Model evaluation utilities.

Computes classification metrics that are useful for an imbalanced
binary classification problem like churn prediction.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, Any]:
    """Compute classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0/1).
    y_pred:
        Predicted labels (0/1).
    y_proba:
        Predicted probability of the positive class.

    Returns
    -------
    dict
        Metrics with keys ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``roc_auc`` and ``confusion_matrix`` (as a nested list).
    """
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm,
    }


def format_metrics_markdown(metrics: Dict[str, Any], model_name: str) -> str:
    """Format metrics as a markdown report fragment."""
    cm = metrics["confusion_matrix"]
    lines = [
        f"## Model: {model_name}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy  | {metrics['accuracy']:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall    | {metrics['recall']:.4f} |",
        f"| F1-score  | {metrics['f1']:.4f} |",
        f"| ROC-AUC   | {metrics['roc_auc']:.4f} |",
        "",
        "### Confusion matrix",
        "",
        "| | Pred: No Churn | Pred: Churn |",
        "|---|---|---|",
        f"| **Actual: No Churn** | {cm[0][0]} | {cm[0][1]} |",
        f"| **Actual: Churn**    | {cm[1][0]} | {cm[1][1]} |",
        "",
    ]
    return "\n".join(lines)
