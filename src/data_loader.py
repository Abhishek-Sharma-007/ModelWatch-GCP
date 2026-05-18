"""Data loading utilities for the customer churn dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SETTINGS


EXPECTED_COLUMNS = [
    "customer_id",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "internet_service",
    "support_tickets",
    "payment_method",
    "is_senior_citizen",
    "churn",
]


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw customer churn CSV into a DataFrame.

    Parameters
    ----------
    path:
        Optional path override. Defaults to ``SETTINGS.raw_data_path``.

    Returns
    -------
    pd.DataFrame
        The full raw dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If expected columns are missing from the file.
    """
    csv_path = Path(path) if path is not None else SETTINGS.raw_data_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {csv_path}. "
            "Make sure data/raw/customer_churn_sample.csv is present."
        )

    df = pd.read_csv(csv_path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def get_feature_target_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into model features (X) and target (y).

    The ``customer_id`` column is dropped because it is an identifier and
    must never be used as a model feature.
    """
    feature_cols = SETTINGS.numeric_features + SETTINGS.categorical_features
    X = df[feature_cols].copy()
    y = df[SETTINGS.target].astype(int).copy()
    return X, y


def train_test_split_data(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the churn label."""
    X, y = get_feature_target_split(df)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=SETTINGS.random_state,
        stratify=y,
    )
