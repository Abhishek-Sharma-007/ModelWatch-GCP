"""Tests for src.data_loader."""

from __future__ import annotations

import pandas as pd

from src.data_loader import (
    EXPECTED_COLUMNS,
    get_feature_target_split,
    load_raw_data,
    train_test_split_data,
)


def test_load_raw_data_returns_dataframe():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in EXPECTED_COLUMNS:
        assert col in df.columns


def test_feature_target_split_shapes():
    df = load_raw_data()
    X, y = get_feature_target_split(df)
    assert "customer_id" not in X.columns
    assert "churn" not in X.columns
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})


def test_train_test_split_is_stratified():
    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_data(df, test_size=0.2)
    assert len(X_train) > len(X_test)
    # Stratification: churn rate should be similar in both splits.
    rate_train = y_train.mean()
    rate_test = y_test.mean()
    assert abs(rate_train - rate_test) < 0.05
