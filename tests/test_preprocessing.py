"""Tests for src.preprocessing."""

from __future__ import annotations

import numpy as np

from src.data_loader import load_raw_data, train_test_split_data
from src.preprocessing import build_preprocessor


def test_preprocessor_fits_and_transforms():
    df = load_raw_data()
    X_train, _, _, _ = train_test_split_data(df)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X_train)
    arr = np.asarray(transformed)

    assert arr.shape[0] == X_train.shape[0]
    # 5 numeric features + at least 3+3+4 one-hot expanded categoricals
    assert arr.shape[1] >= 5 + 3 + 3 + 4 - 3  # categories may collapse
    assert not np.isnan(arr).any()


def test_preprocessor_handles_unknown_category():
    df = load_raw_data()
    X_train, X_test, _, _ = train_test_split_data(df)
    pre = build_preprocessor()
    pre.fit(X_train)
    X_test = X_test.copy()
    X_test.loc[X_test.index[0], "payment_method"] = "Crypto"  # unseen
    out = pre.transform(X_test)
    assert np.asarray(out).shape[0] == X_test.shape[0]
