"""Preprocessing pipeline for the churn dataset.

Builds a single ``sklearn.compose.ColumnTransformer`` that handles
numeric imputation + scaling and categorical imputation + one-hot
encoding. The transformer is intentionally kept as part of the final
model pipeline so the same logic runs during training and inference.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import SETTINGS


def _make_one_hot_encoder() -> OneHotEncoder:
    """Build a OneHotEncoder compatible with both new and old scikit-learn.

    ``sparse_output`` replaced ``sparse`` in scikit-learn 1.2. We try the
    new keyword first and fall back to the old one so the project works
    on a wider range of installed versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - legacy sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, SETTINGS.numeric_features),
            ("cat", categorical_pipeline, SETTINGS.categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor
