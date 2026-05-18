"""Shared pytest fixtures.

We train a small model once per test session to back the prediction
and API tests, and we route all logs / model artifacts written during
tests into a temporary directory so the developer's working tree is
not polluted.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Make ``import src.*`` work when pytest is invoked from any directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _isolate_runtime_paths(tmp_path_factory):
    """Redirect model + log paths to a session-scoped temp dir.

    This must run before importing src.config in test modules, so we
    set the env vars before the import is observed. We then re-load
    settings.
    """
    tmp_dir = tmp_path_factory.mktemp("modelwatch_test")
    model_path = tmp_dir / "churn_model.joblib"
    log_path = tmp_dir / "prediction_logs.csv"
    reports_dir = tmp_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["PREDICTION_LOG_PATH"] = str(log_path)
    os.environ["REPORTS_DIR"] = str(reports_dir)
    # Keep raw data path at its default (project ships the CSV).

    # Reload config to pick up the patched env vars.
    import importlib

    import src.config as cfg

    importlib.reload(cfg)

    # Also reload modules that captured ``SETTINGS`` at import time.
    for mod_name in [
        "src.logger",
        "src.predict",
        "src.train_model",
        "src.drift_detection",
        "src.monitoring",
        "src.data_loader",
        "src.preprocessing",
    ]:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    yield tmp_dir


@pytest.fixture(scope="session")
def trained_model_path(_isolate_runtime_paths):
    """Train the model once for the whole test session."""
    from src.train_model import save_artifacts, train_and_select

    pipeline, best_name, all_metrics = train_and_select()
    return save_artifacts(pipeline, best_name, all_metrics)


@pytest.fixture()
def sample_payload() -> dict:
    return {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "total_charges": 906.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "support_tickets": 3,
        "payment_method": "Electronic check",
        "is_senior_citizen": 0,
    }
