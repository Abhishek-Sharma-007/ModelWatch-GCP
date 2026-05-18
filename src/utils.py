"""Small utility helpers shared across the project."""

from __future__ import annotations

import logging
from pathlib import Path


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "modelwatch") -> logging.Logger:
    """Return a project logger configured once per process.

    Logging is intentionally light-weight: human-readable lines on
    stderr. In production on GCP, Cloud Logging will pick this up via
    the container's standard streams automatically.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return it as a :class:`pathlib.Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
