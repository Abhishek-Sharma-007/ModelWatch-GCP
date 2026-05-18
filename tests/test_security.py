"""Tests for security helpers."""

from __future__ import annotations

import asyncio
import importlib

import pytest
from fastapi import HTTPException


def test_require_api_key_allows_matching_secret(monkeypatch):
    monkeypatch.setenv("API_KEY", "super-secret-test-key")

    import src.config as cfg
    import src.security as security

    importlib.reload(cfg)
    importlib.reload(security)

    asyncio.run(security.require_api_key("super-secret-test-key"))

    monkeypatch.delenv("API_KEY", raising=False)
    importlib.reload(cfg)
    importlib.reload(security)


def test_require_api_key_rejects_missing_secret(monkeypatch):
    monkeypatch.setenv("API_KEY", "super-secret-test-key")

    import src.config as cfg
    import src.security as security

    importlib.reload(cfg)
    importlib.reload(security)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(security.require_api_key(None))
    assert exc_info.value.status_code == 401

    monkeypatch.delenv("API_KEY", raising=False)
    importlib.reload(cfg)
    importlib.reload(security)
