"""Entry-point: run the FastAPI service with uvicorn.

Usage::

    python run_api.py

Equivalent to::

    uvicorn api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload_flag = os.getenv("API_RELOAD", "true").lower() in {"1", "true", "yes"}
    uvicorn.run(
        "api.fastapi_app:app", host=host, port=port, reload=reload_flag
    )


if __name__ == "__main__":
    main()
