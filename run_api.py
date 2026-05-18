"""Entry-point: run the FastAPI service with uvicorn.

Usage::

    python run_api.py

Equivalent to::

    uvicorn api.fastapi_app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import uvicorn

from src.config import SETTINGS


def main() -> None:
    uvicorn.run(
        "api.fastapi_app:app",
        host=SETTINGS.api_host,
        port=SETTINGS.api_port,
        reload=SETTINGS.api_reload,
    )


if __name__ == "__main__":
    main()
