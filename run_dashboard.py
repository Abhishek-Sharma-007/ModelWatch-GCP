"""Entry-point: launch the Streamlit monitoring dashboard.

Usage::

    python run_dashboard.py

Equivalent to::

    streamlit run ui/streamlit_dashboard.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    dashboard = Path(__file__).parent / "ui" / "streamlit_dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard)]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
