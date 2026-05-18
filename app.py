"""Streamlit Cloud entrypoint.

Streamlit Community Cloud and many managed Streamlit runners default to a
root-level ``app.py``. The real dashboard lives in ``ui/streamlit_dashboard.py``
so imports stay organized; this file simply imports that module and lets its
router render the selected page.
"""

from __future__ import annotations

import ui.streamlit_dashboard  # noqa: F401
