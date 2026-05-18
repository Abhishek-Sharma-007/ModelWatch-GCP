"""Lightweight repository security checks.

This script is intentionally dependency-free so it can run in CI and on
fresh clones. It scans tracked project files for common secret patterns while
skipping generated artifacts, caches, Git metadata and sample CSV data.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "__pycache__",
    "data/raw",
}
SKIP_SUFFIXES = {".pyc", ".joblib", ".png", ".jpg", ".jpeg", ".gif", ".pdf"}
PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "Private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "OpenAI-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Assigned secret": re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
}


def _is_skipped(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    rel_parts = rel.parts
    if any(part in SKIP_DIRS for part in rel_parts):
        return True
    if any(str(rel).startswith(skip + "/") for skip in SKIP_DIRS if "/" in skip):
        return True
    return path.suffix.lower() in SKIP_SUFFIXES


def iter_files() -> list[Path]:
    return [path for path in ROOT.rglob("*") if path.is_file() and not _is_skipped(path)]


def main() -> int:
    findings: list[str] = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    rel = path.relative_to(ROOT)
                    findings.append(f"{rel}:{line_no}: possible {name}")

    if findings:
        print("Potential secrets found:")
        print("\n".join(findings))
        return 1
    print("No obvious secrets found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
