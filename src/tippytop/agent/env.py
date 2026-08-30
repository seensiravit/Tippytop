"""Minimal .env loader (stdlib only, no python-dotenv dependency).

Reads KEY=VALUE lines from the repo-root .env into os.environ WITHOUT overriding
values already set in the real environment (the shell wins over the file).
"""
from __future__ import annotations
import os
from pathlib import Path

from ..kit import REPO_ROOT


def load_dotenv(path: str | Path | None = None) -> dict:
    """Load .env into os.environ (no override). Returns the parsed dict.

    Missing file is fine (returns {}). Lines: `KEY=VALUE`, `#` comments, blanks;
    surrounding quotes on the value are stripped.
    """
    p = Path(path) if path else (REPO_ROOT / ".env")
    parsed: dict[str, str] = {}
    if not p.exists():
        return parsed
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        parsed[key] = val
        os.environ.setdefault(key, val)      # shell env takes precedence
    return parsed
