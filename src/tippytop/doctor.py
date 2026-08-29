"""Preflight checks for data, evaluator, storage, and model service."""

from __future__ import annotations

import csv
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig
from .llm import LLMClient
from .starter import STARTER_DIR


DATA_REQUIREMENTS: dict[str, set[str]] = {
    "log_standard_4_08_to_4_21_pure.csv": {
        "date",
        "user_id",
        "video_id",
        "tab",
        "duration_ms",
        "long_view",
    },
    "log_standard_4_22_to_5_08_pure.csv": {
        "date",
        "user_id",
        "video_id",
        "tab",
        "duration_ms",
        "long_view",
    },
    "video_features_basic_pure.csv": {"video_id", "author_id"},
}
ORGANIZER_SHA256 = {
    "data.py": "501624a23c78a9ae23446af876e3fa94abd37c441cdb1e0be26a9be8f6eaf1b3",
    "evaluate.py": "a509ee68e4f91c536c6b286c6d7e3873fe4b6dc51b70d1023cc24f575db72676",
    "submit.py": "d3bda5a41ee9a555aad3eb162c62cdd498df42aaac69a8e6da94d0c9b1a9f59c",
}


def run_doctor(config: RunConfig, *, check_llm: bool = True) -> dict[str, Any]:
    config.validate()
    checks: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "data_dir": str(config.data_dir),
    }
    errors: list[str] = []
    if sys.version_info < (3, 11):
        errors.append("Python 3.11 or newer is required")

    for filename, required_columns in DATA_REQUIREMENTS.items():
        path = config.data_dir / filename
        if not path.is_file():
            errors.append(f"missing data file: {path}")
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                columns = set(next(csv.reader(handle)))
        except (OSError, StopIteration) as error:
            errors.append(f"cannot read {path}: {error}")
            continue
        missing = required_columns - columns
        if missing:
            errors.append(f"{path} is missing columns: {sorted(missing)}")

    organizer_hashes: dict[str, str] = {}
    for filename, expected_hash in ORGANIZER_SHA256.items():
        path = STARTER_DIR / filename
        if not path.is_file():
            errors.append(f"missing frozen organizer file: {path}")
            continue
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        organizer_hashes[filename] = actual_hash
        if actual_hash != expected_hash:
            errors.append(
                f"frozen organizer file hash mismatch for {filename}: "
                f"expected {expected_hash}, found {actual_hash}"
            )
    checks["organizer_sha256"] = organizer_hashes
    checks["evaluator_sha256"] = organizer_hashes.get("evaluate.py")

    storage_target = config.runs_dir.parent if not config.runs_dir.exists() else config.runs_dir
    storage_target.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(storage_target).free
    checks["free_disk_bytes"] = free_bytes
    if free_bytes < 1_000_000_000:
        errors.append("less than 1 GB of free disk space is available")

    if not config.offline:
        bubblewrap = shutil.which("bwrap")
        checks["bubblewrap"] = bubblewrap
        if bubblewrap is None:
            errors.append("bubblewrap (bwrap) is required for generated-code execution")

    if check_llm:
        try:
            models = LLMClient(config).list_models()
            checks["available_models"] = models
            if config.model not in models:
                errors.append(f"configured model {config.model!r} is not available")
        except ConnectionError as error:
            errors.append(str(error))

    checks["ok"] = not errors
    checks["errors"] = errors
    return checks


def format_doctor(checks: dict[str, Any]) -> str:
    lines = [
        f"Python: {checks['python']}",
        f"NumPy: {checks['numpy']}",
        f"Data: {checks['data_dir']}",
        f"Disk free: {checks['free_disk_bytes'] / 1_000_000_000:.1f} GB",
    ]
    if "available_models" in checks:
        lines.append(f"LLM models: {', '.join(checks['available_models'])}")
    if checks.get("bubblewrap"):
        lines.append(f"Sandbox: {checks['bubblewrap']}")
    if checks["ok"]:
        lines.append("Status: ready")
    else:
        lines.append("Status: failed")
        lines.extend(f"- {error}" for error in checks["errors"])
    return "\n".join(lines)
