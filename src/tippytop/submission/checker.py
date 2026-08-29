"""Submission CSV rendering and organizer checker integration."""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from ..artifacts import atomic_write_text
from ..starter import STARTER_DIR


HEADER = ["row_id", "user_id", "video_id", "score"]


def write_submission(path: Path, rows: Sequence[tuple[Any, ...]], scores: Sequence[float]) -> None:
    if len(rows) != len(scores):
        raise ValueError(f"row and score counts differ: {len(rows)} != {len(scores)}")
    rendered = io.StringIO(newline="")
    writer = csv.writer(rendered)
    writer.writerow(HEADER)
    # row_id is the original split order, which the organizer checker requires exactly.
    for row_id, (row, score) in enumerate(zip(rows, scores, strict=True)):
        value = float(score)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"score at row {row_id} is not finite")
        writer.writerow([row_id, row[1], row[2], f"{value:.9g}"])
    atomic_write_text(path, rendered.getvalue())


def validate_with_starter(path: Path, data_dir: Path) -> str:
    command = [
        sys.executable,
        str(STARTER_DIR / "submit.py"),
        str(path.resolve()),
        "--data_dir",
        str(data_dir.resolve()),
        "--split",
        "test",
        "--check",
    ]
    completed = subprocess.run(
        command,
        cwd=STARTER_DIR,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"starter submission validation failed: {completed.stderr or completed.stdout}")
    return completed.stdout.strip()
