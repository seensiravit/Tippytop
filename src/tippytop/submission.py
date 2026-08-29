"""Build & validate submission CSVs.

Format (frozen): header ``row_id,user_id,video_id,score``, one row per eval row,
in exact ``data.load()[split]`` order. ``row_id`` is mandatory — (user_id,
video_id) is NOT unique in the eval set. ``write_submission`` /
``read_submission`` mirror the kit's ``submit.py`` byte-for-byte on format.
"""
from __future__ import annotations
import csv
from pathlib import Path

HEADER = ["row_id", "user_id", "video_id", "score"]


def write_submission(path, rows, scores) -> Path:
    """rows: kit eval rows (row[1]=user_id, row[2]=video_id); scores: aligned."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])
    return path


def read_submission(path, rows) -> list:
    """Validate alignment row by row, return scores. Mismatch => readable error.

    Ported from the kit's submit.read_submission: checks header, consecutive
    row_id from 0, (user_id, video_id) alignment, row count, and rejects
    non-numeric / NaN / Inf scores.
    """
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        head = next(r, None)
        if head != HEADER:
            raise ValueError(f"header must be {','.join(HEADER)}, got {head}")
        scores, n = [], 0
        for ln, rec in enumerate(r, start=2):
            if len(rec) != 4:
                raise ValueError(f"line {ln} has {len(rec)} fields, expected 4")
            rid, uid, vid, sc = rec
            if int(rid) != n:
                raise ValueError(f"line {ln} has row_id={rid}, expected {n} "
                                 f"(must increase consecutively from 0)")
            if n >= len(rows):
                raise ValueError(f"submission has more rows than the eval set "
                                 f"(eval set has {len(rows)} rows)")
            if uid != rows[n][1] or vid != rows[n][2]:
                raise ValueError(f"line {ln} alignment error: submission has "
                                 f"({uid},{vid}), eval set row {n} is "
                                 f"({rows[n][1]},{rows[n][2]})")
            try:
                v = float(sc)
            except ValueError:
                raise ValueError(f"line {ln} score not a number: {sc!r}")
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError(f"line {ln} score is NaN/Inf, not allowed")
            scores.append(v)
            n += 1
    if n != len(rows):
        raise ValueError(f"submission has {n} rows, eval set has {len(rows)}, "
                         f"count mismatch")
    return scores
