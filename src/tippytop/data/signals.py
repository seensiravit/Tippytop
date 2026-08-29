"""Auxiliary log columns the frozen kit drops.

``data.py`` keeps 7 of the log's 19 columns:

    (date, user_id, video_id, author_id, tab, duration_ms, long_view)

Everything else is discarded at load — including all eleven other feedback
signals. So every model in this repo so far, however different its architecture,
has been a different way of squeezing the same seven columns. That is why they
all land within ~1.4% of each other.

This module reads the same two files, in the same order, with the same date
filter, and returns the dropped columns **aligned row-for-row** with
``kit.load()``. Alignment is asserted, not assumed: a silent off-by-one here
would attach every row's auxiliary signals to a different impression and would
not show up as an error anywhere downstream — only as a model that mysteriously
fails to learn.

Nothing here modifies the kit. The label and the splits still come from it.

TRAIN-ONLY. These are outcomes, not inputs: they are observed *after* the
impression, so using them at prediction time would be leakage. They are legal
as **auxiliary training targets** (multi-task supervision) and as the basis for
train-derived aggregates. ``hourmin``, ``time_ms`` and ``is_rand`` are the
exceptions — those are known at impression time and may be used as features.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from ..kit import SPLITS

_FILES = ("log_standard_4_08_to_4_21_pure.csv",
          "log_standard_4_22_to_5_08_pure.csv")

# Observed after the impression — training targets only, never features.
OUTCOME_COLUMNS = [
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "play_time_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter",
]
# Known at impression time — usable as features.
CONTEXT_COLUMNS = ["hourmin", "time_ms", "is_rand"]

ALL_COLUMNS = OUTCOME_COLUMNS + CONTEXT_COLUMNS


def load_aux(data_dir) -> dict[str, dict[str, np.ndarray]]:
    """Return {split: {column: float32 array}}, aligned with ``kit.load()``.

    Also returns ``user_id`` / ``video_id`` / ``date`` per split so callers can
    verify alignment — see :func:`assert_aligned`.
    """
    buckets: dict[str, dict[str, list]] = {
        s: {c: [] for c in ALL_COLUMNS + ["user_id", "video_id", "date"]}
        for s in SPLITS
    }

    for fname in _FILES:
        with open(os.path.join(str(data_dir), fname)) as fh:
            for r in csv.DictReader(fh):
                date = int(r["date"])
                for split, (lo, hi) in SPLITS.items():
                    if lo <= date <= hi:
                        b = buckets[split]
                        for c in ALL_COLUMNS:
                            b[c].append(float(r[c]))
                        b["user_id"].append(r["user_id"])
                        b["video_id"].append(r["video_id"])
                        b["date"].append(date)
                        break

    out: dict[str, dict[str, np.ndarray]] = {}
    for split, cols in buckets.items():
        out[split] = {
            c: (np.asarray(v, dtype=np.float32) if c not in ("user_id", "video_id")
                else np.asarray(v))
            for c, v in cols.items()
        }
    return out


def assert_aligned(aux: dict, splits: dict) -> None:
    """Fail loudly if the auxiliary rows do not match the kit's row-for-row."""
    for name, rows in splits.items():
        a = aux[name]
        if len(a["user_id"]) != len(rows):
            raise AssertionError(
                f"{name}: {len(a['user_id']):,} aux rows vs {len(rows):,} kit rows"
            )
        kit_u = np.asarray([r[1] for r in rows])
        kit_v = np.asarray([r[2] for r in rows])
        if not (np.array_equal(a["user_id"], kit_u)
                and np.array_equal(a["video_id"], kit_v)):
            bad = int(np.argmax((a["user_id"] != kit_u) | (a["video_id"] != kit_v)))
            raise AssertionError(
                f"{name}: aux misaligned with kit at row {bad} — "
                f"aux=({a['user_id'][bad]}, {a['video_id'][bad]}) "
                f"kit=({kit_u[bad]}, {kit_v[bad]})"
            )


def watch_quantile(aux_split: dict, duration_ms: np.ndarray,
                   n_buckets: int = 10) -> np.ndarray:
    """``play_time_ms`` as its quantile *within its duration bucket*, in [0, 1].

    This is the D2Q idea (Zhan et al., KDD 2022, arXiv:2206.06003) adapted as an
    auxiliary target. Raw watch time is dominated by video length: a 10s video
    watched fully scores less watch time than a 200s video abandoned early, so
    regressing it directly teaches the model duration, not interest. Measured on
    this data, ``duration_log`` alone scores **below random** (0.4730 valid
    primary) — the bias is strong and in the obvious direction.

    Ranking watch time *within* videos of similar length strips that out: the
    target becomes "did this user stay longer than other people stay on videos
    this long", which is the interest signal underneath.
    """
    play = aux_split["play_time_ms"]
    edges = np.quantile(duration_ms, np.linspace(0, 1, n_buckets + 1)[1:-1])
    bucket = np.searchsorted(edges, duration_ms)

    q = np.zeros(len(play), dtype=np.float32)
    for b in range(n_buckets):
        idx = np.flatnonzero(bucket == b)
        if len(idx) < 2:
            continue
        order = np.argsort(play[idx], kind="stable")
        ranks = np.empty(len(idx), dtype=np.float32)
        ranks[order] = np.arange(len(idx), dtype=np.float32)
        q[idx] = ranks / (len(idx) - 1)
    return q
