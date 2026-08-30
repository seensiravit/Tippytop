"""Randomised-exposure validation — headroom direction 7, and nobody's touched it.

``log_random_4_22_to_5_08_pure.csv`` holds ~1.18M impressions that KuaiRand served
**at random** rather than by the production ranker — a bias-free sample roughly
the size of the entire training split, shipped with the dataset and so far
completely unused by this project.

Why it is worth an iteration
----------------------------
Every score computed on the standard log is measured on traffic the logging
policy chose. A model can improve on that distribution by learning the policy's
own habits rather than the users' preferences, and the standard validation split
cannot tell the two apart, because it shares the bias. The randomised log can:
it is drawn from a different exposure distribution entirely, so agreement between
the two is evidence of genuine preference learning, and divergence is evidence of
policy overfitting.

This costs no new modelling — it is a second evaluation path over an existing
model. Of the organizers' seven listed directions it is the cheapest, and it is
the one that speaks most directly to research credibility in the write-up.

A date-window warning that matters
----------------------------------
The file spans 2022-04-22 to 2022-05-08, which covers **both** the valid window
(04-22 to 04-28) *and* the test window (04-29 to 05-08). Loading it wholesale
would pull randomised impressions from the hidden-test period into development.
This module therefore defaults to the **valid window only**, and reaching past it
requires passing ``window="test"`` explicitly — which the loop must never do.
"""
from __future__ import annotations

import csv
import os

from ..kit import SPLITS, encode, evaluate

RANDOM_LOG = "log_random_4_22_to_5_08_pure.csv"
_VIDEO_FEATURES = "video_features_basic_pure.csv"


def load_random_rows(data_dir, *, window: str = "valid") -> list[tuple]:
    """Randomised-exposure rows in the kit's row-tuple layout.

    ``window`` selects the date range: ``"valid"`` (default, safe) or ``"test"``
    (the hidden-test period — do not use during development). The returned tuples
    match ``(date, user_id, video_id, author_id, tab, duration_ms, long_view)``,
    so every model and metric in the project accepts them unchanged.
    """
    if window not in SPLITS:
        raise ValueError(f"window must be one of {sorted(SPLITS)}, got {window!r}")
    if window == "test":
        raise ValueError(
            "refusing to load the randomised log over the hidden-test window. "
            "Unbiased validation during development must use window='valid'."
        )

    data_dir = str(data_dir)
    path = os.path.join(data_dir, RANDOM_LOG)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{RANDOM_LOG} not found in {data_dir}. It ships with KuaiRand-Pure; "
            "re-run the download script if it is missing."
        )

    vid2author: dict[str, str] = {}
    vf = os.path.join(data_dir, _VIDEO_FEATURES)
    if os.path.exists(vf):
        with open(vf, newline="") as fh:
            for r in csv.DictReader(fh):
                vid2author[r["video_id"]] = r["author_id"]

    lo, hi = SPLITS[window]
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d = int(r["date"])
            if not (lo <= d <= hi):
                continue
            rows.append((
                d, r["user_id"], r["video_id"],
                vid2author.get(r["video_id"], "UNK"), r["tab"],
                float(r["duration_ms"]), 1 if r.get("long_view", "0") != "0" else 0,
            ))
    return rows


def encode_against_train(train_rows: list[tuple], random_rows: list[tuple]):
    """Encode randomised rows with the *training* vocabulary.

    The kit's ``encode`` builds its vocabularies from ``splits['train']`` and maps
    anything unseen to that field's UNK slot, so passing both splits together
    yields an encoding consistent with the model's embedding table.

    Returns ``(X, y, users, dim)``.
    """
    enc, dim = encode({"train": train_rows, "random": random_rows})
    X, y, users = enc["random"]
    return X, y, users, dim


def evaluate_unbiased(model, train_rows: list[tuple], random_rows: list[tuple]) -> dict:
    """Score a fitted model on randomised exposure.

    ``model`` needs only a ``predict_encoded(X)`` method (or a plain callable
    taking the encoded matrix), keeping this independent of the model registry.
    """
    X, y, users, _ = encode_against_train(train_rows, random_rows)
    if hasattr(model, "predict_encoded"):
        scores = model.predict_encoded(X)
    elif callable(model):
        scores = model(X)
    else:
        raise TypeError(
            "model must expose predict_encoded(X) or be callable on the encoded matrix"
        )
    return evaluate(users, y, scores)


def agreement_report(biased: dict, unbiased: dict) -> dict:
    """Compare standard-log and randomised-log metrics for one model.

    A model that gains on the logged distribution but not on randomised exposure
    has learned the logging policy, not the user. That is a finding worth
    reporting either way — including when it is inconvenient.
    """
    bp, up = biased.get("primary"), unbiased.get("primary")
    gap = None if (bp is None or up is None) else bp - up
    return {
        "biased_primary": bp,
        "unbiased_primary": up,
        "gap": gap,
        "interpretation": (
            "n/a" if gap is None
            else "consistent across exposure distributions" if abs(gap) < 0.01
            else "scores materially higher on logged traffic — check for policy overfitting"
            if gap > 0 else "scores higher on randomised traffic than logged traffic"
        ),
    }
