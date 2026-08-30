"""Score a solution's submission CSV against a split's rows (via the frozen kit)."""
from __future__ import annotations

from ..kit import evaluate
from ..submission import read_submission


def score_submission(out_path, rows) -> dict:
    """Validate the CSV and score it. Returns evaluate()'s metrics dict.

    read_submission enforces header / consecutive row_id / (user,video) alignment /
    exact count / no NaN-Inf, so a mis-shaped output raises a readable ValueError
    (which the orchestrator turns into a recovery event, not a crash).
    """
    scores = read_submission(out_path, rows)
    return evaluate([r[1] for r in rows], [r[6] for r in rows], scores)
