"""Robust final selection — submit an average over the top, not the argmax.

The problem with taking the best
--------------------------------
The scored submission is the validation-best checkpoint. In a regime where every
candidate sits within one to three sigma of every other, an argmax over
validation is not model selection — it is **selection overfitting**. The winner is
the model whose noise happened to point upward, and that component of its score
does not reproduce on the test split.

This is the most likely mechanism behind the project's own observation that a
+0.0011 validation gain became +0.0003 on test.

The fix costs nothing
---------------------
Averaging over the top-K instead of taking the maximum is strictly more robust:
the idiosyncratic noise that lifted any single candidate is diluted, while the
shared signal survives. It requires no retraining — only predictions that already
exist — and the project's own data already points this way, since both of its best
test scores came from ensembles rather than single models.

Combining is by **within-user rank**, matching ``models/ensemble.py``. Score
scales differ between a logloss-trained and a softmax-trained model, and only
within-user order is scored, so ranks are the only defensible common currency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import PerUserStats, per_user_stats, primary_from, paired_bootstrap


def rank_within_user(scores: np.ndarray, users) -> np.ndarray:
    """Within-user rank, normalised to [0, 1]. Matches ``models/ensemble.py``."""
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users)
    out = np.empty(len(scores), dtype=np.float64)

    order = np.argsort(users, kind="stable")
    srt_u = users[order]
    starts = np.flatnonzero(np.r_[True, srt_u[1:] != srt_u[:-1]])
    lengths = np.diff(np.r_[starts, len(srt_u)])

    for s, ln in zip(starts, lengths):
        rows = order[s:s + ln]
        if ln == 1:
            out[rows] = 0.5
            continue
        o = np.argsort(scores[rows], kind="stable")
        r = np.empty(ln, dtype=np.float64)
        r[o] = np.arange(ln, dtype=np.float64)
        out[rows] = r / (ln - 1)
    return out


def rank_average(score_list: list[np.ndarray], users) -> np.ndarray:
    """Mean of within-user ranks across members."""
    if not score_list:
        raise ValueError("need at least one member to average")
    return np.mean([rank_within_user(s, users) for s in score_list], axis=0)


@dataclass
class SelectionResult:
    method: str
    members: list[str]
    valid_primary: float
    scores: np.ndarray

    def summary(self) -> str:
        return (f"{self.method}: valid primary {self.valid_primary:.4f} "
                f"from {len(self.members)} member(s) [{', '.join(self.members)}]")


def topk_selection(named_scores: dict[str, np.ndarray], user_ids, labels, *,
                   k: int = 3, min_k: int = 1) -> SelectionResult:
    """Rank-average the top-``k`` models by validation primary.

    ``named_scores`` maps a model name to its validation-split scores. The
    returned ``scores`` is the ensemble's combined ranking, ready to be written as
    a submission — or recomputed on the test split by passing that split's member
    predictions instead.
    """
    if not named_scores:
        raise ValueError("no candidates supplied")
    k = max(min_k, min(k, len(named_scores)))

    ranked = sorted(
        ((n, primary_from(per_user_stats(user_ids, labels, s)), s)
         for n, s in named_scores.items()),
        key=lambda t: t[1], reverse=True)

    members = [n for n, _, _ in ranked[:k]]
    combined = rank_average([s for _, _, s in ranked[:k]], user_ids)
    return SelectionResult(
        method=f"top-{k} rank-average",
        members=members,
        valid_primary=primary_from(per_user_stats(user_ids, labels, combined)),
        scores=combined,
    )


def compare_argmax_vs_topk(named_scores: dict[str, np.ndarray], user_ids, labels,
                           *, k: int = 3, n_boot: int = 2000, seed: int = 0) -> dict:
    """Is the top-K ensemble actually better than the single best on validation?

    An important caveat, stated because it is easy to misread this number: the
    ensemble is *expected* to look similar to or slightly worse than the argmax
    **on validation**, because the argmax was chosen by maximising exactly that
    quantity. The ensemble's advantage is that it does not carry the selection
    noise, so it should hold up better on test. Comparing the two on validation
    measures the selection bias, not the ensemble's quality.
    """
    best_name, best_scores = max(
        named_scores.items(),
        key=lambda kv: primary_from(per_user_stats(user_ids, labels, kv[1])))
    top = topk_selection(named_scores, user_ids, labels, k=k)

    a = per_user_stats(user_ids, labels, top.scores)
    b = per_user_stats(user_ids, labels, best_scores)
    boot = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)

    return {
        "argmax_model": best_name,
        "argmax_valid": primary_from(b),
        "topk_members": top.members,
        "topk_valid": top.valid_primary,
        "delta_on_valid": boot["delta"],
        "bootstrap": boot,
        "note": ("A small negative delta on validation is expected and is not a "
                 "reason to prefer the argmax: the argmax was selected by "
                 "maximising validation, so it carries selection noise the "
                 "ensemble does not. Prefer the ensemble for the final submission."),
    }
