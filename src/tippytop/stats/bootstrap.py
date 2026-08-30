"""Paired bootstrap over users — a confidence interval for free.

Why this exists
---------------
Every comparison in this project has been a difference between two scalars, each
carrying unquantified uncertainty. With a seed sigma of ~0.0006 against a 0.002
decision threshold, the standard error of a difference between two single-seed
runs is sigma*sqrt(2) ~ 0.00085 — so a reported "+0.0010" sits at z = 1.2 and is
not distinguishable from noise.

There are two independent sources of that uncertainty and they need different
treatments:

* **Initialization noise** — the same recipe with a different seed gives a
  different model. Only repeated training measures this. See ``stats.power``.
* **Evaluation noise** — the eval split is a *sample* of ~24k users, and the
  metric would move if you had drawn different users. This module measures that,
  and it requires **no retraining at all**.

The key property is pairing. Both models are scored on the *same* users, so when
you resample users and take the difference, user-level difficulty largely
cancels. That makes a paired interval far tighter than comparing two independent
scalars — which is why this is worth doing even when you also have seeds.

Why it is fast
--------------
Naively you would re-run ``evaluate`` on each resampled set: ~24k users x 2000
replicates of pure-Python metric code, which is far too slow to use routinely.

Instead this decomposes the metric into **per-user contributions once**, then
resamples those. The decomposition is exact rather than an approximation, because
both components of the primary score are averages over users:

    nDCG@5 = mean over ALL users of ndcg_u
    GAUC   = sum(npos_u * auc_u) / sum(npos_u), over eligible users only
    primary = (GAUC + nDCG@5) / 2

Resampling the per-user table and recomputing those two expressions reproduces
``evaluate`` exactly on the resampled set. ``verify_against_kit`` asserts this.

Usage
-----
    a = per_user_stats(users, labels, scores_a)
    b = per_user_stats(users, labels, scores_b)
    print(paired_bootstrap(a, b))
    # {'delta': 0.0013, 'ci_low': 0.0004, 'ci_high': 0.0021, 'p_two_sided': 0.008, ...}
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..kit import evaluate as kit_evaluate

# Reuse the frozen kit's own metric internals so per-user values cannot drift
# from the official implementation.
from evaluate import auc as _kit_auc, ndcg_at_k as _kit_ndcg  # noqa: E402


@dataclass
class PerUserStats:
    """Per-user decomposition of the primary metric.

    ``ndcg``     : nDCG@k for every user (all-negative users are 0, as the kit does)
    ``npos``     : number of positives per user
    ``auc``      : per-user AUC, 0.0 where the user is not GAUC-eligible
    ``eligible`` : 0 < npos < impressions — the users GAUC actually counts
    """

    users: list
    ndcg: np.ndarray
    npos: np.ndarray
    auc: np.ndarray
    eligible: np.ndarray
    k: int = 5

    def __len__(self) -> int:
        return len(self.users)


def per_user_stats(user_ids, labels, scores, k: int = 5) -> PerUserStats:
    """Decompose one model's predictions into per-user metric contributions.

    Mirrors the kit's ``evaluate`` exactly, including its conventions: users with
    no positives score nDCG 0 and are still counted in the average, and GAUC
    counts only users with 0 < positives < impressions, weighted by positive count.
    """
    by_user: dict = {}
    for u, y, s in zip(user_ids, labels, scores):
        by_user.setdefault(u, []).append((s, y))

    users, nd, npos_a, auc_a, elig = [], [], [], [], []
    for u, lst in by_user.items():
        lst.sort(key=lambda x: -x[0])
        labs = [y for _, y in lst]
        npos = sum(labs)
        users.append(u)
        nd.append(_kit_ndcg(labs, k))
        npos_a.append(npos)
        if 0 < npos < len(labs):
            elig.append(True)
            auc_a.append(_kit_auc(labs, [s for s, _ in lst]))
        else:
            elig.append(False)
            auc_a.append(0.0)

    return PerUserStats(
        users=users,
        ndcg=np.asarray(nd, dtype=np.float64),
        npos=np.asarray(npos_a, dtype=np.float64),
        auc=np.asarray(auc_a, dtype=np.float64),
        eligible=np.asarray(elig, dtype=bool),
        k=k,
    )


def primary_from(st: PerUserStats, idx: np.ndarray | None = None) -> float:
    """Recompute the primary score over a (possibly resampled) user index."""
    ndcg_v = st.ndcg if idx is None else st.ndcg[idx]
    npos_v = st.npos if idx is None else st.npos[idx]
    auc_v = st.auc if idx is None else st.auc[idx]
    elig_v = st.eligible if idx is None else st.eligible[idx]

    ndcg = float(ndcg_v.mean()) if len(ndcg_v) else 0.0
    den = float((npos_v * elig_v).sum())
    gauc = float((npos_v * auc_v * elig_v).sum() / den) if den else 0.5
    return 0.5 * (gauc + ndcg)


def verify_against_kit(user_ids, labels, scores, k: int = 5,
                       tol: float | None = None) -> None:
    """Assert the decomposition reproduces the kit's own score.

    Cheap insurance: if the frozen kit's metric ever changes, this fails loudly
    rather than letting every bootstrap interval drift silently.

    On float32 tolerance
    --------------------
    ``Dataset.y()`` returns **float32** (the kit's ``encode`` builds it that way).
    The kit's ``evaluate`` then accumulates its per-user nDCG list with Python's
    ``sum`` over float32 scalars, so its total carries float32 rounding — around
    1e-8 relative on a 27k-user split.

    This module accumulates in float64, so the two agree only to float32 precision
    when the labels arrive as float32. That is not a defect in either: the
    decomposition here is the *more* accurate of the two, and the difference is
    far below any effect size that matters (the smallest interesting effect on
    this benchmark is 1e-3, five orders of magnitude larger).

    The tolerance therefore adapts to the input dtype rather than being fixed. A
    hard 1e-9 would fail on every real dataset while passing on float64 test
    fixtures — which is exactly how this went unnoticed until the first
    full-scale run.
    """
    st = per_user_stats(user_ids, labels, scores, k=k)
    ours = primary_from(st)
    theirs = kit_evaluate(user_ids, labels, scores, k=k)["primary"]

    if tol is None:
        f32 = any(getattr(np.asarray(x), "dtype", None) == np.float32
                  for x in (labels, scores))
        tol = 1e-6 if f32 else 1e-9

    if abs(float(ours) - float(theirs)) > tol:
        raise AssertionError(
            f"per-user decomposition disagrees with the kit: {ours!r} vs {theirs!r} "
            f"(tolerance {tol:g})"
        )


def paired_bootstrap(a: PerUserStats, b: PerUserStats, *, n_boot: int = 2000,
                     seed: int = 0, alpha: float = 0.05) -> dict:
    """Confidence interval for (a - b) in primary score, resampling users.

    Both models must have been scored on the same users in the same order — that
    pairing is what makes the interval tight, so it is enforced rather than
    assumed.

    Returns the observed delta, the percentile CI, a two-sided bootstrap p-value,
    and ``significant``: whether the CI excludes zero.
    """
    if a.users != b.users:
        raise ValueError(
            "paired bootstrap requires both models scored on the same users in "
            "the same order; got differing user lists"
        )

    obs = primary_from(a) - primary_from(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = primary_from(a, idx) - primary_from(b, idx)

    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided bootstrap p: how often the resampled delta crosses zero.
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "delta": obs,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_two_sided": float(min(p, 1.0)),
        "significant": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
        "n_users": n,
        "se": float(deltas.std(ddof=1)),
    }


def bootstrap_ci(st: PerUserStats, *, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """Unpaired CI for a single model's primary score.

    Useful for reporting a model in isolation. For *comparisons* always prefer
    ``paired_bootstrap`` — the paired interval is much tighter, because it does
    not have to carry the between-user variance that cancels in the difference.
    """
    obs = primary_from(st)
    rng = np.random.default_rng(seed)
    n = len(st)
    vals = np.array([primary_from(st, rng.integers(0, n, size=n))
                     for _ in range(n_boot)])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"primary": obs, "ci_low": float(lo), "ci_high": float(hi),
            "se": float(vals.std(ddof=1)), "n_boot": n_boot, "n_users": n}


def diagnose_eval_split(user_ids, labels, k: int = 5) -> dict:
    """Check whether the metric can move on this split at all — run it first.

    A user who is all-positive or all-negative scores a fixed nDCG regardless of
    the model, and is excluded from GAUC entirely. If a split contains *no*
    discriminative users, the primary score is a constant: every model ties, every
    delta is exactly 0.0000, and every confidence interval is [0, 0].

    That failure is silent and looks exactly like "the change had no effect",
    which is why this is worth checking before trusting any comparison. It is not
    hypothetical — the repository's own ``tests/_mini_data.py`` fixture produces a
    validation split with 24 all-negative and 16 all-positive users and **zero**
    discriminative ones, so any test using it to assert that a score responds to
    predictions is vacuous.

    On the real KuaiRand-Pure test split the figures are 27.1% all-negative, 9.2%
    all-positive, 63.7% discriminative.
    """
    st = per_user_stats(user_ids, labels, np.zeros(len(labels)), k=k)
    n = len(st)
    n_impr = np.zeros(n)
    counts: dict = {}
    for u in st.users:
        counts[u] = 0
    for u in user_ids:
        counts[u] += 1
    n_impr = np.asarray([counts[u] for u in st.users], dtype=float)

    all_neg = int((st.npos == 0).sum())
    all_pos = int((st.npos == n_impr).sum())
    disc = n - all_neg - all_pos
    return {
        "users": n,
        "rows": len(labels),
        "all_negative": all_neg,
        "all_positive": all_pos,
        "discriminative": disc,
        "discriminative_frac": disc / n if n else 0.0,
        "gauc_eligible": int(st.eligible.sum()),
        "metric_can_move": bool(disc > 0),
        "warning": (None if disc > 0 else
                    "DEGENERATE SPLIT: no discriminative users. The primary score "
                    "is constant here — every model will tie and every interval "
                    "will be [0, 0]. Comparisons on this split are meaningless."),
    }


def format_comparison(name_a: str, name_b: str, result: dict) -> str:
    """One-line report suitable for a leaderboard row or a run log."""
    verdict = "significant" if result["significant"] else "inside noise"
    return (f"{name_a} vs {name_b}: {result['delta']:+.4f} "
            f"[95% CI {result['ci_low']:+.4f}, {result['ci_high']:+.4f}] "
            f"p={result['p_two_sided']:.3f} ({verdict}, {result['n_users']:,} users)")
