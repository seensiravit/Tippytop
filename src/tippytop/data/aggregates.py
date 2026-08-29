"""Train-only aggregate features — the direction the organisers never tested.

The starter-kit README reports that "adding static features yields nothing"
(0.5940 vs 0.5950). That result is narrower than it reads: ``ablation_features.py``
opens only ``video_features_basic_pure.csv`` and tests only four *categorical* IDs
(``author_id``, ``music_id``, ``video_type``, ``upload_type``). Those are
redundant given ``video_id``, which the FM already embeds. Continuous engagement
rates were never tested, and they are a different kind of feature:

* they are **item-side**, so they vary within a user's impression list — the only
  kind of feature that can change a within-user ranking at all;
* they **generalise to rare videos**, where a per-id embedding is pure noise;
* they give a tree model something real to split on.

Everything here is computed from the **training split only**. The dataset also
ships ``video_features_statistic_pure.csv`` with equivalent columns, but its
aggregation window is undocumented and may span the test period — in which case
``long_time_play_cnt`` would leak the labels we are predicting. Computing our own
is provably clean and costs one pass.

Leakage within train is handled by **leave-one-out**: a training row's own label
is subtracted from its video's aggregate before the feature is read, otherwise
each row would partly predict itself and the model would learn to trust the
feature far more than it should. Validation and test rows are not in train, so
they use the plain aggregate.
"""
from __future__ import annotations

import collections

import numpy as np

# Kit row tuple: (date, user_id, video_id, author_id, tab, duration_ms, long_view)
_DATE, _USER, _VIDEO, _AUTHOR, _TAB, _DUR, _LABEL = range(7)

PRIOR = 20.0  # smoothing strength, matching the kit's popularity baseline

FEATURE_NAMES = [
    "video_lv_rate",       # smoothed long_view rate of this video (train)
    "video_impressions",   # log1p train impression count
    "author_lv_rate",
    "author_impressions",
    "duration_log",
    "tab",
    "user_lv_rate",        # constant within a user; useful only via crosses
    "user_impressions",
    "video_rate_vs_user",  # video_lv_rate minus the mean over this user's own list
    "duration_vs_user",    # duration_log minus the mean over this user's own list
    # --- personalised crosses ---------------------------------------------
    # Everything above is GLOBAL: the same ranking function for every user.
    # Measured, these cap out around a slightly-better popularity model, because
    # nothing in them depends on *who* is watching. These three do: they vary
    # within a user's list (author/duration/tab differ across their impressions)
    # AND depend on that user's own history. Each is shrunk toward the user's
    # overall rate, so a user with no history on an author falls back to their
    # own base rate rather than to noise.
    "user_author_rate",    # this user's train rate on this author
    "user_durbucket_rate", # this user's train rate on this duration bucket
    "user_tab_rate",       # this user's train rate on this tab
    "user_author_seen",    # log1p times this user saw this author in train
]


def _smoothed(pos: float, imp: float, prior_mean: float) -> float:
    return (pos + PRIOR * prior_mean) / (imp + PRIOR)


def _counts(rows, key_idx):
    pos, imp = collections.Counter(), collections.Counter()
    for r in rows:
        k = r[key_idx]
        imp[k] += 1
        pos[k] += r[_LABEL]
    return pos, imp


CROSS_PRIOR = 5.0  # crosses are sparser than item aggregates — shrink harder


def _dur_buckets(train, n=10):
    edges = np.quantile(np.asarray([r[_DUR] for r in train], dtype=np.float64),
                        np.linspace(0, 1, n + 1)[1:-1])
    return edges


def build_features(splits: dict) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return {split: float32 (N, len(FEATURE_NAMES))} plus the column names.

    Row order matches ``splits[name]`` exactly, so these line up with the kit's
    encoded matrices, the labels, and the submission's ``row_id``.
    """
    train = splits["train"]
    gmean = sum(r[_LABEL] for r in train) / max(len(train), 1)
    edges = _dur_buckets(train)

    def bucket(r):
        return int(np.searchsorted(edges, r[_DUR]))

    v_pos, v_imp = _counts(train, _VIDEO)
    a_pos, a_imp = _counts(train, _AUTHOR)
    u_pos, u_imp = _counts(train, _USER)

    # Personalised crosses, train only. Keyed on (user, attribute).
    ua_pos, ua_imp = collections.Counter(), collections.Counter()
    ub_pos, ub_imp = collections.Counter(), collections.Counter()
    ut_pos, ut_imp = collections.Counter(), collections.Counter()
    for r in train:
        u, y = r[_USER], r[_LABEL]
        for pos, imp, k in ((ua_pos, ua_imp, (u, r[_AUTHOR])),
                            (ub_pos, ub_imp, (u, bucket(r))),
                            (ut_pos, ut_imp, (u, r[_TAB]))):
            imp[k] += 1
            pos[k] += y

    out: dict[str, np.ndarray] = {}
    for name, rows in splits.items():
        loo = name == "train"
        F = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float32)

        for i, r in enumerate(rows):
            y = r[_LABEL] if loo else 0.0
            d = 1.0 if loo else 0.0  # drop this row from its own aggregates

            v, a, u, t = r[_VIDEO], r[_AUTHOR], r[_USER], r[_TAB]
            b = bucket(r)
            v_rate = _smoothed(v_pos[v] - y, v_imp[v] - d, gmean)
            a_rate = _smoothed(a_pos[a] - y, a_imp[a] - d, gmean)
            u_rate = _smoothed(u_pos[u] - y, u_imp[u] - d, gmean)

            F[i, 0] = v_rate
            F[i, 1] = np.log1p(max(v_imp[v] - d, 0.0))
            F[i, 2] = a_rate
            F[i, 3] = np.log1p(max(a_imp[a] - d, 0.0))
            F[i, 4] = np.log1p(r[_DUR])
            F[i, 5] = float(t)
            F[i, 6] = u_rate
            F[i, 7] = np.log1p(max(u_imp[u] - d, 0.0))

            # Crosses shrink toward this user's own base rate, not the global
            # mean: with no history on an author, the best guess for a user is
            # how they behave in general.
            for col, (pos, imp, k) in enumerate(
                ((ua_pos, ua_imp, (u, a)),
                 (ub_pos, ub_imp, (u, b)),
                 (ut_pos, ut_imp, (u, t))), start=10
            ):
                n = imp[k] - d
                F[i, col] = (pos[k] - y + CROSS_PRIOR * u_rate) / (n + CROSS_PRIOR)
            F[i, 13] = np.log1p(max(ua_imp[(u, a)] - d, 0.0))

        _add_within_user_deltas(F, rows)
        out[name] = F

    return out, list(FEATURE_NAMES)


def _add_within_user_deltas(F: np.ndarray, rows) -> None:
    """Centre two columns against the mean over each user's own impressions.

    Scoring is within-user, so what matters is not "is this video good" but "is
    it better than the others this user is being shown". These deltas encode that
    directly, and they are computed from features alone — no labels — so they are
    available at prediction time.
    """
    if len(rows) == 0:
        return
    users = np.array([r[_USER] for r in rows])
    _, codes = np.unique(users, return_inverse=True)
    n_groups = codes.max() + 1

    counts = np.bincount(codes, minlength=n_groups).astype(np.float32)
    for src, dst in ((0, 8), (4, 9)):  # video_lv_rate -> col 8, duration_log -> col 9
        sums = np.bincount(codes, weights=F[:, src], minlength=n_groups)
        F[:, dst] = F[:, src] - (sums / counts)[codes]
