"""LightGBM LambdaRank — a different model family from FM entirely.

Why this belongs here:

* LightGBM's ``lambdarank`` objective takes **query groups**, which map exactly
  onto users in this task. It optimises NDCG directly, weighting each pair by the
  metric change that swapping them would cause — so the training objective is the
  scored objective, without the batching penalty a within-user softmax pays on FM
  (see ``losses/ranking.py`` and results/leaderboard.md).
* ~7 impressions per user is squarely LambdaRank's regime; it was designed for
  short ranked lists.
* It is **axis-aligned splits over engineered features**, where FM is latent
  factors over ids. Ensemble members only compound when they fail differently,
  and our same-family ensembles measurably did not.

The organisers name LightGBM as in scope twice (problem statement 2.3 *In scope*
and 2.4 *Resource policy*). It is an optional dependency:

    uv pip install -e ".[models]"

Features come from ``data/aggregates.py`` — train-only, leave-one-out encoded.
Trees need real per-row numbers to split on; raw ids would just be noise here.
"""
from __future__ import annotations

import numpy as np

from .base import Model
from ..data.dataset import Dataset
from ..data.aggregates import build_features
from . import register
from ..kit import evaluate


def _sorted_groups(users) -> tuple[np.ndarray, np.ndarray]:
    """Return (row order that groups a user's rows together, block sizes).

    LightGBM's ``group`` argument is a list of *consecutive* block lengths, so
    every user's rows must be adjacent. The kit's loader preserves file order,
    which is chronological across all users — measured on this data, 26,210
    train users arrive in 50,939 separate runs, i.e. each user's impressions are
    interleaved with everyone else's. Passing group sizes computed from that
    order would silently define the wrong lists: LightGBM would happily train on
    "queries" made of unrelated rows, and nothing in the metrics would reveal it.

    Sorting is stable, so within a user the original chronological order of their
    impressions is preserved.
    """
    users = np.asarray(users)
    if len(users) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int32)
    order = np.argsort(users, kind="stable")
    srt = users[order]
    starts = np.flatnonzero(np.r_[True, srt[1:] != srt[:-1]])
    sizes = np.diff(np.r_[starts, len(srt)]).astype(np.int32)
    assert sizes.sum() == len(users) and len(sizes) == len(np.unique(users))
    return order, sizes


@register("lgbm_rank")
class LGBMRank(Model):
    """LightGBM LambdaRank over train-only aggregate features."""

    name = "lgbm_rank"

    def __init__(self, n_estimators=400, learning_rate=0.05, num_leaves=31,
                 min_child_samples=50, colsample_bytree=0.9, subsample=0.9,
                 early_stopping_rounds=30, seed=0, verbose=True, **kw):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.seed, self.verbose = seed, verbose
        self._booster = None
        self._feats: dict | None = None
        self._names: list[str] = []

    def _features(self, data: Dataset) -> dict:
        if self._feats is None:
            if self.verbose:
                print("  building train-only aggregate features...")
            self._feats, self._names = build_features(data.splits)
        return self._feats

    def fit(self, data: Dataset) -> "LGBMRank":
        try:
            import lightgbm as lgb
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "lgbm_rank needs LightGBM. Install the optional extra:\n"
                '    uv pip install -e ".[models]"'
            ) from e

        feats = self._features(data)
        # Reorder so each user's impressions are adjacent — required by `group`.
        # Only training needs this; predict() scores rows independently and so
        # returns them in the caller's original order.
        otr, gtr = _sorted_groups(data.users("train"))
        ova, gva = _sorted_groups(data.users("valid"))
        Xtr, ytr = feats["train"][otr], data.y("train")[otr]
        Xva, yva = feats["valid"][ova], data.y("valid")[ova]

        if self.verbose:
            print(f"  train {Xtr.shape[0]:,} rows / {len(gtr):,} users | "
                  f"valid {Xva.shape[0]:,} rows / {len(gva):,} users | "
                  f"{Xtr.shape[1]} features")

        self._booster = lgb.LGBMRanker(
            objective="lambdarank",
            # Optimise the metric we are actually scored on. eval_at=5 makes
            # early stopping track NDCG@5 rather than a proxy.
            metric="ndcg",
            eval_at=[5],
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            subsample_freq=1,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        self._booster.fit(
            Xtr, ytr, group=gtr,
            eval_set=[(Xva, yva)], eval_group=[gva], eval_at=[5],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )

        if self.verbose:
            best = self._booster.best_iteration_ or self.n_estimators
            # Score in the ORIGINAL row order, not the sorted one.
            va = evaluate(data.users("valid"), data.y("valid"),
                          self.predict(data, "valid"))
            print(f"  stopped at {best} trees | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f}")
        return self

    def predict(self, data: Dataset, split: str) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("fit() must be called before predict()")
        return self._booster.predict(self._features(data)[split])
