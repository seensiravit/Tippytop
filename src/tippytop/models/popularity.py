"""Item-popularity baseline (trivial, ~0.5715). Ported from baseline.run_pop.

Scores every video by its smoothed global long_view rate over the train split.
No training, no per-user modelling — same score for a video for every user.
"""
from __future__ import annotations
import collections
import numpy as np

from .base import Model
from ..data.dataset import Dataset
from . import register


@register("pop")
class Popularity(Model):
    name = "pop"

    def __init__(self, prior=20.0, seed=0, **_):
        self.prior = prior
        self._score = None
        self._gmean = 0.0

    def fit(self, data: Dataset) -> "Popularity":
        pos, imp = collections.Counter(), collections.Counter()
        for x in data.splits["train"]:
            imp[x[2]] += 1
            pos[x[2]] += x[6]
        self._gmean = sum(pos.values()) / sum(imp.values())
        p, g, prior = pos, self._gmean, self.prior

        def score(v):
            return (p[v] + prior * g) / (imp[v] + prior) if imp[v] else g

        self._score = score
        return self

    def predict(self, data: Dataset, split: str) -> np.ndarray:
        if self._score is None:
            raise RuntimeError("fit() must be called before predict()")
        return np.array([self._score(x[2]) for x in data.splits[split]],
                        dtype=np.float64)
