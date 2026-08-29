"""Random-scoring lower bound (~0.4753). Ported from baseline.run_random.

Exists purely as a harness sanity check: if this doesn't score primary ~= 0.475,
the eval pipeline is broken — fix that before trusting any real result.
"""
from __future__ import annotations
import numpy as np

from .base import Model
from ..data.dataset import Dataset
from . import register


@register("random")
class RandomModel(Model):
    name = "random"

    def __init__(self, seed=0, **_):
        self.seed = seed

    def fit(self, data: Dataset) -> "RandomModel":
        return self  # nothing to train

    def predict(self, data: Dataset, split: str) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.random(len(data.splits[split]))
