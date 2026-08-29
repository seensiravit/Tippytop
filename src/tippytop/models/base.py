"""The one interface every model must satisfy.

The kit decouples model from scoring: a model only needs to turn eval rows into
a ``scores`` array (any real numbers; only within-user relative order matters).
Implement ``fit`` + ``predict`` and the runner + submission tooling work unchanged.

Contract:
    model.fit(dataset)                     # train on dataset (uses 'train', may
                                           # early-stop on 'valid')
    scores = model.predict(dataset, split) # 1-D float array, one per row of
                                           # dataset.splits[split], in row order
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from ..data.dataset import Dataset


class Model(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, data: Dataset) -> "Model":
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: Dataset, split: str) -> np.ndarray:
        raise NotImplementedError
