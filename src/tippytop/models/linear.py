"""Linear categorical ranking model."""

from __future__ import annotations

import numpy as np

from .base import CategoricalRanker


class LinearModel(CategoricalRanker):
    model_type = "linear"

    def __init__(self, dimension: int, *, learning_rate: float, l2: float, seed: int) -> None:
        super().__init__(
            dimension,
            (dimension, 0),
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )

    def _interaction_logits(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(len(features), dtype=np.float32)

    def _accumulate_interaction_gradient(
        self,
        gradient_v: np.ndarray,
        features: np.ndarray,
        score_gradient: np.ndarray,
    ) -> None:
        return None
