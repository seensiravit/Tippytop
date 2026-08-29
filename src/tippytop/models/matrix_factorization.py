"""User-item matrix factorization with categorical linear terms."""

from __future__ import annotations

import numpy as np

from .base import CategoricalRanker


class MatrixFactorization(CategoricalRanker):
    model_type = "mf"

    def __init__(
        self,
        dimension: int,
        *,
        embedding_dim: int,
        learning_rate: float,
        l2: float,
        seed: int,
    ) -> None:
        super().__init__(
            dimension,
            (dimension, embedding_dim),
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )

    def _interaction_logits(self, features: np.ndarray) -> np.ndarray:
        return (self.V[features[:, 0]] * self.V[features[:, 1]]).sum(axis=1)

    def _accumulate_interaction_gradient(
        self,
        gradient_v: np.ndarray,
        features: np.ndarray,
        score_gradient: np.ndarray,
    ) -> None:
        user_ids = features[:, 0]
        item_ids = features[:, 1]
        users = self.V[user_ids]
        items = self.V[item_ids]
        np.add.at(gradient_v, user_ids, score_gradient[:, None] * items)
        np.add.at(gradient_v, item_ids, score_gradient[:, None] * users)
