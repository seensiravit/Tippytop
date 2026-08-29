"""Field-aware factorization machine (FFM)."""

from __future__ import annotations

import numpy as np

from .base import CategoricalRanker


class FieldAwareFactorizationMachine(CategoricalRanker):
    model_type = "ffm"

    def __init__(
        self,
        dimension: int,
        *,
        field_count: int,
        embedding_dim: int,
        learning_rate: float,
        l2: float,
        seed: int,
    ) -> None:
        self.field_count = field_count
        super().__init__(
            dimension,
            (dimension, field_count, embedding_dim),
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )

    def _interaction_logits(self, features: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(features), dtype=np.float32)
        for left in range(self.field_count):
            for right in range(left + 1, self.field_count):
                left_vectors = self.V[features[:, left], right]
                right_vectors = self.V[features[:, right], left]
                scores += (left_vectors * right_vectors).sum(axis=1)
        return scores

    def _accumulate_interaction_gradient(
        self,
        gradient_v: np.ndarray,
        features: np.ndarray,
        score_gradient: np.ndarray,
    ) -> None:
        for left in range(self.field_count):
            for right in range(left + 1, self.field_count):
                left_ids = features[:, left]
                right_ids = features[:, right]
                left_vectors = self.V[left_ids, right]
                right_vectors = self.V[right_ids, left]
                np.add.at(
                    gradient_v[:, right],
                    left_ids,
                    score_gradient[:, None] * right_vectors,
                )
                np.add.at(
                    gradient_v[:, left],
                    right_ids,
                    score_gradient[:, None] * left_vectors,
                )
