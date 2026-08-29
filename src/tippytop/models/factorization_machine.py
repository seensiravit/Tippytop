"""Standard factorization machine over all categorical field pairs."""

from __future__ import annotations

import numpy as np

from .base import CategoricalRanker


class FactorizationMachine(CategoricalRanker):
    model_type = "fm"

    def __init__(
        self,
        dimension: int,
        *,
        embedding_dim: int = 16,
        learning_rate: float = 0.001,
        l2: float = 1e-6,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dimension,
            (dimension, embedding_dim),
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )

    def _interaction_logits(self, features: np.ndarray) -> np.ndarray:
        embeddings = self.V[features]
        summed = embeddings.sum(axis=1)
        return 0.5 * ((summed**2).sum(axis=1) - (embeddings**2).sum(axis=(1, 2)))

    def _accumulate_interaction_gradient(
        self,
        gradient_v: np.ndarray,
        features: np.ndarray,
        score_gradient: np.ndarray,
    ) -> None:
        embeddings = self.V[features]
        summed = embeddings.sum(axis=1)
        np.add.at(
            gradient_v,
            features,
            score_gradient[:, None, None] * (summed[:, None, :] - embeddings),
        )
