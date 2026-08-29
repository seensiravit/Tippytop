"""Shared optimizer and objectives for categorical ranking models."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


ModelState = tuple[np.ndarray, np.ndarray, np.float32]
MAX_STEP_ROWS = 65_536


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


class CategoricalRanker(ABC):
    """Base class for sparse categorical models trained with Adam."""

    model_type: str

    def __init__(
        self,
        dimension: int,
        embedding_shape: tuple[int, ...],
        *,
        learning_rate: float,
        l2: float,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, embedding_shape).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        self.b = np.float32(0.0)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    @property
    def dimension(self) -> int:
        return len(self.W)

    def logits(self, features: np.ndarray) -> np.ndarray:
        self._validate_features(features)
        return self.b + self.W[features].sum(axis=1) + self._interaction_logits(features)

    def _validate_features(self, features: np.ndarray) -> None:
        if features.ndim != 2:
            raise ValueError(
                f"categorical rankers require a two-dimensional feature array, got {features.shape}"
            )
        if not np.issubdtype(features.dtype, np.integer):
            raise ValueError(
                "categorical rankers require integer IDs from CategoricalEncoder; "
                "TabularEncoder produces dense float features for sklearn and LightGBM"
            )
        if features.size and (int(features.min()) < 0 or int(features.max()) >= self.dimension):
            raise ValueError(
                "categorical feature ID is outside the model dimension; construct the model "
                "with dimension=encoder.dimension"
            )

    @abstractmethod
    def _interaction_logits(self, features: np.ndarray) -> np.ndarray:
        """Return the architecture-specific interaction score."""

    @abstractmethod
    def _accumulate_interaction_gradient(
        self,
        gradient_v: np.ndarray,
        features: np.ndarray,
        score_gradient: np.ndarray,
    ) -> None:
        """Accumulate d(loss)/d(V) for a batch."""

    def _parameter_gradients(
        self,
        features: np.ndarray,
        score_gradient: np.ndarray,
        *,
        weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_w, features, score_gradient[:, None])
        self._accumulate_interaction_gradient(gradient_v, features, score_gradient)
        gradient_v += weight * self.l2 * self.V
        gradient_w += weight * self.l2 * self.W
        return gradient_v, gradient_w

    def _update(self, gradient_v: np.ndarray, gradient_w: np.ndarray) -> None:
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        parameters = [(self.W, gradient_w, self.mW, self.vW)]
        if self.V.size:
            parameters.append((self.V, gradient_v, self.mV, self.vV))
        for parameter, gradient, first, second in parameters:
            first *= beta1
            first += (1 - beta1) * gradient
            second *= beta2
            second += (1 - beta2) * (gradient * gradient)
            corrected_first = first / (1 - beta1**self.t)
            corrected_second = second / (1 - beta2**self.t)
            parameter -= self.learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)

    def fit_pointwise(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = 1,
        batch_size: int = 8192,
        seed: int = 0,
    ) -> list[float]:
        """Run reproducible shuffled BCE minibatch epochs and return mean epoch losses."""

        _validate_fit_arguments(len(features), len(labels), epochs, batch_size, "pointwise")
        rng = np.random.default_rng(seed)
        history: list[float] = []
        for _ in range(epochs):
            order = rng.permutation(len(features))
            losses = [
                self.step_pointwise(features[batch], labels[batch])
                for batch in _batches(order, batch_size)
            ]
            history.append(float(np.mean(losses)))
        return history

    def fit_bpr(
        self,
        features: np.ndarray,
        positive_indices: np.ndarray,
        negative_indices: np.ndarray,
        *,
        epochs: int = 3,
        batch_size: int = 8192,
        seed: int = 0,
    ) -> list[float]:
        """Run reproducible shuffled BPR minibatch epochs and return mean epoch losses."""

        _validate_fit_arguments(
            len(positive_indices),
            len(negative_indices),
            epochs,
            batch_size,
            "BPR",
        )
        self._validate_features(features)
        if positive_indices.size and (
            int(positive_indices.min()) < 0
            or int(negative_indices.min()) < 0
            or int(positive_indices.max()) >= len(features)
            or int(negative_indices.max()) >= len(features)
        ):
            raise ValueError("BPR pair indices must reference rows in the feature matrix")
        rng = np.random.default_rng(seed)
        history: list[float] = []
        for _ in range(epochs):
            order = rng.permutation(len(positive_indices))
            losses = [
                self.step_bpr(
                    features[positive_indices[batch]],
                    features[negative_indices[batch]],
                )
                for batch in _batches(order, batch_size)
            ]
            history.append(float(np.mean(losses)))
        return history

    def step_pointwise(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        weight: float = 1.0,
    ) -> float:
        if len(features) == 0 or len(features) != len(labels):
            raise ValueError("pointwise training requires a non-empty feature/label batch")
        if len(features) > MAX_STEP_ROWS:
            raise ValueError(
                f"step_pointwise performs one optimizer update and accepts at most "
                f"{MAX_STEP_ROWS} rows; use fit_pointwise for complete minibatch training"
            )
        logits = self.logits(features)
        probabilities = sigmoid(logits)
        score_gradient = (weight * (probabilities - labels) / len(labels)).astype(np.float32)
        gradient_v, gradient_w = self._parameter_gradients(
            features,
            score_gradient,
            weight=weight,
        )
        self._update(gradient_v, gradient_w)
        self.b -= self.learning_rate * score_gradient.sum()
        loss = -np.mean(
            labels * np.log(probabilities + 1e-9)
            + (1 - labels) * np.log(1 - probabilities + 1e-9)
        )
        return float(loss)

    def step_bpr(
        self,
        positive: np.ndarray,
        negative: np.ndarray,
        *,
        weight: float = 1.0,
    ) -> float:
        if len(positive) == 0 or len(positive) != len(negative):
            raise ValueError("BPR training requires a non-empty, equally sized positive/negative batch")
        if len(positive) > MAX_STEP_ROWS:
            raise ValueError(
                f"step_bpr performs one optimizer update and accepts at most {MAX_STEP_ROWS} pairs; "
                "use fit_bpr with row indices for complete minibatch training"
            )
        # The global bias cancels in a score difference, so BPR updates only W/V.
        difference = self.logits(positive) - self.logits(negative)
        score_gradient = (weight * (sigmoid(difference) - 1.0) / len(difference)).astype(
            np.float32
        )
        positive_v, positive_w = self._parameter_gradients(
            positive,
            score_gradient,
            weight=weight / 2,
        )
        negative_v, negative_w = self._parameter_gradients(
            negative,
            -score_gradient,
            weight=weight / 2,
        )
        self._update(positive_v + negative_v, positive_w + negative_w)
        return float(np.mean(np.logaddexp(0.0, -difference)))

    def predict(self, features: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        if len(features) == 0:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(
            [
                self.logits(features[start : start + batch_size])
                for start in range(0, len(features), batch_size)
            ]
        )

    def snapshot(self) -> ModelState:
        return self.V.copy(), self.W.copy(), np.float32(self.b)

    def restore(self, state: ModelState) -> None:
        self.V, self.W, self.b = state

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        np.savez_compressed(
            path,
            V=self.V,
            W=self.W,
            b=np.asarray(self.b),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )


def _validate_fit_arguments(
    rows: int,
    targets: int,
    epochs: int,
    batch_size: int,
    objective: str,
) -> None:
    if rows == 0 or rows != targets:
        raise ValueError(f"{objective} training requires non-empty, equally sized inputs")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")


def _batches(order: np.ndarray, batch_size: int) -> list[np.ndarray]:
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
