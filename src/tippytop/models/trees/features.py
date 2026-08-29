"""Leakage-controlled dense features for tree-based rankers."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


CATEGORY_FIELDS = (1, 2, 3, 4)
PAIR_FIELDS = ((1, 2), (1, 3), (1, 4))


@dataclass
class TreeFeatureEncoder:
    smoothing: float
    global_mean: float
    min_date: int
    duration_edges: np.ndarray
    category_vocabs: list[dict[str, int]]
    category_stats: list[dict[str, tuple[int, int]]]
    pair_stats: list[dict[tuple[str, str], tuple[int, int]]]

    @classmethod
    def fit(cls, rows: Sequence[tuple[Any, ...]], *, smoothing: float) -> "TreeFeatureEncoder":
        if not rows:
            raise ValueError("tree feature encoder requires non-empty training rows")
        category_counts = [collections.Counter() for _ in CATEGORY_FIELDS]
        category_positives = [collections.Counter() for _ in CATEGORY_FIELDS]
        pair_counts = [collections.Counter() for _ in PAIR_FIELDS]
        pair_positives = [collections.Counter() for _ in PAIR_FIELDS]
        durations = np.asarray([float(row[5]) for row in rows], dtype=np.float64)

        for row in rows:
            label = int(row[6])
            for index, field in enumerate(CATEGORY_FIELDS):
                value = str(row[field])
                category_counts[index][value] += 1
                category_positives[index][value] += label
            for index, (left, right) in enumerate(PAIR_FIELDS):
                value = (str(row[left]), str(row[right]))
                pair_counts[index][value] += 1
                pair_positives[index][value] += label

        category_vocabs = [
            {value: code for code, value in enumerate(counts)} for counts in category_counts
        ]
        category_stats = [
            {value: (count, positives[value]) for value, count in counts.items()}
            for counts, positives in zip(category_counts, category_positives, strict=True)
        ]
        pair_stats = [
            {value: (count, positives[value]) for value, count in counts.items()}
            for counts, positives in zip(pair_counts, pair_positives, strict=True)
        ]
        return cls(
            smoothing=smoothing,
            global_mean=float(sum(int(row[6]) for row in rows) / len(rows)),
            min_date=min(int(row[0]) for row in rows),
            duration_edges=np.quantile(durations, np.linspace(0, 1, 11)[1:-1]),
            category_vocabs=category_vocabs,
            category_stats=category_stats,
            pair_stats=pair_stats,
        )

    @classmethod
    def fit_transform_training(
        cls,
        rows: Sequence[tuple[Any, ...]],
        *,
        smoothing: float,
        folds: int = 5,
    ) -> tuple["TreeFeatureEncoder", np.ndarray]:
        """Fit the replay encoder and build target features without self-label leakage."""

        encoder = cls.fit(rows, smoothing=smoothing)
        fold_count = min(folds, len(rows))
        if fold_count < 2:
            return encoder, encoder.transform(rows, leave_one_out=True)

        rng = np.random.default_rng(0)
        fold_by_row = np.empty(len(rows), dtype=np.int8)
        permutation = rng.permutation(len(rows))
        fold_by_row[permutation] = np.arange(len(rows), dtype=np.int64) % fold_count
        features = np.empty((len(rows), encoder.feature_count), dtype=np.float32)

        # Entire held-out folds, rather than individual labels, are excluded from target rates.
        for fold in range(fold_count):
            holdout = np.flatnonzero(fold_by_row == fold)
            fitting_rows = [row for index, row in enumerate(rows) if fold_by_row[index] != fold]
            fold_encoder = cls.fit(fitting_rows, smoothing=smoothing)
            holdout_rows = [rows[int(index)] for index in holdout]
            features[holdout] = fold_encoder.transform(holdout_rows)
        return encoder, features

    @property
    def feature_count(self) -> int:
        return 3 + 3 * len(CATEGORY_FIELDS) + 2 * len(PAIR_FIELDS)

    @property
    def categorical_indices(self) -> list[int]:
        return [3 + 3 * index for index in range(len(CATEGORY_FIELDS))]

    def transform(
        self,
        rows: Sequence[tuple[Any, ...]],
        *,
        leave_one_out: bool = False,
    ) -> np.ndarray:
        features = np.empty((len(rows), self.feature_count), dtype=np.float32)
        for row_index, row in enumerate(rows):
            label = int(row[6]) if leave_one_out else 0
            features[row_index, 0] = int(row[0]) - self.min_date
            features[row_index, 1] = np.log1p(max(0.0, float(row[5])))
            features[row_index, 2] = np.searchsorted(self.duration_edges, float(row[5]))
            column = 3
            for field_index, field in enumerate(CATEGORY_FIELDS):
                value = str(row[field])
                vocabulary = self.category_vocabs[field_index]
                features[row_index, column] = vocabulary.get(value, len(vocabulary))
                count, positives = self.category_stats[field_index].get(value, (0, 0))
                if leave_one_out:
                    # Remove this row's label from training target-rate features.
                    count = max(0, count - 1)
                    positives -= label
                features[row_index, column + 1] = np.log1p(count)
                features[row_index, column + 2] = self._smoothed_rate(count, positives)
                column += 3
            for pair_index, (left, right) in enumerate(PAIR_FIELDS):
                value = (str(row[left]), str(row[right]))
                count, positives = self.pair_stats[pair_index].get(value, (0, 0))
                if leave_one_out:
                    # Validation/test use full training aggregates, never their own labels.
                    count = max(0, count - 1)
                    positives -= label
                features[row_index, column] = np.log1p(count)
                features[row_index, column + 1] = self._smoothed_rate(count, positives)
                column += 2
        return features

    def _smoothed_rate(self, count: int, positives: int) -> float:
        return float(
            (positives + self.smoothing * self.global_mean) / (count + self.smoothing)
        )
