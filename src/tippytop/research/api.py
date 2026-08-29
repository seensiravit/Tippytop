"""Small, label-safe building blocks available to generated experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..models.trees.features import TreeFeatureEncoder
from .data import PREDICTION_COLUMNS


class CategoricalEncoder:
    """Training-fitted encoder for the starter kit's five categorical fields."""

    def __init__(self) -> None:
        self.duration_edges: np.ndarray | None = None
        self.vocabularies: list[dict[str, int]] | None = None
        self.offsets: np.ndarray | None = None
        self.dimension = 0

    def fit(self, train_rows: Any) -> "CategoricalEncoder":
        if len(train_rows) == 0:
            raise ValueError("categorical encoding requires non-empty training rows")
        rows = _starter_rows(train_rows, require_label=False)
        durations = np.asarray([float(row[5]) for row in rows], dtype=np.float64)
        edges = np.quantile(durations, np.linspace(0, 1, 11)[1:-1])
        vocabularies = [dict() for _ in range(5)]
        for row in rows:
            for field, value in enumerate(_raw_fields(row, edges)):
                if value not in vocabularies[field]:
                    vocabularies[field][value] = len(vocabularies[field])
        field_dimensions = [len(vocabulary) + 1 for vocabulary in vocabularies]
        self.duration_edges = edges
        self.vocabularies = vocabularies
        self.offsets = np.cumsum([0] + field_dimensions[:-1]).astype(np.int32)
        self.dimension = int(sum(field_dimensions))
        return self

    def transform(self, rows: Any) -> np.ndarray:
        if self.duration_edges is None or self.vocabularies is None or self.offsets is None:
            raise RuntimeError("CategoricalEncoder must be fitted before transform")
        starter_rows = _starter_rows(rows, require_label=False)
        encoded = np.empty((len(starter_rows), len(self.vocabularies)), dtype=np.int32)
        for row_index, row in enumerate(starter_rows):
            for field, value in enumerate(_raw_fields(row, self.duration_edges)):
                vocabulary = self.vocabularies[field]
                encoded[row_index, field] = vocabulary.get(value, len(vocabulary)) + self.offsets[field]
        return encoded

    def fit_transform(self, train_rows: Any) -> np.ndarray:
        """Fit once and return the integer IDs expected by categorical rankers."""

        return self.fit(train_rows).transform(train_rows)


class TabularEncoder:
    """Leakage-safe dense features for sklearn and LightGBM experiments."""

    def __init__(self, *, smoothing: float = 20.0):
        self.smoothing = smoothing
        self.encoder: TreeFeatureEncoder | None = None

    def fit(self, train_rows: Any) -> "TabularEncoder":
        self.encoder = TreeFeatureEncoder.fit(
            _starter_rows(train_rows, require_label=True),
            smoothing=self.smoothing,
        )
        return self

    def fit_transform(self, train_rows: Any) -> np.ndarray:
        """Cross-fit training features once and retain the full-data replay encoder."""

        self.encoder, features = TreeFeatureEncoder.fit_transform_training(
            _starter_rows(train_rows, require_label=True),
            smoothing=self.smoothing,
        )
        return features

    def transform(self, rows: Any) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("TabularEncoder must be fitted before transform")
        return self.encoder.transform(_starter_rows(rows, require_label=False))

    @property
    def categorical_indices(self) -> list[int]:
        if self.encoder is None:
            raise RuntimeError("TabularEncoder must be fitted before reading categorical indices")
        return self.encoder.categorical_indices


def labels(rows: Any) -> np.ndarray:
    """Return training labels; prediction rows intentionally have no label field."""

    if isinstance(rows, pd.DataFrame):
        if "long_view" not in rows:
            raise ValueError("training frame has no long_view label")
        return rows["long_view"].to_numpy(dtype=np.float32, copy=True)
    return np.asarray([float(row[6]) for row in rows], dtype=np.float32)


def user_ids(rows: Any) -> list[str]:
    if isinstance(rows, pd.DataFrame):
        return rows["user_id"].astype(str).tolist()
    return [str(row[1]) for row in rows]


def _raw_fields(row: tuple[Any, ...], duration_edges: np.ndarray) -> list[str]:
    return [
        str(row[1]),
        str(row[2]),
        str(row[3]),
        str(row[4]),
        str(int(np.searchsorted(duration_edges, float(row[5])))),
    ]


def _starter_rows(rows: Any, *, require_label: bool) -> list[tuple[Any, ...]]:
    if not isinstance(rows, pd.DataFrame):
        return [tuple(row) for row in rows]
    required = list(PREDICTION_COLUMNS[:6])
    if require_label:
        required.append("long_view")
    missing = [column for column in required if column not in rows]
    if missing:
        raise ValueError(f"research frame is missing columns: {missing}")
    return list(rows.loc[:, required].itertuples(index=False, name=None))
