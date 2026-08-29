"""Checkpoint prediction and optional score blending."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tippytop.models import load_model
from tippytop.models.trees import TreeModel
from tippytop.starter import encode_splits


def popularity_scores(
    train_rows: Sequence[tuple[Any, ...]],
    rows: Sequence[tuple[Any, ...]],
    *,
    prior: float = 20.0,
) -> np.ndarray:
    positives: collections.Counter[str] = collections.Counter()
    impressions: collections.Counter[str] = collections.Counter()
    # Popularity is derived exclusively from training labels and is safe for later splits.
    for row in train_rows:
        impressions[row[2]] += 1
        positives[row[2]] += int(row[6])
    global_mean = sum(positives.values()) / max(1, sum(impressions.values()))
    probabilities = np.asarray(
        [
            (positives[row[2]] + prior * global_mean) / (impressions[row[2]] + prior)
            if impressions[row[2]]
            else global_mean
            for row in rows
        ],
        dtype=np.float32,
    )
    clipped = np.clip(probabilities, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def blend_scores(
    model_scores: np.ndarray,
    splits: dict[str, list[tuple[Any, ...]]],
    split: str,
    *,
    popularity_weight: float = 0.0,
) -> np.ndarray:
    if popularity_weight == 0.0:
        return model_scores
    item_scores = popularity_scores(splits["train"], splits[split])
    return (1.0 - popularity_weight) * model_scores + popularity_weight * item_scores


def predict_checkpoint(
    checkpoint_path: Path,
    splits: dict[str, list[tuple[Any, ...]]],
    split: str,
    *,
    data_dir: Path | None = None,
) -> np.ndarray:
    if checkpoint_path.is_dir():
        # Generated checkpoints are directories; trusted built-ins use explicit file formats.
        from tippytop.runtime import predict_generated
        from tippytop.research.data import load_prediction_frame

        if data_dir is None:
            raise ValueError("data_dir is required to replay a generated checkpoint")
        return predict_generated(checkpoint_path, load_prediction_frame(data_dir, split))
    if checkpoint_path.name.endswith(".pkl.gz"):
        model, _ = TreeModel.load(checkpoint_path)
        return model.predict(splits[split])
    model, metadata = load_model(checkpoint_path)
    encoded, expected_dimension = encode_splits(splits)
    if model.dimension != expected_dimension:
        raise ValueError(
            f"checkpoint dimension {model.dimension} does not match encoded data {expected_dimension}"
        )
    features, _, _ = encoded[split]
    if features.shape[1] != int(metadata["field_count"]):
        raise ValueError("checkpoint field count does not match encoded data")
    return blend_scores(
        model.predict(features),
        splits,
        split,
        popularity_weight=float(metadata.get("popularity_weight", 0.0)),
    )
