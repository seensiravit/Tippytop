"""Trusted training path for random forests and boosted-tree models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from tippytop.models.trees import fit_tree_model
from tippytop.starter import evaluate


def train_tree(
    splits: dict[str, list[tuple[Any, ...]]],
    artifact_dir: Path,
    *,
    model_type: str,
    parameters: dict[str, int | float],
    verbose: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    model, rounds = fit_tree_model(
        model_type,
        splits["train"],
        splits["valid"],
        parameters,
    )
    valid_rows = splits["valid"]
    valid_scores = model.predict(valid_rows)
    metrics = evaluate(
        [str(row[1]) for row in valid_rows],
        [int(row[6]) for row in valid_rows],
        valid_scores,
    )
    normalized_metrics = {
        key: int(value) if key in {"users", "rows"} else float(value)
        for key, value in metrics.items()
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "checkpoint.pkl.gz"
    metadata = {
        "parameters": parameters,
        "model_type": model_type,
        "feature_count": model.encoder.feature_count,
    }
    model.save(checkpoint_path, metadata)
    np.save(artifact_dir / "valid_scores.npy", valid_scores, allow_pickle=False)
    duration = time.monotonic() - started
    if verbose:
        print(
            f"model={model_type} rounds={rounds} primary={metrics['primary']:.5f}",
            flush=True,
        )
    return {
        "metrics": normalized_metrics,
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "valid_scores": str(artifact_dir / "valid_scores.npy"),
        "history": [
            {
                "rounds": rounds,
                "metrics": normalized_metrics,
                "duration_seconds": duration,
            }
        ],
        "epochs_completed": rounds,
        "pairs": 0,
        "duration_seconds": duration,
    }
