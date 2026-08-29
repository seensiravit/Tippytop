"""Validation-selected training for trusted parametric reference models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from tippytop.models import ModelState, build_model, build_pair_indices, iter_batches
from tippytop.starter import encode_splits, evaluate

from .scoring import blend_scores


def train_parametric(
    splits: dict[str, list[tuple[Any, ...]]],
    artifact_dir: Path,
    *,
    model_type: str,
    objective: str,
    parameters: dict[str, int | float],
    popularity_weight: float = 0.0,
    verbose: bool = False,
) -> dict[str, Any]:
    encoded, dimension = encode_splits(splits)
    train_x, train_y, train_users = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    model = build_model(model_type, dimension, train_x.shape[1], parameters)
    rng = np.random.default_rng(int(parameters["seed"]))
    batch_size = int(parameters["batch_size"])
    ranking_weight = _pairwise_weight(objective, parameters)
    classification_weight = 1.0 - ranking_weight

    positive_indices = negative_indices = np.empty(0, dtype=np.int64)
    if ranking_weight > 0:
        positive_indices, negative_indices = build_pair_indices(
            train_y,
            train_users,
            pairs_per_positive=int(parameters.get("pairs_per_positive", 1)),
            seed=int(parameters["seed"]),
        )
        if len(positive_indices) == 0:
            raise ValueError("pairwise training requires at least one discriminative user")

    best_primary = -1.0
    best_state: ModelState | None = None
    best_metrics: dict[str, Any] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.monotonic()

    for epoch in range(1, int(parameters["epochs"]) + 1):
        losses: list[float] = []
        epoch_started = time.monotonic()
        if classification_weight > 0:
            shuffled = rng.permutation(len(train_y))
            for batch in iter_batches(shuffled, batch_size):
                losses.append(
                    model.step_pointwise(
                        train_x[batch],
                        train_y[batch],
                        weight=classification_weight,
                    )
                )
        if ranking_weight > 0:
            shuffled_pairs = rng.permutation(len(positive_indices))
            for batch in iter_batches(shuffled_pairs, batch_size):
                losses.append(
                    model.step_bpr(
                        train_x[positive_indices[batch]],
                        train_x[negative_indices[batch]],
                        weight=ranking_weight,
                    )
                )

        valid_scores = blend_scores(
            model.predict(valid_x),
            splits,
            "valid",
            popularity_weight=popularity_weight,
        )
        metrics = evaluate(valid_users, valid_y, valid_scores)
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "metrics": _json_metrics(metrics),
            "duration_seconds": time.monotonic() - epoch_started,
        }
        history.append(record)
        if verbose:
            print(
                f"epoch={epoch} model={model_type} loss={record['loss']:.5f} "
                f"primary={metrics['primary']:.5f}",
                flush=True,
            )
        # Validation chooses an epoch checkpoint; test rows never enter this trainer.
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = float(metrics["primary"])
            best_state = model.snapshot()
            best_metrics = _json_metrics(metrics)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(parameters["patience"]):
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("training completed without a valid checkpoint")
    model.restore(best_state)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "checkpoint.npz"
    metadata = {
        "parameters": parameters,
        "model_type": model_type,
        "objective": objective,
        "popularity_weight": popularity_weight,
        "dimension": dimension,
        "field_count": int(train_x.shape[1]),
    }
    model.save(checkpoint_path, metadata)
    valid_scores = blend_scores(
        model.predict(valid_x),
        splits,
        "valid",
        popularity_weight=popularity_weight,
    )
    np.save(artifact_dir / "valid_scores.npy", valid_scores, allow_pickle=False)
    return {
        "metrics": best_metrics,
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "valid_scores": str(artifact_dir / "valid_scores.npy"),
        "history": history,
        "epochs_completed": len(history),
        "pairs": int(len(positive_indices)),
        "duration_seconds": time.monotonic() - started,
    }


def _json_metrics(metrics: dict[str, Any]) -> dict[str, int | float]:
    return {
        key: int(value) if key in {"users", "rows"} else float(value)
        for key, value in metrics.items()
    }


def _pairwise_weight(objective: str, parameters: dict[str, int | float]) -> float:
    if objective == "pointwise":
        return 0.0
    if objective == "bpr":
        return 1.0
    if objective == "hybrid":
        return float(parameters.get("pairwise_weight", 0.5))
    raise ValueError(f"unsupported objective: {objective!r}")
