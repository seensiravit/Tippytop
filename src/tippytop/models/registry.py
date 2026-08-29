"""Model construction and architecture-aware checkpoint loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .base import CategoricalRanker
from .factorization_machine import FactorizationMachine
from .field_aware import FieldAwareFactorizationMachine
from .linear import LinearModel
from .matrix_factorization import MatrixFactorization


MODEL_TYPES = {"linear", "mf", "fm", "ffm"}


def build_model(
    model_type: str,
    dimension: int,
    field_count: int,
    parameters: dict[str, int | float],
) -> CategoricalRanker:
    common = {
        "learning_rate": float(parameters["learning_rate"]),
        "l2": float(parameters["l2"]),
        "seed": int(parameters["seed"]),
    }
    if model_type == "linear":
        return LinearModel(dimension, **common)
    latent = {"embedding_dim": int(parameters["embedding_dim"]), **common}
    if model_type == "mf":
        return MatrixFactorization(dimension, **latent)
    if model_type == "fm":
        return FactorizationMachine(dimension, **latent)
    if model_type == "ffm":
        return FieldAwareFactorizationMachine(dimension, field_count=field_count, **latent)
    raise ValueError(f"unsupported model type: {model_type!r}")


def load_model(path: Path) -> tuple[CategoricalRanker, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as value:
        metadata = json.loads(str(value["metadata"].item()))
        matrix = value["V"].copy()
        weights = value["W"].copy()
        bias = np.float32(value["b"].item())
    model = build_model(
        str(metadata["model_type"]),
        int(metadata["dimension"]),
        int(metadata["field_count"]),
        metadata["parameters"],
    )
    if model.V.shape != matrix.shape or model.W.shape != weights.shape:
        raise ValueError("checkpoint parameter shapes do not match its metadata")
    model.V = matrix
    model.W = weights
    model.b = bias
    return model, metadata
