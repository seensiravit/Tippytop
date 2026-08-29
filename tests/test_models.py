from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tippytop.experiments import predict_checkpoint, train_parametric
from tippytop.models import FactorizationMachine, build_pair_indices, load_model
from tippytop.research import CategoricalEncoder, labels, user_ids
from tippytop.starter import encode_splits


def test_pair_indices_never_cross_users(synthetic_splits: dict[str, list[tuple[object, ...]]]) -> None:
    encoded, _ = encode_splits(synthetic_splits)
    _, labels, users = encoded["train"]
    positives, negatives = build_pair_indices(labels, users, pairs_per_positive=3, seed=7)
    assert len(positives) == 30
    for positive, negative in zip(positives, negatives, strict=True):
        assert users[positive] == users[negative]
        assert labels[positive] == 1
        assert labels[negative] == 0


def test_bpr_learns_positive_over_negative(
    tmp_path: Path,
    synthetic_splits: dict[str, list[tuple[object, ...]]],
) -> None:
    parameters = {
        "embedding_dim": 4,
        "learning_rate": 0.01,
        "l2": 0.0,
        "epochs": 20,
        "patience": 10,
        "pairs_per_positive": 4,
        "batch_size": 256,
        "seed": 0,
    }
    result = train_parametric(
        synthetic_splits,
        tmp_path,
        model_type="fm",
        objective="bpr",
        parameters=parameters,
    )
    scores = predict_checkpoint(Path(result["checkpoint"]), synthetic_splits, "valid")
    assert scores[0] > scores[1]
    assert scores[2] > scores[3]
    assert result["metrics"]["primary"] == 1.0


def test_generated_pairwise_public_api_executes(
    synthetic_splits: dict[str, list[tuple[object, ...]]],
) -> None:
    rows = synthetic_splits["train"]
    encoder = CategoricalEncoder()
    features = encoder.fit_transform(rows)
    positive, negative = build_pair_indices(labels(rows), user_ids(rows), seed=0)
    model = FactorizationMachine(encoder.dimension, embedding_dim=4, seed=0)

    pointwise_history = model.fit_pointwise(
        features,
        labels(rows),
        epochs=1,
        batch_size=4,
        seed=0,
    )
    pairwise_history = model.fit_bpr(
        features,
        positive,
        negative,
        epochs=2,
        batch_size=4,
        seed=0,
    )
    scores = model.predict(encoder.transform(synthetic_splits["valid"]))

    assert len(pointwise_history) == 1
    assert len(pairwise_history) == 2
    assert np.isfinite(pointwise_history + pairwise_history).all()
    assert model.t > 3
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()


def test_categorical_encoder_fit_transform_accepts_dataframes(
    synthetic_splits: dict[str, list[tuple[object, ...]]],
) -> None:
    frame = pd.DataFrame(
        synthetic_splits["train"],
        columns=["date", "user_id", "video_id", "author_id", "tab", "duration_ms", "long_view"],
    )
    encoder = CategoricalEncoder()

    fitted_features = encoder.fit_transform(frame)

    assert fitted_features.dtype == np.int32
    assert np.array_equal(fitted_features, encoder.transform(frame))
    assert int(fitted_features.max()) < encoder.dimension


def test_categorical_ranker_rejects_wrong_feature_representation() -> None:
    model = FactorizationMachine(8, embedding_dim=4, seed=0)

    with pytest.raises(ValueError, match="CategoricalEncoder"):
        model.predict(np.zeros((2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="encoder.dimension"):
        model.predict(np.full((2, 5), 8, dtype=np.int32))
    with pytest.raises(ValueError, match="non-empty"):
        model.step_bpr(
            np.empty((0, 5), dtype=np.int32),
            np.empty((0, 5), dtype=np.int32),
        )
    oversized = np.zeros((65_537, 5), dtype=np.int32)
    with pytest.raises(ValueError, match="use fit_bpr"):
        model.step_bpr(oversized, oversized)
    with pytest.raises(ValueError, match="use fit_pointwise"):
        model.step_pointwise(oversized, np.zeros(len(oversized), dtype=np.float32))


def test_all_parametric_model_families_checkpoint(
    tmp_path: Path,
    synthetic_splits: dict[str, list[tuple[object, ...]]],
) -> None:
    for model_type in ("linear", "mf", "fm", "ffm"):
        parameters = {
            "learning_rate": 0.01,
            "l2": 0.0,
            "epochs": 4,
            "patience": 2,
            "pairs_per_positive": 2,
            "batch_size": 256,
            "seed": 0,
        }
        if model_type != "linear":
            parameters["embedding_dim"] = 4
        result = train_parametric(
            synthetic_splits,
            tmp_path / model_type,
            model_type=model_type,
            objective="bpr",
            parameters=parameters,
        )
        model, metadata = load_model(Path(result["checkpoint"]))
        scores = predict_checkpoint(Path(result["checkpoint"]), synthetic_splits, "valid")
        assert metadata["model_type"] == model_type
        assert model.dimension > 0
        assert scores.shape == (4,)
        assert np.isfinite(scores).all()
