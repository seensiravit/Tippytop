from __future__ import annotations

from pathlib import Path

import numpy as np

from tippytop.experiments import predict_checkpoint, train_tree
from tippytop.models.trees import TreeFeatureEncoder, TreeModel


def test_leave_one_out_features_remove_the_current_label(
    synthetic_splits: dict[str, list[tuple[object, ...]]],
) -> None:
    rows = synthetic_splits["train"]
    encoder = TreeFeatureEncoder.fit(rows, smoothing=1.0)
    train_features = encoder.transform(rows, leave_one_out=True)
    normal_features = encoder.transform(rows)
    assert train_features.shape[1] == encoder.feature_count
    assert not np.array_equal(train_features, normal_features)
    assert np.isfinite(train_features).all()


def test_cross_fitted_row_features_do_not_depend_on_own_label(
    synthetic_splits: dict[str, list[tuple[object, ...]]],
) -> None:
    rows = list(synthetic_splits["train"])
    _, before = TreeFeatureEncoder.fit_transform_training(rows, smoothing=10.0)
    changed = list(rows)
    changed[0] = (*changed[0][:6], 1 - int(changed[0][6]))
    _, after = TreeFeatureEncoder.fit_transform_training(changed, smoothing=10.0)

    np.testing.assert_array_equal(before[0], after[0])


def test_tree_model_families_train_and_reload(
    tmp_path: Path,
    synthetic_splits: dict[str, list[tuple[object, ...]]],
) -> None:
    model_parameters = {
        "random_forest": {
            "n_estimators": 10,
            "max_depth": 4,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "smoothing": 1.0,
            "seed": 0,
        },
        "gradient_boosting": {
            "max_iter": 10,
            "max_leaf_nodes": 4,
            "min_samples_leaf": 1,
            "tree_learning_rate": 0.1,
            "l2_regularization": 0.0,
            "smoothing": 1.0,
            "seed": 0,
        },
        "lightgbm_ranker": {
            "n_estimators": 10,
            "num_leaves": 4,
            "max_depth": 3,
            "min_child_samples": 1,
            "patience": 2,
            "tree_learning_rate": 0.1,
            "reg_lambda": 0.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "smoothing": 1.0,
            "seed": 0,
        },
    }
    for model_type, parameters in model_parameters.items():
        result = train_tree(
            synthetic_splits,
            tmp_path / model_type,
            model_type=model_type,
            parameters=parameters,
        )
        checkpoint = Path(result["checkpoint"])
        model, metadata = TreeModel.load(checkpoint)
        scores = predict_checkpoint(checkpoint, synthetic_splits, "valid")
        assert model.model_type == model_type
        assert metadata["model_type"] == model_type
        assert scores.shape == (4,)
        assert np.isfinite(scores).all()
