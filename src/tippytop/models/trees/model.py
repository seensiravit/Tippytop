"""Tree estimator construction, fitting, prediction, and persistence."""

from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from .features import TreeFeatureEncoder


TREE_MODEL_TYPES = {"random_forest", "gradient_boosting", "lightgbm_ranker"}


@dataclass
class TreeModel:
    model_type: str
    encoder: TreeFeatureEncoder
    estimator: Any

    def predict(self, rows: Sequence[tuple[Any, ...]]) -> np.ndarray:
        features = self.encoder.transform(rows)
        if self.model_type == "lightgbm_ranker":
            return np.asarray(self.estimator.predict(features), dtype=np.float32)
        classes = np.asarray(self.estimator.classes_)
        positive = np.flatnonzero(classes == 1)
        if not len(positive):
            return np.zeros(len(rows), dtype=np.float32)
        probabilities = self.estimator.predict_proba(features)
        return np.asarray(probabilities[:, positive[0]], dtype=np.float32)

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        with gzip.open(path, "wb", compresslevel=3) as handle:
            pickle.dump(
                {"model": self, "metadata": metadata},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> tuple["TreeModel", dict[str, Any]]:
        # Checkpoints are generated and consumed within one trusted run directory.
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        model = payload["model"]
        if not isinstance(model, cls) or model.model_type not in TREE_MODEL_TYPES:
            raise ValueError("invalid tree checkpoint")
        return model, payload["metadata"]


def fit_tree_model(
    model_type: str,
    train_rows: Sequence[tuple[Any, ...]],
    valid_rows: Sequence[tuple[Any, ...]],
    parameters: dict[str, int | float],
) -> tuple[TreeModel, int]:
    encoder, train_x = TreeFeatureEncoder.fit_transform_training(
        train_rows,
        smoothing=float(parameters["smoothing"]),
    )
    train_y = np.asarray([int(row[6]) for row in train_rows], dtype=np.int8)

    if model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=int(parameters["max_depth"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=float(parameters["max_features"]),
            n_jobs=-1,
            random_state=int(parameters["seed"]),
        )
        estimator.fit(train_x, train_y)
        rounds = int(parameters["n_estimators"])
    elif model_type == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(parameters["tree_learning_rate"]),
            max_iter=int(parameters["max_iter"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=float(parameters["l2_regularization"]),
            early_stopping=False,
            random_state=int(parameters["seed"]),
        )
        estimator.fit(train_x, train_y)
        rounds = int(estimator.n_iter_)
    elif model_type == "lightgbm_ranker":
        valid_x = encoder.transform(valid_rows)
        valid_y = np.asarray([int(row[6]) for row in valid_rows], dtype=np.int8)
        train_order, train_groups = _group_order([str(row[1]) for row in train_rows])
        valid_order, valid_groups = _group_order([str(row[1]) for row in valid_rows])
        estimator = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=int(parameters["n_estimators"]),
            learning_rate=float(parameters["tree_learning_rate"]),
            num_leaves=int(parameters["num_leaves"]),
            max_depth=int(parameters["max_depth"]),
            min_child_samples=int(parameters["min_child_samples"]),
            reg_lambda=float(parameters["reg_lambda"]),
            subsample=float(parameters["subsample"]),
            subsample_freq=1,
            colsample_bytree=float(parameters["colsample_bytree"]),
            label_gain=[0, 1],
            n_jobs=-1,
            random_state=int(parameters["seed"]),
            verbosity=-1,
        )
        estimator.fit(
            train_x[train_order],
            train_y[train_order],
            group=train_groups,
            eval_X=valid_x[valid_order],
            eval_y=valid_y[valid_order],
            eval_group=[valid_groups],
            eval_at=[5],
            callbacks=[
                lgb.early_stopping(int(parameters["patience"]), verbose=False),
                lgb.log_evaluation(period=0),
            ],
            categorical_feature=encoder.categorical_indices,
        )
        rounds = int(estimator.best_iteration_ or parameters["n_estimators"])
    else:
        raise ValueError(f"unsupported tree model: {model_type!r}")
    return TreeModel(model_type, encoder, estimator), rounds


def _group_order(users: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(users)
    # LambdaRank requires each query's rows to be contiguous; stable order keeps replay deterministic.
    order = np.argsort(values, kind="stable")
    _, counts = np.unique(values[order], return_counts=True)
    return order, counts.astype(np.int32)
