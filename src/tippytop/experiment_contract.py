"""Runtime and data contract supplied to generated experiment authors."""

from __future__ import annotations

from typing import Any


def experiment_contract() -> dict[str, Any]:
    return {
        "response_schema": {
            "hypothesis": "specific falsifiable research claim",
            "expected_effect": "expected effect on mean(GAUC, nDCG@5)",
            "source": "complete Python module encoded as a JSON string",
        },
        "required_functions": [
            "fit(train_rows, seed)",
            "predict(model, rows)",
        ],
        "data_contract": {
            "representation": "pandas DataFrame",
            "train_columns": [
                "date", "user_id", "video_id", "author_id", "tab", "duration_ms",
                "hourmin", "time_ms", "long_view", "play_time_ms", "is_click", "is_like",
                "is_follow", "is_comment", "is_forward", "is_hate", "profile_stay_time",
                "comment_stay_time", "is_profile_enter",
            ],
            "prediction_columns": [
                "date", "user_id", "video_id", "author_id", "tab", "duration_ms",
                "hourmin", "time_ms",
            ],
            "auxiliary_rule": (
                "play_time and engagement columns exist only in training; use them as auxiliary "
                "supervision or to fit historical aggregates, never as current-row prediction features"
            ),
        },
        "runtime": {
            "allowed_libraries": [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "lightgbm",
                "tippytop.models",
                "tippytop.research",
            ],
            "fit_return": "a pickleable model, optionally (model, JSON-safe metadata)",
            "predict_return": "one finite numeric score per input row",
            "cpu_only": True,
            "no_file_network_process_access": True,
            "serialization": (
                "return the model directly; the trusted host serializes it after fit returns"
            ),
        },
        "available_helpers": {
            "CategoricalEncoder (import from tippytop.research)": (
                "encoder = CategoricalEncoder(); encoder.fit_transform(train_rows) fits it and "
                "returns an int32 [rows, 5] array; encoder.transform(rows) reuses it for "
                "user/video/author/tab/duration-decile fields"
            ),
            "TabularEncoder (import from tippytop.research)": (
                "encoder = TabularEncoder(); encoder.fit_transform(train_rows) returns out-of-fold, "
                "leakage-safe float32 training features and retains the full-train encoder; "
                "encoder.transform(rows) uses only saved training aggregates; it reads only the "
                "starter date/user/video/author/tab/duration fields and ignores hourmin, time_ms, "
                "auxiliary outcomes, and any custom columns added by generated code"
            ),
            "labels (import from tippytop.research)": "labels(train_rows) returns float32 long_view labels",
            "user_ids (import from tippytop.research)": "user_ids(rows) returns string user IDs",
            "FactorizationMachine (import from tippytop.models)": (
                "FactorizationMachine(dimension, embedding_dim=16, learning_rate=0.001, "
                "l2=1e-6, seed=seed); model.fit_pointwise(features, labels, epochs, batch_size, seed) "
                "and model.fit_bpr(features, positive_indices, negative_indices, epochs, batch_size, seed) "
                "run complete seeded minibatch training and return epoch losses; "
                "model.step_pointwise(features, labels) updates one BCE batch; "
                "model.step_bpr(positive_features, negative_features) updates one pairwise batch; "
                "each step call performs exactly one Adam update, not an epoch or fit routine; "
                "model.predict(features) returns ranking logits; "
                "features must be integer IDs from CategoricalEncoder and dimension must equal "
                "encoder.dimension"
            ),
            "build_pair_indices (import from tippytop.models)": (
                "build_pair_indices(labels_array, user_id_list, pairs_per_positive=1, seed=seed) "
                "returns same-user positive and negative row-index arrays"
            ),
        },
        "canonical_imports": [
            "from tippytop.research import CategoricalEncoder, TabularEncoder, labels, user_ids",
            "from tippytop.models import FactorizationMachine, build_pair_indices",
        ],
        "short_tabular_path": [
            "create a TabularEncoder and call X_train = encoder.fit_transform(train_rows)",
            "fit any sklearn or LightGBM estimator on X_train and labels(train_rows)",
            "return (encoder, estimator) as the model bundle",
            "in predict, unpack the bundle and score encoder.transform(rows)",
        ],
        "custom_feature_path": [
            "persist every train-fitted custom transformer or aggregate in the returned model bundle",
            "build X_base with TabularEncoder, build X_custom separately, then pass np.column_stack([X_base, X_custom]) to the estimator",
            "repeat the identical X_custom construction from persisted state in predict",
            "a claimed feature has no effect unless its values are present in the final matrix passed to fit and predict",
            "training-only outcomes cannot be direct X_custom columns; convert them into past-only or train-fitted key aggregates available at prediction",
        ],
        "sparse_pairwise_path": [
            "create CategoricalEncoder and call X_train = encoder.fit_transform(train_rows)",
            "construct FactorizationMachine with dimension=encoder.dimension, never a guessed dimension",
            "optionally call model.fit_pointwise(X_train, y_train, epochs=1 or 2, batch_size=8192, seed=seed) before pairwise tuning",
            "build same-user pairs with build_pair_indices(labels(train_rows), user_ids(train_rows), ...)",
            "call model.fit_bpr(X_train, positive_indices, negative_indices, epochs=several, batch_size=8192, seed=seed)",
            "one step_bpr call over all pairs is only one undertrained optimizer update; use explicit minibatch and epoch loops",
            "return (encoder, model); in predict transform rows with that encoder before model.predict",
            "do not pass TabularEncoder float features to FactorizationMachine",
        ],
        "implementation_guidance": [
            "prefer one compact module under 120 lines over elaborate abstractions",
            "use the documented short tabular path unless measured history justifies another approach",
            "pass complete DataFrames to helper transforms; direct pandas column feature engineering is also allowed",
            "never add a DataFrame column and then expect TabularEncoder to preserve it; concatenate custom numeric features explicitly",
            "verify the final estimator input contains every feature named in the hypothesis",
            "keep fit and predict vectorized enough for 1.14 million training rows",
            "prefer reliable executable code over speculative feature engineering",
        ],
        "research_rules": [
            "fit sees training rows only; never attempt to access validation or test labels",
            "do not perform validation or model selection inside generated source",
            "fit all encoders and aggregates on train_rows only",
            "derive every random generator and estimator random_state from the supplied seed",
            "prediction frames contain only the eight documented impression-time columns",
            "never expect long_view, watch time, or engagement outcomes during prediction",
            "pandas, pickle, and safe standard-library modules are available when useful",
            "optimize within-user ranking, not only global calibration",
            "use the measured history to make a substantive change rather than repeat source",
            "do not repeat a direction identified as flat or a dead end in the research context",
        ],
    }
