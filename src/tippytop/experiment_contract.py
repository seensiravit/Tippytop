"""Research environment and runtime contract supplied to the LLM stages."""

from __future__ import annotations

from typing import Any


PREDICTION_COLUMNS = [
    "date",
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "duration_ms",
    "hourmin",
    "time_ms",
]

TRAINING_ONLY_COLUMNS = [
    "long_view",
    "play_time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
]


def research_environment() -> dict[str, Any]:
    """Describe the scientific freedom without anchoring the model to code recipes."""

    return {
        "objective": "maximize mean(GAUC, nDCG@5) for within-user impression ranking",
        "data": {
            "representation": "pandas DataFrame",
            "training_columns": [*PREDICTION_COLUMNS, *TRAINING_ONLY_COLUMNS],
            "prediction_columns": PREDICTION_COLUMNS,
            "training_only_rule": (
                "labels and engagement/watch outcomes may be used for supervision or causal, "
                "train-fitted state; predict never receives them"
            ),
        },
        "evaluation": {
            "generated_code_sees": "training rows only during fit and feature-only rows during predict",
            "trusted_host_owns": "public-validation labels, frozen metrics, selection, and test finalization",
        },
        "compute": {
            "cpu_only": True,
            "training_rows": 1_141_112,
            "libraries": ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm"],
            "sandbox": "networkless with bounded wall time and memory",
        },
        "scientific_freedom": [
            "design custom classes, transformers, losses, ensembles, or multi-stage training",
            "use raw DataFrame columns directly; host helpers are optional conveniences",
            "exploit causal temporal histories and training-only outcomes when inference can reproduce the features",
            "change representation, supervision, objective, model family, or combinations of them",
            "write as much clean vectorized code as the substantive experiment requires",
        ],
        "hard_constraints": [
            "no validation or test access inside generated code",
            "all learned state must be returned in a pickleable model bundle",
            "predict must return one finite continuous ranking score per row",
            "fit/predict must be deterministic from the supplied seed",
            "full-data work must be vectorized or batched",
        ],
    }


def experiment_contract() -> dict[str, Any]:
    return {
        "response_schema": {
            "hypothesis": "the plan's falsifiable research claim",
            "expected_effect": "the plan's expected effect on mean(GAUC, nDCG@5)",
            "source": "complete Python module encoded as a JSON string",
        },
        "required_functions": ["fit(train_rows, seed)", "predict(model, rows)"],
        "environment": research_environment(),
        "runtime": {
            "additional_allowed_imports": [
                "safe Python standard library",
                "joblib",
                "threadpoolctl",
                "typing_extensions",
                "tippytop.models",
                "tippytop.research",
            ],
            "fit_return": "a pickleable model, optionally (model, JSON-safe metadata)",
            "predict_return": "one finite numeric score per input row",
            "no_file_network_process_access": True,
            "serialization": "return state directly; the trusted host serializes it",
        },
        "optional_host_api": {
            "tippytop.research": {
                "labels(train_rows)": "float32 long_view labels",
                "user_ids(rows)": "string user IDs",
                "CategoricalEncoder": (
                    "five-field integer-ID encoder; fit_transform(train_rows), transform(rows), dimension"
                ),
                "TabularEncoder": (
                    "starter-field dense encoder; ignores hour/time, auxiliary outcomes, and custom columns"
                ),
            },
            "tippytop.models": {
                "FactorizationMachine": (
                    "integer categorical model with fit_pointwise, fit_bpr, and predict"
                ),
                "build_pair_indices": "same-user positive/negative row-index sampler",
            },
            "warning": (
                "These helpers are optional, not the search space. Do not substitute them for the "
                "research plan or assume added DataFrame columns flow through an encoder."
            ),
        },
        "implementation_requirements": [
            "implement the supplied research plan faithfully rather than falling back to a familiar baseline",
            "persist every fitted encoder, aggregate, model, and ensemble weight needed by predict",
            "trace every claimed feature into the actual fit matrix and reconstruct it in predict",
            "convert training-only outcomes into valid supervision or causal/train-fitted state, never inference columns",
            "use probability, decision, regression, or logit scores rather than predicted class labels",
            "derive random generators and estimator random_state values from seed",
            "avoid internal validation/model selection because generated code receives no validation labels",
        ],
    }
