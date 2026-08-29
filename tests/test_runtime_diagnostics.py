from __future__ import annotations

from tippytop.runtime.diagnostics import runtime_failure_diagnostics


def test_unexpected_lightgbm_keyword_includes_installed_signatures() -> None:
    diagnostics = runtime_failure_diagnostics(
        "import lightgbm as lgb\nmodel = lgb.LGBMRanker()",
        "TypeError: LGBMRanker.fit() got an unexpected keyword argument 'objective'",
    )

    assert diagnostics["unexpected_keyword"] == "objective"
    assert "objective" not in diagnostics["installed_signatures"]["fit"]
    assert "LGBMRanker.fit" in diagnostics["installed_signatures"]["fit"]
    assert "constructor" in diagnostics["installed_signatures"]


def test_ranker_group_size_mismatch_explains_the_required_invariant() -> None:
    diagnostics = runtime_failure_diagnostics(
        "import lightgbm as lgb",
        "LightGBMError: Sum of query counts (3398) differs from the length of #data (37378)",
    )

    assert diagnostics["reported_query_count"] == 3398
    assert diagnostics["reported_data_length"] == 37378
    assert diagnostics["kind"] == "ranker_group_size_mismatch"
    assert "positive and negative" in diagnostics["required_action"]


def test_missing_prediction_outcomes_require_persisted_training_state() -> None:
    diagnostics = runtime_failure_diagnostics(
        "import pandas as pd",
        'File "experiment.py", line 10, in predict\n'
        'KeyError: "Columns not found: \'long_view\', \'play_time_ms\', \'is_click\'"',
    )

    assert diagnostics["missing_prediction_columns"] == [
        "is_click",
        "long_view",
        "play_time_ms",
    ]
    assert "persist" in diagnostics["required_action"]
    assert "inside predict" in diagnostics["required_action"]


def test_missing_column_text_outside_predict_is_not_promoted_to_environment_fact() -> None:
    diagnostics = runtime_failure_diagnostics(
        "import pandas as pd",
        'File "experiment.py", line 10, in fit\n'
        'KeyError: "Columns not found: \'long_view\'"',
    )

    assert diagnostics == {}


def test_pandas_key_error_reports_the_canonical_case_sensitive_column() -> None:
    diagnostics = runtime_failure_diagnostics(
        "import pandas as pd\nframe.groupby('UserID')",
        "pandas/core/groupby/grouper.py\nKeyError: 'UserID'",
    )

    assert diagnostics["missing_dataframe_column"] == "UserID"
    assert diagnostics["kind"] == "missing_dataframe_column"
    assert diagnostics["canonical_column"] == "user_id"
    assert "case-sensitive" in diagnostics["required_action"]


def test_pandas_key_error_explains_when_a_derived_frame_dropped_group_id() -> None:
    diagnostics = runtime_failure_diagnostics(
        "X_train_sparse.groupby('user_id')",
        "pandas/core/groupby/grouper.py\nKeyError: 'user_id'",
    )

    assert diagnostics["canonical_column"] == "user_id"
    assert "derived" in diagnostics["required_action"]
    assert "group sizes" in diagnostics["required_action"]


def test_infeasible_smoke_requires_vectorized_full_data_fit() -> None:
    diagnostics = runtime_failure_diagnostics(
        "for _, row in train_rows.iterrows(): pass",
        "generated smoke fit is not full-data feasible: 20.0s for 10000 sampled rows projects to "
        "at least 1505.0s for 1000000 rows, exceeding the remaining budget",
    )

    assert diagnostics["sample_rows"] == 10_000
    assert diagnostics["full_rows"] == 1_000_000
    assert "row loops" in diagnostics["required_action"]
