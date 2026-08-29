"""Trusted diagnostics that make generated-code repair prompts actionable."""

from __future__ import annotations

import inspect
import re
from typing import Any

from ..research.data import AUXILIARY_COLUMNS, PREDICTION_COLUMNS, TRAINING_COLUMNS


_UNEXPECTED_KEYWORD = re.compile(
    r"(?P<class_name>[A-Za-z_][A-Za-z0-9_]*)\.(?P<method>[A-Za-z_][A-Za-z0-9_]*)\(\) "
    r"got an unexpected keyword argument ['\"](?P<keyword>[^'\"]+)['\"]"
)
_GROUP_SIZE_MISMATCH = re.compile(
    r"Sum of query counts \((?P<group_rows>\d+)\) differs from the length of #data "
    r"\((?P<data_rows>\d+)\)"
)
_MISSING_COLUMNS = re.compile(r"Columns not found: (?P<columns>.+)")
_PANDAS_KEY_ERROR = re.compile(r"KeyError: ['\"](?P<column>[^'\"]+)['\"]")
_INFEASIBLE_SMOKE = re.compile(
    r"(?P<smoke_seconds>[0-9.]+)s for (?P<sample_rows>\d+) sampled rows projects to at least "
    r"(?P<projected_seconds>[0-9.]+)s for (?P<full_rows>\d+) rows"
)


def runtime_failure_diagnostics(source: str, error: str) -> dict[str, Any]:
    """Return installed API facts relevant to a generated runtime failure."""

    infeasible_match = _INFEASIBLE_SMOKE.search(error)
    if infeasible_match is not None:
        return {
            "smoke_seconds": float(infeasible_match.group("smoke_seconds")),
            "sample_rows": int(infeasible_match.group("sample_rows")),
            "projected_full_seconds": float(infeasible_match.group("projected_seconds")),
            "full_rows": int(infeasible_match.group("full_rows")),
            "required_action": (
                "The fit path must scale to the full training set. Remove Python row loops, per-row "
                "global scans/sorts, and repeated whole-frame work; use vectorized pandas/NumPy, bounded "
                "group operations, or estimator-native batching while preserving the research objective."
            ),
        }

    missing_match = _MISSING_COLUMNS.search(error)
    prediction_failure = re.search(r"\bin predict\b", error) is not None
    if missing_match is not None and prediction_failure:
        missing = sorted(set(re.findall(r"'([^']+)'", missing_match.group("columns"))))
        training_only = {"long_view", *AUXILIARY_COLUMNS}
        if missing and set(missing) <= training_only:
            return {
                "missing_prediction_columns": missing,
                "prediction_columns": [
                    "date",
                    "user_id",
                    "video_id",
                    "author_id",
                    "tab",
                    "duration_ms",
                    "hourmin",
                    "time_ms",
                ],
                "required_action": (
                    "Prediction rows never contain labels or auxiliary outcomes. Fit every outcome-derived "
                    "aggregate on training rows, persist the resulting maps/statistics in the returned model "
                    "state, and make predict apply only that stored state to prediction-time columns. Do not "
                    "group or aggregate long_view, play_time_ms, click, like, or other outcomes inside predict."
                ),
            }

    key_errors = _PANDAS_KEY_ERROR.findall(error)
    if key_errors and ("pandas" in error or ".groupby(" in source):
        missing_column = key_errors[-1]
        canonical_by_normalized = {
            column.replace("_", "").lower(): column for column in TRAINING_COLUMNS
        }
        canonical = canonical_by_normalized.get(missing_column.replace("_", "").lower())
        if canonical is not None and canonical != missing_column:
            required_action = (
                f"Generated input columns are case-sensitive: use {canonical!r}, never "
                f"{missing_column!r}. Audit every reference and do not invent alternate spellings."
            )
        elif canonical is not None:
            required_action = (
                f"The original generated input does contain {canonical!r}; this KeyError means a derived "
                "DataFrame dropped or renamed it. Preserve a separate query/group vector or calculate group "
                "sizes before selecting numeric feature columns. Do not rename the field to UserID."
            )
        else:
            required_action = (
                "The missing name is not part of the generated input schema. Derive it explicitly before use "
                "or correct the reference; do not guess a different capitalization."
            )
        return {
            "missing_dataframe_column": missing_column,
            "canonical_column": canonical,
            "training_columns": list(TRAINING_COLUMNS),
            "prediction_columns": list(PREDICTION_COLUMNS),
            "required_action": required_action,
        }

    group_match = _GROUP_SIZE_MISMATCH.search(error)
    if group_match is not None:
        return {
            "reported_query_count": int(group_match.group("group_rows")),
            "reported_data_length": int(group_match.group("data_rows")),
            "required_action": (
                "For LGBMRanker, sum(group) must equal len(X) and len(y), and each group must "
                "describe one contiguous query block. Build X and y from every positive and negative "
                "row represented by group; do not train on only the positive subset."
            ),
        }

    match = _UNEXPECTED_KEYWORD.search(error)
    if match is None:
        return {}

    class_name = match.group("class_name")
    method_name = match.group("method")
    diagnostics: dict[str, Any] = {
        "unexpected_keyword": match.group("keyword"),
        "failing_call": f"{class_name}.{method_name}",
    }
    if "lightgbm" not in source.lower() and "lgb" not in source:
        return diagnostics

    try:
        import lightgbm

        estimator = getattr(lightgbm, class_name)
        method = getattr(estimator, method_name)
    except (AttributeError, ImportError):
        return diagnostics

    diagnostics["installed_signatures"] = {
        "constructor": f"{class_name}{inspect.signature(estimator)}",
        method_name: f"{class_name}.{method_name}{inspect.signature(method)}",
    }
    diagnostics["required_action"] = (
        "Audit every keyword at this call site against the installed method signature. "
        "Move estimator configuration to the constructor and remove all unsupported method keywords "
        "in one repair, including any passed through **kwargs or parameter dictionaries."
    )
    return diagnostics
