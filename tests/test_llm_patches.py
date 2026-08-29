from __future__ import annotations

import pytest

from tippytop.generated import GeneratedExperiment
from tippytop.llm.patches import apply_repair_payload


SOURCE = """def fit(train_rows, seed):
    return 1

def predict(model, rows):
    return [model] * len(rows)
"""


def experiment(source: str = SOURCE) -> GeneratedExperiment:
    return GeneratedExperiment("test repair", "execute successfully", source)


def test_repair_edits_are_applied_and_revalidated() -> None:
    repaired = apply_repair_payload(
        experiment(),
        {"edits": [{"old": "return 1", "new": "return float(seed)"}]},
    )

    assert "return float(seed)" in repaired.source
    assert repaired.hypothesis == "test repair"


def test_repair_edit_must_match_exactly_once() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        apply_repair_payload(
            experiment(),
            {"edits": [{"old": "return", "new": "yield"}]},
        )


def test_repair_edit_cannot_create_invalid_module() -> None:
    with pytest.raises(ValueError, match="valid Python"):
        apply_repair_payload(
            experiment(),
            {"edits": [{"old": "return 1", "new": "return ("}]},
        )


def test_complete_source_replacement_cannot_bypass_patch_constraints() -> None:
    with pytest.raises(ValueError, match="exactly one 'edits' field"):
        apply_repair_payload(experiment(), experiment().to_dict())
