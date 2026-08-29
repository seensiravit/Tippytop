from __future__ import annotations

import time
from pathlib import Path

import pytest

from tippytop.artifacts import RunStore, read_jsonl
from tippytop.llm import LLMResult
from tippytop.search_iteration import (
    REFLECTION_CODE_CHARS,
    _reflect,
    _validation_diagnostics,
)
from tippytop.search_journal import commit_iteration, recover_transactions


def test_iteration_transaction_recovers_state_and_log(tmp_path: Path) -> None:
    (tmp_path / "transactions").mkdir()
    store = RunStore(tmp_path)
    state = {"iteration": 1, "best": {"id": "new-best"}, "elapsed_seconds": 0.0}
    record = {"iteration": 1, "id": "iteration-001", "status": "completed"}

    commit_iteration(store, state, record, time.monotonic())
    stale_state = {"iteration": 0, "best": {"id": "baseline"}, "elapsed_seconds": 0.0}
    store.write_json("state.json", stale_state)

    recover_transactions(store, stale_state)
    recover_transactions(store, stale_state)

    assert stale_state["iteration"] == 1
    assert stale_state["best"]["id"] == "new-best"
    assert read_jsonl(tmp_path / "iterations.jsonl") == [record]


def test_reflection_bounds_generated_code_diff(tmp_path: Path) -> None:
    class RecordingClient:
        context: dict[str, object] | None = None

        def reflect(self, context: dict[str, object], *, deadline: float) -> LLMResult:
            self.context = context
            return LLMResult("measured reflection", {})

    client = RecordingClient()
    record = {
        "iteration": 1,
        "hypothesis": "Measure a substantive change.",
        "code_diff": "prefix\n" + "x" * 20_000 + "\nsuffix",
    }

    result = _reflect(
        client,  # type: ignore[arg-type]
        {"baseline_valid": {"primary": 0.6}},
        record,
        RunStore(tmp_path),
        {"primary": 0.6},
        time.monotonic() + 10,
    )

    assert result is not None
    assert client.context is not None
    experiment = client.context["experiment"]
    assert isinstance(experiment, dict)
    code_diff = experiment["code_diff"]
    assert isinstance(code_diff, str)
    assert len(code_diff) == REFLECTION_CODE_CHARS
    assert code_diff.startswith("prefix")
    assert code_diff.endswith("suffix")
    assert "truncated for reflection" in code_diff


def test_validation_diagnostics_identify_repeated_outcome() -> None:
    baseline = {"GAUC": 0.66, "nDCG@5": 0.54, "primary": 0.60}
    repeated = {"GAUC": 0.65, "nDCG@5": 0.53, "primary": 0.59}

    diagnostics = _validation_diagnostics(
        repeated,
        baseline,
        baseline,
        [{"iteration": 2, "metrics": repeated}],
    )

    assert diagnostics["matching_prior_iteration"] == 2
    assert diagnostics["primary_delta_from_baseline"] == pytest.approx(-0.01)
    assert diagnostics["primary_delta_from_previous_best"] == pytest.approx(-0.01)
