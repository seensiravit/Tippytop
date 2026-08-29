from __future__ import annotations

import time
from pathlib import Path

import pytest

from tippytop.artifacts import RunStore, read_json, read_jsonl
from tippytop.llm import GenerationFailure, LLMResult
from tippytop.config import RunConfig
from tippytop.generated import GeneratedExperiment
from tippytop.research import ResearchPlan
from tippytop.search.execution import execute_with_repairs
from tippytop.search.records import (
    REFLECTION_CODE_CHARS,
    reflect,
    validation_diagnostics,
)
from tippytop.search.journal import (
    begin_attempt,
    commit_iteration,
    recover_inflight_attempts,
    recover_transactions,
)


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


def test_interrupted_repair_preserves_exact_source_and_responses(tmp_path: Path) -> None:
    (tmp_path / "transactions").mkdir()
    (tmp_path / "experiments").mkdir()
    store = RunStore(tmp_path)
    source = "def fit(train_rows, seed):\n    return 1\n\ndef predict(model, rows):\n    return [model] * len(rows)\n"
    attempt = {
        "iteration": 1,
        "experiment_id": "iteration-001-repair-1",
        "parent_id": "baseline-fm",
        "hypothesis": "Repair the generated model.",
        "expected_effect": "Complete execution.",
        "source_hash": "repaired-hash",
        "source_hashes": ["initial-hash", "repaired-hash"],
        "source_revision": "revision",
        "started_at": "2026-01-01T00:00:00Z",
        "research_plan": {"hypothesis": "plan"},
        "research_responses": [{"content": "plan response"}],
        "responses": [{"content": "repair response"}],
        "experiment": {
            "hypothesis": "Repair the generated model.",
            "expected_effect": "Complete execution.",
            "source": source,
        },
        "recovery": [{"action": "request_llm_code_repair"}],
    }
    begin_attempt(store, attempt)
    state = {
        "iteration": 0,
        "used_source_hashes": [],
        "manual_interventions": 0,
        "elapsed_seconds": 0.0,
    }

    recover_inflight_attempts(store, state, time.monotonic())

    record = read_json(tmp_path / "transactions" / "001.json")["record"]
    assert record["executed_experiment"]["source"] == source
    assert record["responses"] == [{"content": "repair response"}]
    assert state["used_source_hashes"] == ["initial-hash", "repaired-hash"]


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

    result = reflect(
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

    diagnostics = validation_diagnostics(
        repeated,
        baseline,
        baseline,
        [{"iteration": 2, "metrics": repeated}],
    )

    assert diagnostics["matching_prior_iteration"] == 2
    assert diagnostics["primary_delta_from_baseline"] == pytest.approx(-0.01)
    assert diagnostics["primary_delta_from_previous_best"] == pytest.approx(-0.01)


def test_execution_deadline_is_persisted_as_a_durable_outcome(tmp_path: Path) -> None:
    for directory in ("checkpoints", "diffs", "transactions"):
        (tmp_path / directory).mkdir()
    store = RunStore(tmp_path)
    state = {"elapsed_seconds": 0.0}
    experiment = GeneratedExperiment(
        "Test deadline routing",
        "Stop without losing state",
        "def fit(train_rows, seed):\n    return 1\n\ndef predict(model, rows):\n    return [model] * len(rows)\n",
    )
    plan = ResearchPlan(
        "Test deadline routing",
        "Stop without losing state",
        "Exercise the deadline branch.",
        "No prior work.",
        (),
        "Constant smoke model",
        ("Attempt execution",),
        ("Deadline expires",),
    )
    attempt = {
        "iteration": 1,
        "experiment_id": "iteration-001",
        "recovery": [],
    }

    outcome = execute_with_repairs(
        RunConfig(),
        store,
        state,
        object(),  # type: ignore[arg-type]
        {},
        plan,
        experiment,
        "iteration-001",
        {experiment.source_hash},
        [],
        tmp_path / "unused.pkl",
        "revision",
        time.monotonic() - 1,
        time.monotonic(),
        attempt,
    )

    assert outcome.result is None
    assert state["stopping_reason"] == "wall_clock_limit"
    inflight = read_json(tmp_path / "transactions" / "001.inflight.json")
    assert inflight["recovery"][-1]["action"] == "wall_clock_limit_restore_previous_best"


def test_invalid_repair_response_does_not_abandon_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for directory in ("checkpoints", "diffs", "transactions"):
        (tmp_path / directory).mkdir()
    store = RunStore(tmp_path)
    state = {
        "elapsed_seconds": 0.0,
        "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    source = "def fit(train_rows, seed):\n    return 1\n\ndef predict(model, rows):\n    return [model] * len(rows)\n"
    repaired_source = source.replace("return 1", "return 2")
    experiment = GeneratedExperiment("Repair it", "Complete execution", source)
    repaired = GeneratedExperiment("Repair it", "Complete execution", repaired_source)
    plan = ResearchPlan(
        "Repair it",
        "Complete execution",
        "Exercise repeated repair requests.",
        "No prior work.",
        (),
        "Simple model",
        ("Repair then execute",),
        ("Invalid patch response",),
    )
    attempt = {
        "iteration": 1,
        "experiment_id": "iteration-001",
        "recovery": [],
    }

    class RepairClient:
        calls = 0

        def repair(self, *_args: object, **_kwargs: object):
            self.calls += 1
            if self.calls == 1:
                response = LLMResult("bad patch", {"total_tokens": 3})
                raise GenerationFailure("constant ranker labels", [response])
            return repaired, [LLMResult("good patch", {"total_tokens": 5})]

    executions = 0

    def run(*_args: object, **_kwargs: object):
        nonlocal executions
        executions += 1
        if executions == 1:
            from tippytop.runtime import ExperimentFailure

            raise ExperimentFailure("unsupported fit keywords")
        return {"valid_metrics": {"primary": 0.61}}, {"stdout": "ok", "stderr": ""}

    monkeypatch.setattr("tippytop.search.execution.run_experiment", run)
    client = RepairClient()

    outcome = execute_with_repairs(
        RunConfig(),
        store,
        state,
        client,  # type: ignore[arg-type]
        {},
        plan,
        experiment,
        "iteration-001",
        {experiment.source_hash},
        [],
        tmp_path / "unused.pkl",
        "revision",
        time.monotonic() + 30,
        time.monotonic(),
        attempt,
    )

    assert outcome.result == {"valid_metrics": {"primary": 0.61}}
    assert outcome.experiment.source == repaired_source
    assert client.calls == 2
    assert executions == 2
    actions = [item["action"] for item in outcome.recovery]
    assert "llm_code_repair_response_rejected" in actions
    assert "llm_code_repair_succeeded" in actions
    assert state["llm_usage"]["total_tokens"] == 8


def test_accepted_fourth_patch_gets_a_fresh_repair_budget_after_execution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for directory in ("checkpoints", "diffs", "transactions"):
        (tmp_path / directory).mkdir()
    store = RunStore(tmp_path)
    state = {
        "elapsed_seconds": 0.0,
        "llm_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    source = "def fit(train_rows, seed):\n    return 1\n\ndef predict(model, rows):\n    return [model] * len(rows)\n"
    first_repair_source = source.replace("return 1", "return 2")
    second_repair_source = source.replace("return 1", "return 3")
    experiment = GeneratedExperiment("Repair it", "Complete execution", source)
    first_repair = GeneratedExperiment("Repair it", "Complete execution", first_repair_source)
    second_repair = GeneratedExperiment("Repair it", "Complete execution", second_repair_source)
    plan = ResearchPlan(
        "Repair it",
        "Complete execution",
        "Exercise independent response and execution repair budgets.",
        "No prior work.",
        (),
        "Simple model",
        ("Keep repairing the same experiment",),
        ("Invalid patch response", "New runtime failure after an accepted patch"),
    )
    attempt = {"iteration": 1, "experiment_id": "iteration-001", "recovery": []}

    class RepairClient:
        calls = 0

        def repair(self, *_args: object, **_kwargs: object):
            self.calls += 1
            if self.calls <= 3:
                response = LLMResult(f"bad patch {self.calls}", {"total_tokens": 1})
                raise GenerationFailure("invalid patch", [response])
            if self.calls == 4:
                return first_repair, [LLMResult("first accepted patch", {"total_tokens": 2})]
            return second_repair, [LLMResult("second accepted patch", {"total_tokens": 3})]

    executions = 0

    def run(*_args: object, **_kwargs: object):
        nonlocal executions
        executions += 1
        if executions <= 2:
            from tippytop.runtime import ExperimentFailure

            raise ExperimentFailure(f"runtime failure {executions}")
        return {"valid_metrics": {"primary": 0.61}}, {"stdout": "ok", "stderr": ""}

    monkeypatch.setattr("tippytop.search.execution.run_experiment", run)
    client = RepairClient()

    outcome = execute_with_repairs(
        RunConfig(),
        store,
        state,
        client,  # type: ignore[arg-type]
        {},
        plan,
        experiment,
        "iteration-001",
        {experiment.source_hash},
        [],
        tmp_path / "unused.pkl",
        "revision",
        time.monotonic() + 30,
        time.monotonic(),
        attempt,
    )

    assert outcome.result == {"valid_metrics": {"primary": 0.61}}
    assert outcome.experiment.source == second_repair_source
    assert client.calls == 5
    assert executions == 3
    assert outcome.experiment_id == "iteration-001-repair-2"
    assert state["llm_usage"]["total_tokens"] == 8
