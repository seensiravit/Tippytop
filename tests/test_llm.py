from __future__ import annotations

import json

import pytest

from tippytop.config import RunConfig
from tippytop.generated import GeneratedExperiment
from tippytop.llm import GenerationFailure, LLMClient, LLMResult, LLMTransportFailure
from tippytop.llm.prompts import repair_messages
from tippytop.research import ResearchPlan


SOURCE = """import numpy as np

def fit(train_rows, seed):
    return float(np.mean([row[6] for row in train_rows]))

def predict(model, rows):
    return np.full(len(rows), model, dtype=np.float32)
"""

REPAIRED_SOURCE = SOURCE.replace(
    "return np.full(len(rows), model, dtype=np.float32)",
    "return np.full(len(rows), float(model), dtype=np.float32)",
)

PLAN_PAYLOAD = {
    "hypothesis": "Use a substantive learned ranking signal.",
    "expected_effect": "Improve within-user ordering.",
    "rationale": "The representation and objective should match ranking errors.",
    "departure_from_prior_work": "Change both the representation and supervision.",
    "data_and_features": ["Build causal training aggregates and replayable prediction features."],
    "model_and_objective": "Fit a seeded continuous-score ranking model.",
    "implementation_outline": ["Fit state on train.", "Return state and score prediction rows."],
    "failure_modes": ["Prevent leakage and class-label outputs."],
}
PLAN = ResearchPlan.from_dict(PLAN_PAYLOAD)


class FakeClient(LLMClient):
    def __init__(self, responses: list[str]):
        super().__init__(RunConfig())
        self.responses = iter(responses)
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def complete(self, *args: object, **kwargs: object) -> LLMResult:
        self.calls.append((args, kwargs))
        return LLMResult(next(self.responses), {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})


class FailingClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(RunConfig())

    def complete(self, *args: object, **kwargs: object) -> LLMResult:
        raise ConnectionError("temporary timeout")


def test_invalid_generated_experiment_is_corrected() -> None:
    valid = {
        "hypothesis": "Use the training prior as a smoke test.",
        "expected_effect": "Produce finite scores.",
        "source": SOURCE,
    }
    client = FakeClient(["not json", json.dumps(valid)])
    experiment, responses = client.generate({"best": 0.6}, PLAN)
    assert experiment.source == SOURCE
    assert len(responses) == 2
    assert all(call[1]["attempts"] == 1 for call in client.calls)


def test_transport_failure_is_distinct_from_invalid_generation() -> None:
    with pytest.raises(LLMTransportFailure, match="temporary timeout"):
        FailingClient().generate({"best": 0.6}, PLAN)


def test_rejected_correction_preserves_raw_responses() -> None:
    client = FakeClient(["not json", "still not json", "also not json"])
    with pytest.raises(GenerationFailure) as captured:
        client.generate({"best": 0.6}, PLAN)

    assert [response.content for response in captured.value.responses] == [
        "not json",
        "still not json",
        "also not json",
    ]


def test_runtime_repair_returns_new_source() -> None:
    failed = {
        "hypothesis": "Repair a failed prior model.",
        "expected_effect": "Run without the reported exception.",
        "source": SOURCE,
    }
    patch = {
        "edits": [
            {
                "old": "return np.full(len(rows), model, dtype=np.float32)",
                "new": "return np.full(len(rows), float(model), dtype=np.float32)",
            }
        ]
    }
    client = FakeClient([json.dumps(patch)])
    experiment, responses = client.repair(
        {"best": 0.6},
        GeneratedExperiment.from_dict(failed),
        "ValueError: failed",
        plan=PLAN,
    )
    assert experiment.source == REPAIRED_SOURCE
    assert len(responses) == 1


def test_rejected_runtime_repair_preserves_raw_response() -> None:
    failed = {
        "hypothesis": "Repair a failed prior model.",
        "expected_effect": "Run without the reported exception.",
        "source": SOURCE,
    }
    client = FakeClient(["not json", "still not json", "also not json"])
    with pytest.raises(GenerationFailure) as captured:
        client.repair(
            {"best": 0.6},
            GeneratedExperiment.from_dict(failed),
            "ValueError: failed",
            plan=PLAN,
        )

    assert [response.content for response in captured.value.responses] == [
        "not json",
        "still not json",
        "also not json",
    ]


def test_comment_only_runtime_repair_is_corrected() -> None:
    failed = {
        "hypothesis": "Repair a failed prior model.",
        "expected_effect": "Run without the reported exception.",
        "source": SOURCE,
    }
    comment_only = {"edits": [{"old": SOURCE, "new": SOURCE + "\n# claimed repair\n"}]}
    executable_repair = {
        "edits": [
            {
                "old": "return np.full(len(rows), model, dtype=np.float32)",
                "new": "return np.full(len(rows), float(model), dtype=np.float32)",
            }
        ]
    }
    client = FakeClient([json.dumps(comment_only), json.dumps(executable_repair)])

    experiment, responses = client.repair(
        {"best": 0.6},
        GeneratedExperiment.from_dict(failed),
        "ValueError: failed",
        plan=PLAN,
    )

    assert experiment.source == REPAIRED_SOURCE
    assert len(responses) == 2


def test_research_plan_schema_is_corrected_before_coding() -> None:
    client = FakeClient(["{}", json.dumps(PLAN_PAYLOAD)])

    plan, responses = client.research({"recent_experiments": []})

    assert plan == PLAN
    assert len(responses) == 2


def test_research_plan_accepts_descriptive_extras_and_scalar_feature_text() -> None:
    payload = {
        **PLAN_PAYLOAD,
        "experiment_title": "A descriptive title",
        "data_and_features": "Use causal user histories and prediction-time context.",
        "model_and_objective": {"model": "ranker", "objective": "pairwise"},
    }

    plan = ResearchPlan.from_dict(payload)

    assert plan.data_and_features == (
        "Use causal user histories and prediction-time context.",
    )
    assert plan.model_and_objective == '{"model": "ranker", "objective": "pairwise"}'


@pytest.mark.parametrize("wrapper", ["research_plan", "experiment_plan", "plan"])
def test_research_plan_accepts_common_response_envelopes(wrapper: str) -> None:
    assert ResearchPlan.from_dict({wrapper: PLAN_PAYLOAD}) == PLAN


def test_research_plan_round_trips_without_optional_feature_notes() -> None:
    payload = {**PLAN_PAYLOAD, "data_and_features": []}
    plan = ResearchPlan.from_dict(payload)

    assert plan.data_and_features == ()
    assert ResearchPlan.from_dict(plan.to_dict()) == plan


def test_runtime_repair_receives_cumulative_failures() -> None:
    failed = {
        "hypothesis": "Repair all invalid fit arguments.",
        "expected_effect": "Complete smoke execution.",
        "source": SOURCE,
    }
    repaired = {
        "edits": [
            {
                "old": "return np.full(len(rows), model, dtype=np.float32)",
                "new": "return np.full(len(rows), float(model), dtype=np.float32)",
            }
        ]
    }
    history = [
        {
            "source_hash": "first",
            "error": "unexpected keyword objective",
            "diagnostics": {
                "unexpected_keyword": "objective",
                "installed_signatures": {"fit": "fit(X, y)"},
            },
        },
        {
            "source_hash": "second",
            "error": "unexpected keyword metric",
            "diagnostics": {"unexpected_keyword": "metric"},
        },
    ]
    client = FakeClient([json.dumps(repaired)])

    client.repair(
        {},
        GeneratedExperiment.from_dict(failed),
        history[-1]["error"],
        plan=PLAN,
        failure_history=history,
    )

    messages = client.calls[0][0][0]
    payload = json.loads(messages[1]["content"])
    failures = payload["cumulative_runtime_failures"]
    assert [item["source_hash"] for item in failures] == ["first", "second"]
    assert failures[-1]["diagnostics"] == history[-1]["diagnostics"]
    assert failures[0]["diagnostics"]["installed_signatures"] == {"fit": "fit(X, y)"}
    assert "source" not in payload["failed_experiment"]
    assert payload["failed_source_excerpts"]


def test_stateful_repair_excerpts_include_fit_and_predict() -> None:
    source = """class StatefulModel:
    def __init__(self):
        self.mapping = None

    def fit(self, train_rows):
        self.mapping = {"known": 1.0}
        return self

    def predict(self, rows):
        return rows["missing"].map(self.mapping)

def fit(train_rows, seed):
    return StatefulModel().fit(train_rows)

def predict(model, rows):
    return model.predict(rows)
"""
    experiment = GeneratedExperiment("Persist state", "Complete replay", source)
    messages = repair_messages(
        {},
        experiment,
        'File "experiment.py", line 10, in predict\nKeyError: missing',
        PLAN,
    )

    payload = json.loads(messages[1]["content"])
    excerpts = "\n".join(item["source"] for item in payload["failed_source_excerpts"])
    assert "self.mapping = {\"known\": 1.0}" in excerpts
    assert "return model.predict(rows)" in excerpts
