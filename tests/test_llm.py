from __future__ import annotations

import json

import pytest

from tippytop.config import RunConfig
from tippytop.generated import GeneratedExperiment
from tippytop.llm import GenerationFailure, LLMClient, LLMResult, LLMTransportFailure


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
    experiment, responses = client.generate({"best": 0.6})
    assert experiment.source == SOURCE
    assert len(responses) == 2
    assert all(call[1]["attempts"] == 1 for call in client.calls)


def test_transport_failure_is_distinct_from_invalid_generation() -> None:
    with pytest.raises(LLMTransportFailure, match="temporary timeout"):
        FailingClient().generate({"best": 0.6})


def test_rejected_correction_preserves_raw_responses() -> None:
    client = FakeClient(["not json", "still not json", "also not json"])
    with pytest.raises(GenerationFailure) as captured:
        client.generate({"best": 0.6})

    assert [response.content for response in captured.value.responses] == [
        "not json",
        "still not json",
        "also not json",
    ]


def test_runtime_repair_returns_new_source() -> None:
    repaired = {
        "hypothesis": "Repair a failed prior model.",
        "expected_effect": "Run without the reported exception.",
        "source": REPAIRED_SOURCE,
    }
    client = FakeClient([json.dumps(repaired)])
    experiment, responses = client.repair(
        {"best": 0.6},
        GeneratedExperiment.from_dict({**repaired, "source": SOURCE}),
        "ValueError: failed",
    )
    assert experiment.hypothesis == repaired["hypothesis"]
    assert len(responses) == 1


def test_rejected_runtime_repair_preserves_raw_response() -> None:
    failed = {
        "hypothesis": "Repair a failed prior model.",
        "expected_effect": "Run without the reported exception.",
        "source": SOURCE,
    }
    client = FakeClient(["not json", "still not json"])
    with pytest.raises(GenerationFailure) as captured:
        client.repair(
            {"best": 0.6},
            GeneratedExperiment.from_dict(failed),
            "ValueError: failed",
        )

    assert [response.content for response in captured.value.responses] == [
        "not json",
        "still not json",
    ]


def test_comment_only_runtime_repair_is_corrected() -> None:
    failed = {
        "hypothesis": "Repair a failed prior model.",
        "expected_effect": "Run without the reported exception.",
        "source": SOURCE,
    }
    comment_only = {**failed, "source": SOURCE + "\n# claimed repair\n"}
    executable_repair = {**failed, "source": REPAIRED_SOURCE}
    client = FakeClient([json.dumps(comment_only), json.dumps(executable_repair)])

    experiment, responses = client.repair(
        {"best": 0.6},
        GeneratedExperiment.from_dict(failed),
        "ValueError: failed",
    )

    assert experiment.source == REPAIRED_SOURCE
    assert len(responses) == 2


def test_pre_execution_review_rewrites_blocking_source() -> None:
    revised = {
        "verdict": "revise",
        "critique": "The original returns calibrated constants rather than useful ranking scores.",
        "hypothesis": "Produce a nonconstant score using the available rows.",
        "expected_effect": "Improve within-user ordering.",
        "source": REPAIRED_SOURCE,
    }
    client = FakeClient([json.dumps(revised)])

    review, responses = client.review(
        {"baseline_validation": {"primary": 0.6}},
        GeneratedExperiment.from_dict(
            {
                "hypothesis": "Use a constant prior.",
                "expected_effect": "Establish a control.",
                "source": SOURCE,
            }
        ),
    )

    assert review.verdict == "revise"
    assert review.experiment.source == REPAIRED_SOURCE
    assert len(responses) == 1


def test_pre_execution_review_rejects_comment_only_revision() -> None:
    proposed = GeneratedExperiment.from_dict(
        {
            "hypothesis": "Use a constant prior.",
            "expected_effect": "Establish a control.",
            "source": SOURCE,
        }
    )
    comment_only = {
        "verdict": "revise",
        "critique": "Claimed a correction.",
        "hypothesis": proposed.hypothesis,
        "expected_effect": proposed.expected_effect,
        "source": SOURCE + "\n# reviewed\n",
    }
    corrected = {**comment_only, "source": REPAIRED_SOURCE}
    client = FakeClient([json.dumps(comment_only), json.dumps(corrected)])

    review, responses = client.review({}, proposed)

    assert review.experiment.source == REPAIRED_SOURCE
    assert len(responses) == 2
