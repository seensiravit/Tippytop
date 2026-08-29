from __future__ import annotations

import json

from tippytop.artifacts import RunStore
from tippytop.research.context import (
    RECENT_CODE_LIMIT,
    RECENT_OUTCOME_LIMIT,
    SOURCE_CONTEXT_CHARS,
    _outcome_summaries,
    build_research_context,
    snapshot_prior_research,
)


def test_context_includes_bounded_recent_source_and_diagnostics(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.write_json("dataset_summary.json", {"train": {"rows": 20}, "valid": {"rows": 4}})
    history = []
    for iteration in range(1, 11):
        source = f"# iteration {iteration}\n" + ("x" * 20_000)
        record = {
            "iteration": iteration,
            "status": "failed",
            "hypothesis": f"hypothesis {iteration}",
            "error": "runtime failure",
            "test_metrics": {"primary": 1.0},
        }
        history.append(record)
        store.write_json(
            f"experiments/{iteration:03d}.json",
            {
                "status": "failed",
                "source_hash": str(iteration),
                "executed_experiment": {
                    "hypothesis": record["hypothesis"],
                    "expected_effect": "improve validation ranking",
                    "source": source,
                },
                "recovery": [
                    {
                        "action": "request_llm_code_repair",
                        "error": "traceback " + ("e" * 10_000),
                        "diagnostics": {
                            "unexpected_keyword": "objective",
                            "required_action": "move estimator options into the constructor",
                        },
                    }
                ],
                "test_secret": "MUST_NOT_LEAK",
            },
        )

    state = {
        "iteration": 10,
        "config": {
            "epsilon": 0.002,
            "max_iterations": 20,
            "experiment_timeout": 1800,
        },
        "baseline_valid": {"primary": 0.60},
        "best": {"metrics": {"primary": 0.61}},
    }

    context = build_research_context(store, state, history, "# current best")

    assert len(context["recent_experiments"]) == RECENT_OUTCOME_LIMIT
    assert len(context["recent_code_attempts"]) == RECENT_CODE_LIMIT
    assert context["recent_code_attempts"][-1]["source"].startswith("# iteration 10")
    assert len(context["recent_code_attempts"][-1]["source"]) == SOURCE_CONTEXT_CHARS
    rendered = json.dumps(context)
    assert "request_llm_code_repair" in rendered
    assert "unexpected_keyword" in rendered
    assert "move estimator options into the constructor" in rendered
    assert "MUST_NOT_LEAK" not in rendered
    assert "test_metrics" not in rendered


def test_prior_research_snapshot_rejects_test_evaluated_runs(tmp_path) -> None:
    current = RunStore(tmp_path / "current")
    prior = RunStore(tmp_path / "prior")
    prior.write_json("state.json", {"test_evaluated": True})

    try:
        snapshot_prior_research(current, prior.path)
    except ValueError as error:
        assert "must not have evaluated test" in str(error)
    else:
        raise AssertionError("test-evaluated prior run was accepted")


def test_context_recovers_source_from_rejected_generation(tmp_path) -> None:
    store = RunStore(tmp_path)
    store.write_json("dataset_summary.json", {"train": {}, "valid": {}})
    source = "def predict(model, rows):\n    return []\n"
    history = [
        {
            "iteration": 1,
            "status": "generation_failed",
            "error": "missing fit",
            "responses": [
                {
                    "content": json.dumps(
                        {
                            "hypothesis": "broken module",
                            "expected_effect": "none",
                            "source": source,
                        }
                    )
                }
            ],
        }
    ]
    state = {
        "iteration": 1,
        "config": {"epsilon": 0.002, "max_iterations": 2, "experiment_timeout": 30},
        "baseline_valid": {"primary": 0.60},
        "best": {"metrics": {"primary": 0.60}},
    }

    context = build_research_context(store, state, history, "")

    assert context["recent_code_attempts"][0]["source"] == source
    assert context["recent_code_attempts"][0]["error"] == "missing fit"


def test_outcome_summary_explains_exact_metric_repeat() -> None:
    summaries = _outcome_summaries(
        [
            {
                "iteration": 2,
                "diagnostics": {
                    "matching_prior_iteration": 1,
                    "primary_delta_from_baseline": -0.01,
                },
            }
        ]
    )

    diagnostics = summaries[0]["diagnostics"]
    assert diagnostics is not None
    assert diagnostics["outcome_classification"] == "exact_repeat_likely_no_op"
