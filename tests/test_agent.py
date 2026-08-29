from dataclasses import replace

from tippytop.agent import _new_state, _open_run
from tippytop.artifacts import RunStore, read_json, read_jsonl
from tippytop.config import RunConfig


def test_resume_persists_active_runtime_overrides(tmp_path) -> None:
    initial = RunConfig(runs_dir=tmp_path, max_iterations=5, llm_timeout=120)
    store = RunStore.create(initial)
    store.write_json("state.json", _new_state(initial))
    updated = replace(initial, max_iterations=10, max_hours=4, llm_timeout=300)

    _, state = _open_run(updated, store.path)

    assert state["config"]["max_iterations"] == 10
    assert state["config"]["max_hours"] == 4
    assert state["config"]["llm_timeout"] == 300
    assert read_json(store.path / "config.json") == state["config"]
    assert any(
        event["event"] == "run_configuration_updated"
        for event in read_jsonl(store.path / "events.jsonl")
    )


def test_new_run_snapshots_validation_only_research_memory(tmp_path) -> None:
    prior = RunStore(tmp_path / "prior")
    prior.write_json(
        "state.json",
        {"test_evaluated": False, "used_source_hashes": ["old-source"]},
    )
    prior.append(
        "iterations.jsonl",
        {"iteration": 1, "status": "generation_failed", "error": "old failure"},
    )
    config = RunConfig(runs_dir=tmp_path / "runs", prior_run=prior.path)

    store, state = _open_run(config, None)

    assert state["research_parent"] == str(prior.path.resolve())
    assert state["used_source_hashes"] == ["old-source"]
    assert read_json(store.path / "prior_research.json")["source_run"] == str(
        prior.path.resolve()
    )
