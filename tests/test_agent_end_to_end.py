"""Drive the whole agent loop with a MockLLM — zero network, tiny data.

Exercises: seed -> improve -> a broken solution (runtime error) -> debug/recovery
-> finalize on test. Asserts machinery + the test-set wall.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _mini_data import make_mini_data                      # noqa: E402

from tippytop.agent import AgentConfig, run_agent          # noqa: E402
from tippytop.agent.llm.mock import MockLLMClient          # noqa: E402
from tippytop.agent import orchestrator                    # noqa: E402
from tippytop.submission import read_submission            # noqa: E402
from tippytop.data.dataset import load_dataset             # noqa: E402

FENCE = "```"

_GOOD = f'''Hypothesis: same FM, different seed.
{FENCE}python
import argparse
from tippytop.data.dataset import load_dataset
from tippytop.training.runner import train_model
from tippytop.submission import write_submission
a = argparse.ArgumentParser()
a.add_argument("--data_dir"); a.add_argument("--split", default="valid"); a.add_argument("--out")
args = a.parse_args()
data = load_dataset(args.data_dir)
model = train_model("fm", data, seed=1)
write_submission(args.out, data.splits[args.split], model.predict(data, args.split))
{FENCE}
'''

_BROKEN = f'''Hypothesis: (buggy) reference an undefined helper.
{FENCE}python
import argparse
a = argparse.ArgumentParser()
a.add_argument("--data_dir"); a.add_argument("--split"); a.add_argument("--out")
args = a.parse_args()
scores = make_the_scores(args)   # NameError at runtime
{FENCE}
'''


def _scripts(idx, system, user):
    # LLM call 0 -> improved; call 1 -> broken; call 2+ -> fixed (in a DEBUG prompt)
    return [_GOOD, _BROKEN, _GOOD][idx] if idx < 3 else _GOOD


def test_full_loop_with_recovery_and_walling(tmp_path, monkeypatch):
    data_dir = make_mini_data(tmp_path / "data")

    # spy on which split every solution execution uses
    seen_splits = []
    real_run = orchestrator.run_solution

    def spy_run(code, *, iter_dir, data_dir, split, timeout_s, python_exe):
        seen_splits.append(split)
        return real_run(code, iter_dir=iter_dir, data_dir=data_dir, split=split,
                        timeout_s=timeout_s, python_exe=python_exe)

    monkeypatch.setattr(orchestrator, "run_solution", spy_run)

    cfg = AgentConfig(
        data_dir=str(data_dir), run_dir=tmp_path / "run", run_id="t",
        max_iters=5, conv_n=10,          # conv_n high so all mock scripts run
        iter_timeout_s=120, final_out=tmp_path / "final.csv",
        python_exe=sys.executable)
    llm = MockLLMClient(_scripts)

    state = run_agent(cfg, llm, verbose=False)

    # loop completed and found a working best
    assert state.best_code is not None and state.best_metrics is not None
    assert state.stop_reason == "max_iters"

    # per-iteration artifacts
    journal = (tmp_path / "run" / "journal.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(journal) == 5
    assert (tmp_path / "run" / "report.md").exists()

    # recovery happened: at least one runtime failure, and a later success
    import json
    recs = [json.loads(x) for x in journal]
    assert any(r["error_kind"] == "runtime" for r in recs)
    assert recs[0]["phase"] == "SEED"
    assert any(r["phase"] == "DEBUG" for r in recs)     # debug prompt was used

    # fully autonomous
    assert state.interventions == 0

    # final submission written and valid against the TEST split
    assert state.final_out and Path(state.final_out).exists()
    test_rows = load_dataset(str(data_dir)).splits["test"]
    read_submission(state.final_out, test_rows)          # raises if misaligned

    # ---- TEST-SET WALL ----
    # every in-loop execution used valid; test touched exactly once, at the end
    assert seen_splits[-1] == "test"
    assert all(s == "valid" for s in seen_splits[:-1])
    assert seen_splits.count("test") == 1
    # no test metric ever appeared in any prompt the LLM saw
    for call in llm.calls:
        blob = call["system"] + call["user"]
        assert "final test" not in blob.lower()
        assert "splits['test']" not in blob and 'splits["test"]' not in blob


@pytest.mark.parametrize("bad_reply", ["no code here at all", ""])
def test_malformed_reply_recovers(tmp_path, bad_reply):
    """A reply with no code fence -> parse error -> reprompt, loop still finishes."""
    data_dir = make_mini_data(tmp_path / "data")
    cfg = AgentConfig(data_dir=str(data_dir), run_dir=tmp_path / "run", run_id="t2",
                      max_iters=2, conv_n=10, iter_timeout_s=120,
                      final_out=tmp_path / "final.csv", python_exe=sys.executable)
    # iter0 is the seed (works); iter1's LLM reply is malformed both attempts
    llm = MockLLMClient([bad_reply])
    state = run_agent(cfg, llm, verbose=False)
    assert state.best_code is not None          # seed still gives a valid best
    import json
    recs = [json.loads(x) for x in
            (tmp_path / "run" / "journal.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert any(r["error_kind"] == "parse" for r in recs)
