"""The two things the deleted agent lane took with it, now wired to the live one.

Both are graded, and both were previously untested *in the lane that ships*:

1. **The validation wall.** `context.build_context` feeds the last failure's
   `error` string back into the proposal prompt. That string is the tail of a
   crashed run's stdout, and `baseline.py` prints a summary block containing the
   *test* primary. So a run that crashes after printing its summary is the one
   path by which a test metric can reach the model that decides what to try
   next. Everything else the model sees is validation-only by construction.

2. **The manual-intervention count.** Deliverable 3 asks for it and Impact &
   Relevance (20%) is scored on it. It has to be derived from a durable log, not
   an in-memory counter, or a resumed run reports a flattering zero.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tippytop.runlog import InterventionLog, contains_test_signal, scrub  # noqa: E402
from autoresearch_lg import experiment as exp_mod  # noqa: E402
from autoresearch_lg import graph as graph_mod  # noqa: E402


# A realistic crash tail: baseline.py printed its summary, then died.
CRASH_TAIL = """\
valid GAUC 0.6674 | nDCG@5 0.5363 | primary 0.6019
test  GAUC 0.6591 | nDCG@5 0.5323 | primary 0.5957
Traceback (most recent call last):
  File "baseline.py", line 210, in <module>
    main()
ValueError: operands could not be broadcast together
"""


class _FakeResult(dict):
    pass


def _fake_run_baseline(stdout, *, crashed=True, timed_out=False):
    def _run(cwd, data_dir, seed=0):
        return {"stdout": stdout, "wall_seconds": 1.0,
                "crashed": crashed, "timed_out": timed_out}
    return _run


def _state():
    return {"step_failed": False, "retry_count": 0, "exp_dir": "/tmp/x",
            "data_dir": "/tmp/d"}


# --- 1. the wall ----------------------------------------------------------

def test_crash_tail_would_leak_the_test_metric_unscrubbed():
    """Establishes the hazard is real before asserting it is handled."""
    assert contains_test_signal(CRASH_TAIL)


def test_run_and_evaluate_scrubs_the_test_metric_out_of_the_error(monkeypatch):
    monkeypatch.setattr(exp_mod.tools, "run_baseline", _fake_run_baseline(CRASH_TAIL))
    out = exp_mod.run_and_evaluate(_state())
    assert out["step_failed"] is True
    assert not contains_test_signal(out["failure_error"]), out["failure_error"]
    # The diagnostic value survives — the model still learns what broke.
    assert "ValueError" in out["failure_error"]


def test_scrub_keeps_the_validation_line():
    """Over-scrubbing would be its own failure: valid metrics are the signal."""
    cleaned = scrub(CRASH_TAIL)
    assert "valid GAUC 0.6674" in cleaned


def test_timeout_path_is_scrubbed_too(monkeypatch):
    monkeypatch.setattr(exp_mod.tools, "run_baseline",
                        _fake_run_baseline(CRASH_TAIL, timed_out=True))
    out = exp_mod.run_and_evaluate(_state())
    assert "timed out" in out["failure_error"]
    assert not contains_test_signal(out["failure_error"])


def test_clean_run_is_untouched(monkeypatch):
    monkeypatch.setattr(exp_mod.tools, "run_baseline",
                        _fake_run_baseline(CRASH_TAIL, crashed=False))
    out = exp_mod.run_and_evaluate(_state())
    assert out["step_failed"] is False
    # run_stdout is state, not prompt text: collect_metrics needs the test
    # number to LOG it. Reporting test is required; feeding it back is not.
    assert "test" in out["run_stdout"]


# --- 2. the intervention count -------------------------------------------

def test_resume_is_detected_without_the_operator_declaring_it(tmp_path):
    (tmp_path / "runs.jsonl").write_text('{"iteration": 1}\n{"iteration": 2}\n')
    log = InterventionLog(tmp_path)
    assert log.count == 0
    rec = log.detect_resume(tmp_path / "runs.jsonl")
    assert rec is not None and rec.kind == "resume"
    assert log.count == 1


def test_a_fresh_run_records_nothing(tmp_path):
    log = InterventionLog(tmp_path)
    assert log.detect_resume(tmp_path / "runs.jsonl") is None
    assert log.count == 0
    assert log.summary().startswith("0 ")


def test_count_survives_a_restart_of_the_process(tmp_path):
    """The bug this replaced: a new object reset the count to zero."""
    InterventionLog(tmp_path).record("manual_note", "installed lightgbm by hand")
    assert InterventionLog(tmp_path).count == 1


def test_finalize_reports_the_count_into_resource_report(tmp_path, monkeypatch):
    log = InterventionLog(tmp_path)
    log.record("manual_note", "restarted after a dependency install")

    monkeypatch.setattr(graph_mod.tools, "make_submission",
                        lambda *a, **k: (True, "ok"))
    state = {
        "repo_root": str(tmp_path), "data_dir": str(tmp_path), "start_time": 0.0,
        "best_valid_primary": 0.60, "best_test_primary": 0.59,
        "best_checkpoint_id": 3, "best_exp_dir": "", "iteration": 7,
        "history": [{"tokens_in": 10, "tokens_out": 5}], "concepts": [],
    }
    out = graph_mod.finalize(state)
    report = out["resource_report"]
    assert report["manual_interventions"] == 1
    assert "manual_note" in report["intervention_summary"]
    assert report["interventions"][0]["reason"].startswith("restarted")

    on_disk = json.loads((tmp_path / "resource_report.json").read_text())
    assert on_disk["manual_interventions"] == 1


def test_run_artifacts_archive_the_intervention_log(tmp_path):
    """A fresh `setup` must not inherit the previous run's intervention count."""
    from autoresearch_lg import tools
    assert "interventions.jsonl" in tools.RUN_ARTIFACTS


# --- 3. the artifact packager --------------------------------------------
#
# The packager is the last gate before submission, so its job is to REFUSE.
# These tests are about the refusals, not the copy.

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "package_final_run",
    Path(__file__).resolve().parents[1] / "scripts" / "package_final_run.py")
pack = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pack)


def _good_run(tmp_path, *, iterations=8, interventions=1, sub_ok=True):
    # Shaped like a real record, because the packager now checks Deliverable 3's
    # required fields (hypothesis + the code diff applied) and Deliverable 4's
    # per-metric scores, not just the row count.
    (tmp_path / "runs.jsonl").write_text("".join(
        json.dumps({
            "iteration": i, "outcome": "improved" if i % 3 else "error",
            "mode": "tune", "hypothesis": "h", "diff": "--- a/baseline.py\n+k = 32\n",
            "metrics": {"valid_primary": 0.60, "test_primary": 0.59,
                        "valid": {"GAUC": 0.667, "nDCG@5": 0.536, "primary": 0.60},
                        "test": {"GAUC": 0.661, "nDCG@5": 0.528, "primary": 0.59}},
        }) + "\n" for i in range(iterations)))
    (tmp_path / "results.tsv").write_text("commit\tvalid_primary\n")
    (tmp_path / "concepts.json").write_text("[]")
    (tmp_path / "submission.csv").write_text("row_id,user_id,video_id,score\n0,u,v,0.5\n")
    (tmp_path / "interventions.jsonl").write_text("".join(
        json.dumps({"kind": "manual_note", "reason": "r", "iso": "x", "ts": 0.0}) + "\n"
        for _ in range(interventions)))
    (tmp_path / "resource_report.json").write_text(json.dumps({
        "iterations": iterations, "elapsed_seconds": 4200.0,
        "tokens_in_total": 120000, "tokens_out_total": 30000,
        "manual_interventions": interventions, "intervention_summary": "1 (manual_note x1)",
        "score_dataset": 0.0032, "test_delta_vs_baseline": {"GAUC": 0.004, "nDCG@5": 0.0024},
    }))
    # Stand-in for the frozen kit's checker: the packager only cares about its
    # exit code, and the real one needs the 46 MB dataset.
    (tmp_path / "submit.py").write_text(
        f"import sys; sys.exit({0 if sub_ok else 1})\n")
    return tmp_path


def test_packager_accepts_a_complete_run(tmp_path):
    res = pack.check(_good_run(tmp_path), strict=True)
    assert res["problems"] == []
    assert any("8 iterations" in n for n in res["notes"])
    assert any("manual interventions: 1" in n for n in res["notes"])


def test_packager_refuses_a_missing_artifact(tmp_path):
    root = _good_run(tmp_path)
    (root / "concepts.json").unlink()
    with pytest.raises(pack.Problem, match="concepts.json"):
        pack.check(root, strict=True)


def test_packager_refuses_a_rejected_submission(tmp_path):
    """The failure mode that costs everything: a CSV that scores zero on format."""
    root = _good_run(tmp_path, sub_ok=False)
    with pytest.raises(pack.Problem, match="REJECTED"):
        pack.check(root, strict=True)


def test_packager_refuses_a_smoke_test(tmp_path):
    with pytest.raises(pack.Problem, match="smoke test"):
        pack.check(_good_run(tmp_path, iterations=2), strict=True)


def test_packager_catches_an_intervention_count_that_disagrees_with_its_log(tmp_path):
    """An unsupported zero is worth nothing; a wrong number is worse."""
    root = _good_run(tmp_path, interventions=1)
    rep = json.loads((root / "resource_report.json").read_text())
    rep["manual_interventions"] = 0
    (root / "resource_report.json").write_text(json.dumps(rep))
    with pytest.raises(pack.Problem, match="do not submit either number"):
        pack.check(root, strict=True)


def test_packager_refuses_a_report_predating_the_intervention_wiring(tmp_path):
    root = _good_run(tmp_path)
    rep = json.loads((root / "resource_report.json").read_text())
    del rep["manual_interventions"]
    (root / "resource_report.json").write_text(json.dumps(rep))
    with pytest.raises(pack.Problem, match="Deliverable 3"):
        pack.check(root, strict=True)


def test_packager_refuses_a_run_log_with_no_code_diff(tmp_path):
    """Deliverable 3 says "the code diff applied", verbatim. Filenames are not a
    diff, and runs/ is gitignored — so a log without this field is unmarkable."""
    root = _good_run(tmp_path)
    lines = [json.loads(x) for x in (root / "runs.jsonl").read_text().splitlines()]
    for ln in lines:
        ln["diff"] = ""
    (root / "runs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    with pytest.raises(pack.Problem, match="no 'diff' field"):
        pack.check(root, strict=True)


def test_packager_refuses_a_run_log_with_no_hypothesis(tmp_path):
    root = _good_run(tmp_path)
    lines = [json.loads(x) for x in (root / "runs.jsonl").read_text().splitlines()]
    for ln in lines:
        ln["hypothesis"] = ""
    (root / "runs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    with pytest.raises(pack.Problem, match="hypothesis"):
        pack.check(root, strict=True)


def test_packager_reports_the_per_metric_deltas(tmp_path):
    res = pack.check(_good_run(tmp_path), strict=True)
    assert any("score_dataset" in n for n in res["notes"])
