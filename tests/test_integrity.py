"""Adversarial tests for the boundaries the agent can actually cross.

The loop's editable surface is small — the LLM writes `baseline.py` and
`data.py` into one folder, the harness runs them, and a regex reads a number
back out. Everything that could go badly wrong goes wrong at one of those three
seams, and each one is a place where a *plausible-looking* proposal produces a
result that is wrong rather than absent:

  1. path      — the model chooses the filenames it writes
  2. metric    — the harness believes whatever number the run prints
  3. incumbent — every later proposal is built from the current best folder

A failure at seam 1 or 2 does not look like a failure. It looks like a win.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch_lg import experiment as exp_mod  # noqa: E402
from autoresearch_lg import propose as propose_mod  # noqa: E402
from autoresearch_lg import resilience as res  # noqa: E402
from autoresearch_lg import tools  # noqa: E402
from autoresearch_lg.critic import classify_outcome  # noqa: E402


# =========================================================================
# Seam 1: the model chooses the paths it writes
# =========================================================================

@pytest.fixture
def exp_dir(tmp_path):
    root = tmp_path / "repo"
    (root / "runs").mkdir(parents=True)
    (root / "evaluate.py").write_text("REAL_EVALUATOR = True\n")
    (root / "baseline.py").write_text("REAL_BASELINE = True\n")
    return root


@pytest.mark.parametrize("path", [
    "../../baseline.py",            # escape to the frozen root kit
    "../exp_0002/baseline.py",      # reach into a sibling experiment
    "a/../../../baseline.py",       # escape via an innocent-looking prefix
    "/etc/passwd",                  # absolute
    "C:/Windows/System32/x.py",     # Windows absolute
    "..\\..\\baseline.py",          # Windows separators
])
def test_path_traversal_is_refused(exp_dir, path):
    """The LLM produces these strings. `Path(exp_dir, "../../baseline.py")`
    resolves outside the folder and overwrites the frozen root kit — which is
    the single thing the whole runs/ isolation design exists to prevent."""
    d = tools.make_experiment_dir(str(exp_dir), "exp_0001")
    with pytest.raises(tools.UnsafeExperimentPath):
        tools.write_experiment_files(d, {path: "OWNED = True\n"})
    assert (exp_dir / "baseline.py").read_text() == "REAL_BASELINE = True\n"


@pytest.mark.parametrize("path", ["evaluate.py", "submit.py", "baseline_scores.json"])
def test_the_scoring_spec_cannot_be_rewritten(exp_dir, path):
    """The highest-reward move available to anything optimising the number this
    harness reads is to replace the thing that produces the number.

    make_experiment_dir copies evaluate.py into the folder BEFORE
    write_experiment_files runs, so without this guard the model's version
    simply wins, the experiment is scored by its own evaluator, and runs.jsonl
    records a triumph.
    """
    d = tools.make_experiment_dir(str(exp_dir), "exp_0002")
    with pytest.raises(tools.UnsafeExperimentPath, match="protected"):
        tools.write_experiment_files(d, {path: "def evaluate(*a, **k): return {'primary': 1.0}\n"})
    assert "REAL_EVALUATOR" in (Path(d) / "evaluate.py").read_text()


def test_only_the_declared_editable_files_are_writable(exp_dir):
    d = tools.make_experiment_dir(str(exp_dir), "exp_0003")
    with pytest.raises(tools.UnsafeExperimentPath, match="not one of the editable"):
        tools.write_experiment_files(d, {"sneaky.py": "x = 1\n"},
                                     allowed=["baseline.py", "data.py"])


def test_a_legitimate_proposal_still_writes(exp_dir):
    d = tools.make_experiment_dir(str(exp_dir), "exp_0004")
    tools.write_experiment_files(d, {"baseline.py": "x = 1\n", "data.py": "y = 2\n"},
                                 allowed=["baseline.py", "data.py"])
    assert (Path(d) / "baseline.py").read_text() == "x = 1\n"


def test_one_bad_path_writes_nothing_at_all(exp_dir):
    """A half-written experiment is worse than none: it still runs, and it still
    gets scored."""
    d = tools.make_experiment_dir(str(exp_dir), "exp_0005")
    with pytest.raises(tools.UnsafeExperimentPath):
        tools.write_experiment_files(
            d, {"baseline.py": "good = 1\n", "../../data.py": "bad = 1\n"})
    assert not (Path(d) / "baseline.py").exists(), "partial write left behind"


def test_an_unsafe_path_becomes_a_repairable_failure_not_a_crash(exp_dir, monkeypatch):
    """It is a defective proposal, so it belongs on the failure branch with a
    reason the model can act on — not in a traceback that ends the run."""
    real_mkdir = tools.make_experiment_dir          # capture BEFORE patching
    monkeypatch.setattr(exp_mod.tools, "make_experiment_dir",
                        lambda r, n: real_mkdir(str(exp_dir), n))
    out = exp_mod.apply_diff({
        "iteration": 5, "repo_root": str(exp_dir),
        "edited_files": {"../../evaluate.py": "cheat = 1\n"},
        "editable_files": ["baseline.py", "data.py"], "best_exp_dir": "",
    })
    assert out["step_failed"] is True
    assert "refused to write" in out["failure_error"]
    assert res.classify_run_error(out["failure_error"]) == "deterministic"


# =========================================================================
# Seam 2: the harness believes the number the run prints
# =========================================================================

def _summary(valid, test="0.5957"):
    return (f"valid GAUC 0.6674 | nDCG@5 0.5363 | primary {valid}\n"
            f"test  GAUC 0.6591 | nDCG@5 0.5323 | primary {test}\n")


def test_a_real_summary_parses():
    assert tools.parse_summary(_summary("0.6019")) == (0.6019, 0.5957)


@pytest.mark.parametrize("bad", ["99.0", "1.5", "42"])
def test_an_impossible_score_is_refused(bad):
    """mean(GAUC, nDCG@5) is a mean of two quantities in [0, 1].

    Without the bound, code printing `primary 99.0` becomes the incumbent
    forever — best_valid_primary only moves up, every later proposal is built
    from that folder, and finalize ships it.
    """
    assert tools.parse_summary(_summary(bad)) is None


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "NaN"])
def test_non_numeric_scores_are_refused(bad):
    assert tools.parse_summary(_summary(bad)) is None


def test_scientific_notation_is_refused_rather_than_truncated():
    """`[\\d.]+` alone matches the '1' of '1e9' and reports a primary of 1.0 —
    a perfect score, silently, from a malformed line."""
    assert tools.parse_summary(_summary("1e9")) is None


def test_a_bad_metric_is_reported_differently_from_a_missing_one():
    """The model can only fix what it is told. These are different defects."""
    missing = exp_mod.collect_metrics({"step_failed": False, "run_stdout": "nothing here"})
    impossible = exp_mod.collect_metrics({"step_failed": False, "run_stdout": _summary("99.0")})
    assert "no summary block" in missing["failure_error"]
    assert "outside [0, 1]" in impossible["failure_error"]
    assert impossible["step_failed"] and missing["step_failed"]


# =========================================================================
# Seam 3: every later proposal is built from the incumbent's folder
# =========================================================================

def test_a_vanished_incumbent_falls_back_loudly_not_silently(tmp_path, monkeypatch):
    """best_exp_dir points at a folder that no longer exists (cleaned runs/, a
    full disk). Unguarded this surfaces as a bare OSError inside the node whose
    retry policy would back off five times and then declare the *provider*
    dead — the wrong diagnosis and the wrong response to a local, permanent
    problem."""
    root = tmp_path / "repo"
    root.mkdir()
    for f in ("baseline.py", "data.py"):
        (root / f).write_text(f"# pristine {f}\n")

    seen = {}

    def _capture(model, system_prompt, user_content):
        seen["prompt"] = user_content
        return ({"concept": "c", "hypothesis": "h", "description": "d",
                 "files": [{"path": "baseline.py", "content": "x = 1\n"}]}, 1, 1)

    monkeypatch.setattr(propose_mod, "_call_anthropic", _capture)
    out = propose_mod.llm_generate({
        "mode": "tune", "repo_root": str(root), "best_exp_dir": str(root / "runs" / "gone"),
        "editable_files": ["baseline.py", "data.py"], "eda_summary": {},
        "context_summary": "", "retrieved_options": "", "diff_error": "",
        "history": [], "best_valid_primary": 0.60, "concepts": [],
        "active_concept_id": "", "iteration": 3, "propose_attempt": 0,
        "system_prompt": "sys", "model": "claude-sonnet-5",
    })
    assert out["edited_files"], "the run should continue, not die"
    assert "NOT the current best" in seen["prompt"], \
        "the model was handed baseline code while being told it was the incumbent"
    assert res.read_recovery(root)[0]["action"] == "fallback-to-root"


def test_repair_shows_the_code_that_actually_failed(tmp_path, monkeypatch):
    """Regression guard for the whole point of repair mode: debugging the file
    the model never wrote is worse than useless."""
    root = tmp_path / "repo"
    (root / "runs" / "exp_0007").mkdir(parents=True)
    (root / "runs" / "best").mkdir(parents=True)
    for f in ("baseline.py", "data.py"):
        (root / f).write_text("# pristine\n")
        (root / "runs" / "exp_0007" / f).write_text(f"# THE BROKEN {f}\n")
        (root / "runs" / "best" / f).write_text(f"# the good {f}\n")

    seen = {}

    def _capture(model, system_prompt, user_content):
        seen["prompt"] = user_content
        return ({"concept": "c", "hypothesis": "h", "description": "d",
                 "files": [{"path": "baseline.py", "content": "x = 1\n"}]}, 1, 1)

    monkeypatch.setattr(propose_mod, "_call_anthropic", _capture)
    propose_mod.llm_generate({
        "mode": "repair", "repo_root": str(root),
        "best_exp_dir": str(root / "runs" / "best"),
        "repair_exp_dir": str(root / "runs" / "exp_0007"),
        "repair_error": "NameError: name 'lgb' is not defined",
        "editable_files": ["baseline.py", "data.py"], "eda_summary": {},
        "context_summary": "", "retrieved_options": "", "diff_error": "",
        "history": [], "best_valid_primary": 0.60,
        "concepts": [{"id": "c1", "statement": "s", "status": "active",
                      "rationale": "", "closed_reason": "", "opened_at_iteration": 1,
                      "attempts": []}],
        "active_concept_id": "c1", "iteration": 7, "propose_attempt": 0,
        "system_prompt": "sys", "model": "claude-sonnet-5",
    })
    assert "THE BROKEN baseline.py" in seen["prompt"]
    assert "the good baseline.py" not in seen["prompt"]


# =========================================================================
# Malformed proposals
# =========================================================================

def _payload(**over):
    p = {"concept": "c", "hypothesis": "h", "description": "d",
         "files": [{"path": "baseline.py", "content": "x = 1\n"}]}
    p.update(over)
    return p


@pytest.mark.parametrize("key", ["concept", "hypothesis", "description"])
def test_an_empty_required_field_is_rejected(key):
    with pytest.raises(res.ProposalError, match="missing"):
        propose_mod._validate_payload(_payload(**{key: ""}))


def test_no_files_is_rejected():
    with pytest.raises(res.ProposalError):
        propose_mod._validate_payload(_payload(files=[]))


@pytest.mark.parametrize("entry", [{"path": "baseline.py"}, {"content": "x"}, "notadict"])
def test_a_malformed_file_entry_is_rejected(entry):
    with pytest.raises(res.ProposalError, match="malformed"):
        propose_mod._validate_payload(_payload(files=[entry]))


def test_a_duplicated_path_is_rejected_rather_than_silently_last_wins():
    """`{f["path"]: f["content"] for f in files}` keeps the LAST entry. That is
    deterministic, but it is not obviously the model's intent, and a training
    run on code nobody chose costs more than one regeneration."""
    with pytest.raises(res.ProposalError, match="more than once"):
        propose_mod._validate_payload(_payload(files=[
            {"path": "baseline.py", "content": "A"},
            {"path": "baseline.py", "content": "B"},
        ]))


def test_a_wellformed_payload_passes():
    assert propose_mod._validate_payload(_payload())["concept"] == "c"


# =========================================================================
# Critic boundaries — epsilon is a decision threshold, so its edges matter
# =========================================================================

@pytest.mark.parametrize("delta,expected", [
    (0.010, "improved"),
    (0.002001, "improved"),
    (0.002, "parity"),        # inclusive lower edge of "not better"
    (0.001999, "parity"),
    (0.0, "parity"),
    (-0.002, "parity"),       # inclusive upper edge of "not worse"
    (-0.002001, "failed"),
    (-0.5, "failed"),
])
def test_epsilon_boundaries_are_exact(delta, expected):
    out = classify_outcome({"step_failed": False, "delta": delta, "epsilon": 0.002})
    assert out["outcome"] == expected


def test_a_failed_step_is_an_error_whatever_the_delta_says():
    assert classify_outcome({"step_failed": True, "delta": 99.0,
                             "epsilon": 0.002})["outcome"] == "error"


# =========================================================================
# Error classification precedence
# =========================================================================

def test_a_deterministic_bug_whose_message_contains_nan_is_not_reseeded():
    """The classifier is regex-based, so an ordinary error message that happens
    to contain 'nan' or 'inf' would otherwise earn a free reseed that cannot
    possibly work."""
    err = "ValueError: could not convert string to float: 'nan'"
    assert res.classify_run_error(err) == "deterministic"
    assert res.repair_strategy(err, attempt=0) == "repair"


def test_an_importerror_mentioning_infinity_is_still_deterministic():
    assert res.classify_run_error("ImportError: cannot import name 'inf'") == "deterministic"


def test_a_genuine_nan_loss_is_still_stochastic():
    """The precedence fix must not break the case it was built around."""
    assert res.classify_run_error("RuntimeWarning: loss became nan at epoch 3") == "nondeterministic"


# =========================================================================
# Deliverable shape — what a grader actually reads
# =========================================================================

def test_the_run_log_carries_a_real_diff(tmp_path):
    """Deliverable 3 asks for "the code diff applied". Only the filenames were
    recorded, and runs/ is gitignored — so the graded log said
    `["baseline.py"]` and nothing about what the agent changed."""
    src, dst = tmp_path / "a", tmp_path / "b"
    src.mkdir(); dst.mkdir()
    (src / "baseline.py").write_text("k = 16\nlr = 1e-3\n")
    (dst / "baseline.py").write_text("k = 32\nlr = 1e-3\n")
    (src / "data.py").write_text("same\n")
    (dst / "data.py").write_text("same\n")
    d = tools.unified_diff(str(src), str(dst), ["baseline.py", "data.py"])
    assert "-k = 16" in d and "+k = 32" in d
    assert "data.py" not in d, "unchanged files should not appear"


def test_an_identical_experiment_says_so_in_the_diff(tmp_path):
    """The no-op case is the one worth naming: it is how two iterations were
    lost to code that never executed."""
    src = tmp_path / "a"; src.mkdir()
    (src / "baseline.py").write_text("x = 1\n")
    assert "no change" in tools.unified_diff(str(src), str(src), ["baseline.py"])


def test_a_huge_diff_is_truncated_not_dumped(tmp_path):
    src, dst = tmp_path / "a", tmp_path / "b"
    src.mkdir(); dst.mkdir()
    (src / "baseline.py").write_text("")
    (dst / "baseline.py").write_text("".join(f"line {i}\n" for i in range(5000)))
    d = tools.unified_diff(str(src), str(dst), ["baseline.py"])
    assert "elided" in d and len(d.splitlines()) < tools.MAX_DIFF_LINES + 10


def test_diffing_never_breaks_an_otherwise_good_experiment(tmp_path):
    """Documentation must not be able to kill the thing it documents."""
    assert tools.unified_diff(str(tmp_path / "gone"), str(tmp_path / "also_gone"),
                              ["baseline.py"]) is not None


def test_per_metric_scores_reach_the_run_log():
    """The judging formula is mean(delta(GAUC), delta(nDCG@5)); logging only the
    primary leaves the results table to be reconstructed by hand."""
    out = exp_mod.collect_metrics({"step_failed": False, "run_stdout": _summary("0.6019")})
    assert out["valid_metrics"]["GAUC"] == 0.6674
    assert out["valid_metrics"]["nDCG@5"] == 0.5363
    assert out["test_metrics"]["primary"] == 0.5957


def test_finalize_computes_the_results_table(tmp_path, monkeypatch):
    from autoresearch_lg import graph as graph_mod
    import json as _json
    (tmp_path / "baseline_scores.json").write_text(_json.dumps({"scores": {"fm_official": {
        "valid": {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016},
        "test": {"GAUC": 0.6610, "nDCG@5": 0.5282, "primary": 0.5946}}}}))
    monkeypatch.setattr(graph_mod.tools, "make_submission", lambda *a, **k: (True, "ok"))
    report = graph_mod.finalize({
        "repo_root": str(tmp_path), "data_dir": str(tmp_path), "start_time": 0.0,
        "best_valid_primary": 0.6034, "best_test_primary": 0.5978,
        "best_checkpoint_id": 4, "best_exp_dir": "", "iteration": 12, "concepts": [],
        "history": [{"outcome": "improved", "metrics": {
            "valid_primary": 0.6034, "test_primary": 0.5978,
            "valid": {"GAUC": 0.6700, "nDCG@5": 0.5368, "primary": 0.6034},
            "test": {"GAUC": 0.6650, "nDCG@5": 0.5306, "primary": 0.5978}}}],
    })["resource_report"]
    assert report["test_delta_vs_baseline"] == {"GAUC": 0.004, "nDCG@5": 0.0024}
    # mean of the per-metric deltas == delta(primary), by construction
    assert report["score_dataset"] == pytest.approx(0.5978 - 0.5946, abs=1e-6)
    assert report["baseline_metrics"]["test"]["GAUC"] == 0.6610
