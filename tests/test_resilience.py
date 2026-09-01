"""The failure policy: what happens when things the agent does not control break.

The harness already routed every *expected* experiment failure into a
FailureRecord. These tests cover the four layers it did not:

  L1 transient provider errors   -> retried with backoff, not fatal
  L2 malformed model output      -> a typed error the retry policy understands
  L3 defective generated code    -> repaired with the traceback, not rerolled
  L4 terminal                    -> the graded deliverables still get written

Each test names the failure it prevents. A test that only asserts a helper
returns the right string is not evidence that a six-hour unattended run
survives; where possible these drive the real router and the real graph nodes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch_lg import graph as graph_mod  # noqa: E402
from autoresearch_lg import resilience as res  # noqa: E402
from autoresearch_lg.bootstrap import reconstruct_counters  # noqa: E402
from autoresearch_lg.critic import update_counters  # noqa: E402


# --- L1: which exceptions are worth retrying ------------------------------

class _Fake(Exception):
    def __init__(self, msg="", status_code=None):
        super().__init__(msg)
        self.status_code = status_code


def _named(name, base=_Fake, **kw):
    return type(name, (base,), {})(**kw)


@pytest.mark.parametrize("name", [
    "RateLimitError", "OverloadedError", "InternalServerError",
    "APITimeoutError", "APIConnectionError", "ServiceUnavailableError",
])
def test_transient_provider_errors_are_retried(name):
    """A 529 during a six-hour run must cost 30 seconds, not the whole run."""
    assert res.is_transient(_named(name))


@pytest.mark.parametrize("name", [
    "AuthenticationError", "PermissionDeniedError", "BadRequestError",
    "NotFoundError", "RequestTooLargeError",
])
def test_permanent_errors_are_not_retried(name):
    """A bad API key will not fix itself. Backing off five times before dying
    just spends the wall-clock that finalize needs."""
    assert not res.is_transient(_named(name))


def test_a_4xx_status_on_a_retryable_class_is_still_not_retried():
    assert not res.is_transient(_named("APIStatusError", status_code=400))
    assert res.is_transient(_named("APIStatusError", status_code=503))


def test_control_flow_exceptions_are_never_retried():
    assert not res.is_transient(KeyboardInterrupt())
    assert not res.is_transient(MemoryError())


def test_unknown_exceptions_get_the_benefit_of_the_doubt():
    """Losing a six-hour run to an exception class we have not seen is a worse
    error than spending one extra attempt on it."""
    assert res.is_transient(RuntimeError("something new"))


def test_our_own_parse_failure_is_retryable():
    """A different sample from the model may well be well-formed."""
    assert res.is_transient(res.ProposalError("no tool_use block"))


def test_retry_policy_is_actually_attached_to_propose():
    """The policy existing in a module is not the same as it being wired in."""
    compiled = graph_mod.build_graph()
    spec = compiled.builder.nodes["propose"]
    assert spec.retry_policy, "propose has no retry policy"
    # RetryPolicy is a NamedTuple, so an isinstance(..., tuple) check would
    # happily iterate its *fields*. Normalise on the attribute instead.
    policies = ([spec.retry_policy] if hasattr(spec.retry_policy, "max_attempts")
                else list(spec.retry_policy))
    assert any(p.max_attempts >= 3 for p in policies)
    assert any(p.retry_on is res.is_transient for p in policies), \
        "propose retries on the default policy, not ours — a bad API key would " \
        "back off five times before dying"
    # LangGraph compiles an error handler into its own node and points the
    # guarded node at it, so the assertion has to look at both.
    assert spec.error_handler_node == "__error_handler__propose"
    assert "__error_handler__propose" in compiled.builder.nodes, \
        "no terminal failsafe on propose — a dead provider would kill the run"


def test_experiment_and_critic_are_not_retried():
    """Second-order check. Retrying `experiment` would silently re-run training
    on an already-broken idea; retrying `critic` would double-write runs.jsonl.
    A blanket set_node_defaults would have done both."""
    compiled = graph_mod.build_graph()
    for name in ("experiment", "critic"):
        assert not compiled.builder.nodes[name].retry_policy, name


# --- L3: repair, not reroll ------------------------------------------------

@pytest.mark.parametrize("err", [
    "NameError: name 'np' is not defined",
    "ValueError: operands could not be broadcast together with shapes (3,) (4,)",
    "AttributeError: 'NoneType' object has no attribute 'shape'",
    "IndentationError: unexpected indent",
])
def test_deterministic_crashes_are_repaired(err):
    """Rerolling the seed on a NameError is a guaranteed no-op that still costs
    a full training run. Three of them spend ~30 min of a 6h budget."""
    assert res.repair_strategy(err, attempt=0) == "repair"


@pytest.mark.parametrize("err", [
    "RuntimeWarning: invalid value encountered in log",
    "loss became nan at epoch 3",
    "numpy.linalg.LinAlgError: Singular matrix",
    "timed out (>10min)",
])
def test_stochastic_crashes_get_one_free_reseed(err):
    """No tokens, no model call — worth exactly one attempt before paying."""
    assert res.repair_strategy(err, attempt=0) == "reseed"
    # ...and only one. After that, pay for a real fix.
    assert res.repair_strategy(err, attempt=1) == "repair"


def test_router_sends_a_deterministic_crash_into_repair(tmp_path):
    state = {
        "outcome": "error", "retry_count": 0, "retry_cap": 3, "tune_count": 0,
        "tune_cap": 3, "iteration": 4, "repo_root": str(tmp_path),
        "failure_error": "NameError: name 'lgb' is not defined",
        "exp_dir": str(tmp_path / "runs" / "exp_0005"),
        "concepts": [{"id": "c1", "status": "active", "closed_reason": ""}],
        "active_concept_id": "c1",
    }
    out = graph_mod.router(state)
    assert out["mode"] == "repair"
    assert out["retry_now"] is False, "repair must not re-run the same code"
    assert "NameError" in out["repair_error"]
    assert out["repair_exp_dir"].endswith("exp_0005")


def test_router_reseeds_a_stochastic_crash_without_an_llm_call(tmp_path):
    state = {
        "outcome": "error", "retry_count": 0, "retry_cap": 3, "tune_count": 0,
        "tune_cap": 3, "iteration": 4, "repo_root": str(tmp_path),
        "failure_error": "loss became nan at epoch 2", "exp_dir": "",
        "concepts": [{"id": "c1", "status": "active", "closed_reason": ""}],
        "active_concept_id": "c1",
    }
    out = graph_mod.router(state)
    assert out["retry_now"] is True and "mode" not in out


def test_exhausting_the_repair_budget_pivots_and_closes_the_concept(tmp_path):
    state = {
        "outcome": "error", "retry_count": 3, "retry_cap": 3, "tune_count": 0,
        "tune_cap": 3, "iteration": 9, "repo_root": str(tmp_path),
        "failure_error": "NameError", "exp_dir": "",
        "concepts": [{"id": "c1", "status": "active", "closed_reason": ""}],
        "active_concept_id": "c1",
    }
    out = graph_mod.router(state)
    assert out["mode"] == "pivot"
    assert out["concepts"][0]["status"] == "closed"
    assert "unrecoverable" in out["concepts"][0]["closed_reason"]


def test_every_recovery_is_written_down(tmp_path):
    """Robustness (20%) is graded on recovery. A run that recovered silently is
    indistinguishable from one that never had a problem."""
    state = {
        "outcome": "error", "retry_count": 0, "retry_cap": 3, "tune_count": 0,
        "tune_cap": 3, "iteration": 2, "repo_root": str(tmp_path),
        "failure_error": "TypeError: bad operand", "exp_dir": "",
        "concepts": [{"id": "c1", "status": "active", "closed_reason": ""}],
        "active_concept_id": "c1",
    }
    graph_mod.router(state)
    events = res.read_recovery(tmp_path)
    assert len(events) == 1
    assert events[0]["action"] == "repair" and events[0]["layer"] == "experiment"


# --- crashes must not be mistaken for a plateau ---------------------------

def _counters_after(outcomes, n_start=0):
    st = {"no_improve_count": n_start, "iteration": 0, "concepts": [], "active_concept_id": ""}
    for o in outcomes:
        st.update(update_counters({**st, "outcome": o, "step_failed": o == "error",
                                   "valid_primary": 0.0}))
    return st["no_improve_count"]


def test_three_crashes_do_not_converge_the_run():
    """The bug this fixes: n_plateau=3 plus 'every non-improvement counts' meant
    three broken experiments ended the run and shipped early — the agent
    stopping because it hit bugs, not because it was out of ideas."""
    assert _counters_after(["error", "error", "error"]) == 0


def test_three_completed_non_improvements_still_converge():
    """The plateau rule itself must survive the fix."""
    assert _counters_after(["failed", "parity", "failed"]) == 3


def test_an_improvement_resets_the_plateau():
    assert _counters_after(["failed", "failed", "improved"]) == 0


def test_resume_reconstructs_the_same_plateau_count():
    """If disk-reconstruction disagreed with the live counter, resuming a run
    would silently change when it converges."""
    history = [{"outcome": o} for o in ["improved", "failed", "error", "failed"]]
    live = _counters_after(["failed", "error", "failed"])
    _, _, from_disk = reconstruct_counters(history, [], "")
    assert from_disk == live == 2


# --- L4: budget and the guaranteed deliverable ----------------------------

def test_the_loop_stops_while_there_is_still_time_to_ship():
    """The old check only asked whether the budget was already spent, so at
    5h58m it would start a 10-minute experiment, overshoot, and leave nothing
    for finalize."""
    hist = [{"wall_clock_s": 300.0}] * 5
    ok, _ = res.budget_allows_another_experiment(elapsed=21000, max_wall_seconds=21600,
                                                 history=hist)
    assert not ok


def test_a_fresh_run_is_allowed_to_start():
    ok, why = res.budget_allows_another_experiment(0, 21600, [])
    assert ok and why == ""


def test_one_timeout_does_not_scare_the_agent_into_stopping_early():
    """Median, not mean: a single 600s timeout must not double the estimate."""
    hist = [{"wall_clock_s": 60.0}] * 8 + [{"wall_clock_s": 600.0}]
    assert res.experiment_time_estimate(hist) < 120


def test_finalize_reserve_is_actually_held_back():
    ok, why = res.budget_allows_another_experiment(21599, 21600, [])
    assert not ok and "finalize" in why


def test_convergence_reports_why_it_stopped(tmp_path):
    base = {"start_time": 0.0, "no_improve_count": 0, "n_plateau": 3, "iteration": 1,
            "epsilon": 0.002, "max_iterations": 50, "max_wall_seconds": 10 ** 12,
            "history": [], "repo_root": str(tmp_path), "llm_unavailable": ""}
    assert graph_mod.check_convergence(base)["converged"] is False
    flat = graph_mod.check_convergence(
        {**base, "history": [{"outcome": "improved", "metrics": {"valid_primary": 0.60}}] * 5})
    assert flat["converged"] and flat["stop_reason"].startswith("plateau")

def test_a_dead_provider_ships_instead_of_dying(tmp_path):
    """The L4 failsafe. propose being unreachable must end the run at finalize,
    not with a traceback and no deliverables."""
    out = graph_mod.check_convergence({
        "start_time": 0.0, "no_improve_count": 0, "n_plateau": 3, "iteration": 5,
        "epsilon": 0.002, "max_iterations": 50, "max_wall_seconds": 10 ** 12,
        "history": [], "repo_root": str(tmp_path), "llm_unavailable": "OverloadedError: ..."})
    assert out["converged"] and out["stop_reason"] == "llm-unavailable"

def test_propose_error_handler_routes_to_finalize(tmp_path):
    class _Err:
        node = "propose"
        error = _named("OverloadedError")

    cmd = graph_mod._propose_error_handler(
        {"repo_root": str(tmp_path), "iteration": 7}, _Err())
    assert cmd.goto == "finalize"
    assert cmd.update["converged"] is True
    assert "OverloadedError" in cmd.update["llm_unavailable"]
    assert res.read_recovery(tmp_path)[0]["action"] == "finalize-early"


def test_finalize_records_the_recovery_evidence(tmp_path, monkeypatch):
    for i, action in enumerate(["repair", "repair", "reseed"]):
        res.log_recovery(tmp_path, res.recovery_event(i, "experiment", "crash", action))
    monkeypatch.setattr(graph_mod.tools, "make_submission", lambda *a, **k: (True, "ok"))
    report = graph_mod.finalize({
        "repo_root": str(tmp_path), "data_dir": str(tmp_path), "start_time": 0.0,
        "best_valid_primary": 0.60, "best_test_primary": 0.59, "best_checkpoint_id": 1,
        "best_exp_dir": "", "iteration": 12, "history": [], "concepts": [],
        "stop_reason": "plateau",
    })["resource_report"]
    assert report["recovery_events"] == 3
    assert report["recovery_by_action"] == {"repair": 2, "reseed": 1}
    assert report["stop_reason"] == "plateau"
    assert json.loads((tmp_path / "resource_report.json").read_text())["recovery_events"] == 3


def test_logging_a_recovery_never_breaks_a_run(tmp_path):
    """Instrumentation must not be able to kill the thing it instruments."""
    res.log_recovery(tmp_path / "does" / "not" / "exist",
                     res.recovery_event(1, "llm", "x", "retry"))  # must not raise


# --- end to end: the loop under real failure ------------------------------
#
# The unit tests above check each policy in isolation. These drive the actual
# compiled graph, because the failure being prevented is a *whole-run* failure:
# every individual piece can be correct while the run still dies.

BASELINE_SCORES = {"scores": {"fm_official": {"valid": {"primary": 0.6016},
                                              "test": {"primary": 0.5946}}}}

GOOD_STDOUT = ("valid GAUC 0.6674 | nDCG@5 0.5363 | primary 0.6019\n"
               "test  GAUC 0.6591 | nDCG@5 0.5323 | primary 0.5957\n")


@pytest.fixture
def fake_repo(tmp_path):
    (tmp_path / "baseline_scores.json").write_text(json.dumps(BASELINE_SCORES))
    (tmp_path / "baseline.py").write_text("# baseline\n")
    (tmp_path / "data.py").write_text("# data\n")
    (tmp_path / "runs.jsonl").write_text("")
    return tmp_path


def _base_state(root, **over):
    st = {
        "repo_root": str(root), "data_dir": str(root), "model": "claude-sonnet-5",
        "max_iterations": 3, "max_wall_seconds": 3600.0, "epsilon": 0.002,
        "n_plateau": 3, "retry_cap": 2, "tune_cap": 3,
    }
    st.update(over)
    return st


def _stub_infra(monkeypatch, root, *, stdout=GOOD_STDOUT, crashed=False):
    from autoresearch_lg import experiment as exp_mod
    from autoresearch_lg import tools as tools_mod
    monkeypatch.setattr(tools_mod, "run_eda", lambda *a, **k: {})
    monkeypatch.setattr(tools_mod, "make_submission", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(exp_mod.tools, "run_baseline", lambda *a, **k: {
        "stdout": stdout, "wall_seconds": 1.0, "crashed": crashed, "timed_out": False})
    monkeypatch.setattr(exp_mod.tools, "make_experiment_dir",
                        lambda r, n: str(Path(r, "runs", n)))
    monkeypatch.setattr(exp_mod.tools, "write_experiment_files", lambda *a, **k: None)
    monkeypatch.setattr(exp_mod.tools, "read_experiment_files",
                        lambda d, fs: {f: "# x\n" for f in fs})
    monkeypatch.setattr("autoresearch_lg.propose.tools.read_experiment_files",
                        lambda d, fs: {f: "# x\n" for f in fs})


def test_a_provider_outage_ends_at_finalize_not_at_a_traceback(fake_repo, monkeypatch):
    """THE failure this whole module exists to prevent.

    Before: one unhandled OverloadedError in llm_generate propagated out of two
    sub-graphs and out of .stream(), and the run died with runs.jsonl on disk but
    no submission.csv and no resource_report.json — Deliverables 3 and 4 empty
    despite the work being done.
    """
    _stub_infra(monkeypatch, fake_repo)
    calls = {"n": 0}

    def _always_overloaded(*a, **k):
        calls["n"] += 1
        raise _named("OverloadedError")

    monkeypatch.setattr("autoresearch_lg.propose._call_anthropic", _always_overloaded)
    # Backoff is real; make the test fast without disabling the retry itself.
    monkeypatch.setattr(res, "LLM_RETRY",
                        res.RetryPolicy(max_attempts=3, initial_interval=0.01,
                                        backoff_factor=1.0, max_interval=0.01,
                                        jitter=False, retry_on=res.is_transient))
    import importlib
    import autoresearch_lg.graph as g
    importlib.reload(g)

    final = g.build_graph().invoke(_base_state(fake_repo),
                                   config={"recursion_limit": 60})

    assert calls["n"] >= 3, "the provider error was not retried at all"
    assert final["converged"] is True
    assert "OverloadedError" in final["llm_unavailable"]
    assert final["resource_report"]["submission"], "no submission produced"
    assert (fake_repo / "resource_report.json").exists(), \
        "the graded deliverable was not written"
    assert res.read_recovery(fake_repo)[-1]["action"] == "finalize-early"
    importlib.reload(g)


def test_a_transient_blip_is_survived_and_the_run_continues(fake_repo, monkeypatch):
    """One 429 must cost a retry, not the run."""
    _stub_infra(monkeypatch, fake_repo)
    calls = {"n": 0}

    def _flaky(model, system_prompt, user_content):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _named("RateLimitError")
        return ({"concept": "c", "hypothesis": "h", "description": "d",
                 "files": [{"path": "baseline.py", "content": "x = 1\n"}]}, 10, 5)

    monkeypatch.setattr("autoresearch_lg.propose._call_anthropic", _flaky)
    monkeypatch.setattr(res, "LLM_RETRY",
                        res.RetryPolicy(max_attempts=3, initial_interval=0.01,
                                        backoff_factor=1.0, max_interval=0.01,
                                        jitter=False, retry_on=res.is_transient))
    import importlib
    import autoresearch_lg.graph as g
    importlib.reload(g)

    final = g.build_graph().invoke(_base_state(fake_repo, max_iterations=1),
                                   config={"recursion_limit": 80})

    assert calls["n"] >= 2, "the flaky call was not retried"
    assert not final.get("llm_unavailable"), "a single 429 should not end the run"
    assert final["iteration"] == 1, "the experiment did not actually run"
    assert final["resource_report"]["iterations"] == 1
    importlib.reload(g)


def test_a_crashing_experiment_is_repaired_with_its_own_traceback(fake_repo, monkeypatch):
    """The L3 loop, end to end: the second proposal must be told what broke."""
    _stub_infra(monkeypatch, fake_repo,
                stdout="Traceback...\nNameError: name 'lgb' is not defined\n",
                crashed=True)
    seen: list[str] = []

    def _record(model, system_prompt, user_content):
        seen.append(user_content)
        return ({"concept": "c", "hypothesis": "h", "description": "d",
                 "files": [{"path": "baseline.py", "content": "x = 1\n"}]}, 10, 5)

    monkeypatch.setattr("autoresearch_lg.propose._call_anthropic", _record)
    import importlib
    import autoresearch_lg.graph as g
    importlib.reload(g)

    g.build_graph().invoke(_base_state(fake_repo, max_iterations=2, retry_cap=2),
                           config={"recursion_limit": 120})

    assert len(seen) >= 2, "the agent never asked for a fix"
    repair_prompts = [p for p in seen if "MODE: repair" in p]
    assert repair_prompts, "no repair prompt was ever built"
    assert "NameError: name 'lgb' is not defined" in repair_prompts[0], \
        "the repair prompt did not carry the traceback — the model is debugging blind"
    importlib.reload(g)
