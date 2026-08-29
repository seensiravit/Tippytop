"""The `parity` outcome: not-better must not be treated as worse.

Motivated by a measured failure in this repo's own run log. The agent proposed
field-aware FM, implemented it from scratch, scored valid 0.6015 against a 0.6015
incumbent, and the router pivoted away from it. FFM is the strongest model
measured here (0.6025 +-0.0004 vs FM's 0.6016 +-0.0003 over six seeds each) --
the agent had the right idea and the two-way outcome split discarded it.

A first implementation of a good concept arrives untuned and lands at parity.
Pivoting on that throws away the concept before it has been given a chance.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoresearch_lg.critic import classify_outcome  # noqa: E402
from autoresearch_lg.graph import router  # noqa: E402

EPS = 0.002


def _state(delta, failed=False):
    return {"step_failed": failed, "delta": delta, "epsilon": EPS}


def _router_state(outcome, tune_count=0, tune_cap=3):
    return {
        "outcome": outcome, "retry_count": 0, "retry_cap": 3,
        "tune_count": tune_count, "tune_cap": tune_cap,
        "concepts": [{"id": "c1", "status": "active", "closed_reason": ""}],
        "active_concept_id": "c1",
    }


@pytest.mark.parametrize("delta,expected", [
    (0.005, "improved"),      # clear win
    (0.0021, "improved"),     # just over epsilon
    (0.0, "parity"),          # exactly the incumbent — the FFM case
    (-0.0015, "parity"),      # inside the noise band
    (-EPS, "parity"),         # boundary is inclusive
    (-0.010, "failed"),       # a real regression
])
def test_three_way_classification(delta, expected):
    assert classify_outcome(_state(delta))["outcome"] == expected


def test_crash_still_wins_over_everything():
    assert classify_outcome(_state(0.5, failed=True))["outcome"] == "error"


def test_parity_tunes_rather_than_pivoting():
    """The regression this file exists to prevent."""
    assert router(_router_state("parity"))["mode"] == "tune"


def test_clear_regression_still_pivots():
    assert router(_router_state("failed"))["mode"] == "pivot"


def test_parity_is_bounded_by_tune_cap():
    """A concept that only ever reaches parity must not tune forever."""
    at_cap = router(_router_state("parity", tune_count=2, tune_cap=3))
    assert at_cap["mode"] == "pivot"
    closed = [c for c in at_cap["concepts"] if c["status"] == "closed"]
    assert closed and "parity" in closed[0]["closed_reason"]


def test_parity_pivots_while_improvement_expands():
    """Expand is for successes; a concept that never delivered gets abandoned."""
    parity = router(_router_state("parity", tune_count=2, tune_cap=3))
    improved = router(_router_state("improved", tune_count=2, tune_cap=3))
    assert parity["mode"] == "pivot"
    assert improved["mode"] == "expand"


def test_improved_path_is_unchanged():
    assert router(_router_state("improved"))["mode"] == "tune"
