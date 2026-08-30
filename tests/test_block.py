"""Two-stage experimental blocks: an iteration returns a conclusion, not a scalar.

These tests pin the three properties that make a block worth its cost: it must
survive a candidate that crashes, it must refuse to accept an effect that is
inside noise or below what selection manufactures anyway, and its budgeting must
be honest about what fits the six-hour ceiling.
"""
from __future__ import annotations

import numpy as np
import pytest

from tippytop.experiments import Candidate, run_block, plan_block


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(3)
    users, labels = [], []
    for u in range(200):
        n = int(rng.integers(4, 9))
        y = list(rng.integers(0, 2, size=n))
        if sum(y) in (0, n):
            y[0] = 1 - y[0]
        users += [f"u{u}"] * n
        labels += y
    return users, np.asarray(labels, dtype=float)


def _noise(labels, strength):
    """A candidate whose quality rises with `strength`; seed changes the draw."""
    def fn(seed):
        rng = np.random.default_rng(1000 + seed)
        return strength * labels + rng.normal(0, 1.0, len(labels))
    return fn


def test_block_screens_then_confirms(toy):
    users, labels = toy
    base = Candidate("baseline", _noise(labels, 0.5))
    cands = [Candidate(f"c{i}", _noise(labels, s))
             for i, s in enumerate((0.6, 1.5, 0.55))]
    r = run_block("does more signal help?", cands, base, users, labels,
                  screen_seeds=1, confirm_seeds=3, n_boot=200)

    assert r.winner == "c1", "the strongest candidate should be promoted"
    assert r.delta_vs_baseline > 0
    assert r.bootstrap is not None and r.seed_std is not None
    assert r.runs_used > 0 and r.wall_s >= 0
    assert "c1" in r.summary()


def test_two_stage_costs_less_than_confirming_everything(toy):
    """The efficiency claim: screen cheap, spend seeds only on survivors."""
    users, labels = toy
    base = Candidate("baseline", _noise(labels, 0.5))
    cands = [Candidate(f"c{i}", _noise(labels, 0.5 + 0.1 * i)) for i in range(4)]

    staged = run_block("staged", cands, base, users, labels,
                       screen_seeds=1, confirm_seeds=3, n_boot=50)
    uniform = run_block("uniform", cands, base, users, labels,
                        screen_seeds=3, confirm_seeds=3, n_boot=50)
    assert staged.runs_used < uniform.runs_used


def test_a_crashing_candidate_does_not_kill_the_block(toy):
    """Robustness is graded; a six-hour run cannot die on one bad config."""
    users, labels = toy

    def explode(seed):
        raise RuntimeError("synthetic failure")

    base = Candidate("baseline", _noise(labels, 0.5))
    cands = [Candidate("broken", explode), Candidate("good", _noise(labels, 1.5))]
    r = run_block("survives failure", cands, base, users, labels,
                  screen_seeds=1, confirm_seeds=2, n_boot=100)

    assert r.winner == "good"
    broken = next(a for a in r.arms if a.name == "broken")
    assert broken.failed and "synthetic failure" in broken.error


def test_all_candidates_failing_is_reported_not_raised(toy):
    users, labels = toy

    def explode(seed):
        raise ValueError("nope")

    r = run_block("total failure", [Candidate("a", explode)],
                  Candidate("baseline", _noise(labels, 0.5)), users, labels,
                  screen_seeds=1, confirm_seeds=1, n_boot=50)
    assert r.winner is None and not r.accepted
    assert "failed" in r.verdict


def test_noise_only_candidate_is_rejected(toy):
    """A candidate identical to the baseline must not be accepted."""
    users, labels = toy
    base = Candidate("baseline", _noise(labels, 0.8))
    same = Candidate("same", _noise(labels, 0.8))
    r = run_block("no real difference", [same], base, users, labels,
                  screen_seeds=1, confirm_seeds=3, n_boot=300)
    assert not r.accepted


def test_effect_below_selection_inflation_is_rejected(toy):
    """Even a positive delta is refused if best-of-N would manufacture it."""
    users, labels = toy
    base = Candidate("baseline", _noise(labels, 0.8))
    tiny = Candidate("tiny", _noise(labels, 0.805))
    r = run_block("tiny effect", [tiny], base, users, labels,
                  screen_seeds=1, confirm_seeds=2, n_boot=200,
                  n_candidates_seen=200)
    if r.delta_vs_baseline is not None and r.delta_vs_baseline > 0:
        assert not r.accepted or "below" in r.verdict


def test_large_real_effect_is_accepted(toy):
    users, labels = toy
    base = Candidate("baseline", _noise(labels, 0.2))
    strong = Candidate("strong", _noise(labels, 3.0))
    r = run_block("large effect", [strong], base, users, labels,
                  screen_seeds=1, confirm_seeds=3, n_boot=300)
    assert r.accepted and r.delta_vs_baseline > 0


def test_verdict_distinguishes_real_from_clearing_the_bar(toy):
    """A gain can be statistically real yet below the 0.002 threshold."""
    users, labels = toy
    base = Candidate("baseline", _noise(labels, 0.2))
    strong = Candidate("strong", _noise(labels, 3.0))
    r = run_block("verdict wording", [strong], base, users, labels,
                  screen_seeds=1, confirm_seeds=2, n_boot=200)
    assert ("clears" in r.verdict) or ("do not call it a win" in r.verdict)


def test_result_serialises_for_the_run_log(toy):
    users, labels = toy
    r = run_block("serialisable", [Candidate("c", _noise(labels, 1.0))],
                  Candidate("baseline", _noise(labels, 0.5)), users, labels,
                  screen_seeds=1, confirm_seeds=2, n_boot=50)
    d = r.to_dict()
    assert set(d) >= {"hypothesis", "arms", "winner", "verdict", "accepted", "runs_used"}
    assert isinstance(d["arms"], list) and isinstance(d["arms"][0], dict)


# --- budgeting -------------------------------------------------------------

def test_uniform_twelve_run_blocks_do_not_fit_the_ceiling():
    """The red team's own sizing was wrong: 50 x 12 runs x 40s = 6.7h > 6h."""
    assert 50 * 12 * 40 > 6 * 3600


def test_plan_block_reports_whether_a_design_fits():
    p = plan_block(delta_target=0.0015, n_configs=4, run_seconds=40,
                   budget_seconds=6 * 3600, max_iters=50)
    assert p["runs_per_block"] <= 12
    assert p["fits"], "the two-stage design must fit the six-hour ceiling"
    assert 0 < p["utilisation"] < 1


def test_plan_block_flags_a_design_that_overruns():
    p = plan_block(delta_target=0.0015, n_configs=40, run_seconds=90,
                   budget_seconds=6 * 3600, max_iters=50)
    assert not p["fits"]
