"""Statistical infrastructure: paired bootstrap, power, transfer, selection.

The load-bearing test is ``test_decomposition_matches_kit_exactly``. Every
bootstrap interval is computed from a per-user decomposition of the metric rather
than by re-running the frozen kit's ``evaluate`` thousands of times; if that
decomposition ever drifts from the kit, every interval silently becomes wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from tippytop.kit import evaluate as kit_evaluate
from tippytop.stats import (per_user_stats, primary_from, paired_bootstrap,
                            bootstrap_ci, verify_against_kit, seeds_needed,
                            is_detectable, expected_max, selection_inflation,
                            recommend_seeds, transfer_slope, rescale_target,
                            required_valid_gain, rank_within_user, rank_average,
                            topk_selection, compare_argmax_vs_topk, diagnose_eval_split)
from tippytop.stats.power import expected_max_asymptotic


@pytest.fixture(scope="module")
def toy():
    """Users with mixed label patterns, including the metric's edge cases."""
    rng = np.random.default_rng(7)
    users, labels = [], []
    for u in range(300):
        n = int(rng.integers(3, 10))
        if u % 9 == 0:                       # all-negative: nDCG pinned at 0
            y = [0] * n
        elif u % 11 == 0:                    # all-positive: nDCG pinned at 1
            y = [1] * n
        else:
            y = list(rng.integers(0, 2, size=n))
            if sum(y) in (0, n):
                y[0] = 1 - y[0]
        users += [f"u{u}"] * n
        labels += y
    return users, np.asarray(labels, dtype=float)


# --- the decomposition must equal the frozen kit --------------------------

def test_decomposition_matches_kit_exactly(toy):
    users, labels = toy
    rng = np.random.default_rng(0)
    for _ in range(5):
        scores = rng.normal(size=len(labels))
        ours = primary_from(per_user_stats(users, labels, scores))
        theirs = kit_evaluate(users, labels, scores)["primary"]
        assert abs(ours - theirs) < 1e-12, f"{ours} != {theirs}"


def test_verify_against_kit_helper(toy):
    users, labels = toy
    verify_against_kit(users, labels, np.random.default_rng(1).normal(size=len(labels)))


def test_decomposition_handles_all_negative_and_all_positive(toy):
    """Edge-case users must be counted the way the kit counts them."""
    users, labels = toy
    st = per_user_stats(users, labels, np.zeros(len(labels)))
    assert (~st.eligible).sum() > 0, "fixture must contain GAUC-ineligible users"
    assert st.ndcg[st.npos == 0].max() == 0.0, "all-negative users score nDCG 0"


# --- paired bootstrap ------------------------------------------------------

def test_identical_models_give_zero_delta_and_ci_containing_zero(toy):
    users, labels = toy
    s = np.random.default_rng(2).normal(size=len(labels))
    a = per_user_stats(users, labels, s)
    r = paired_bootstrap(a, per_user_stats(users, labels, s.copy()), n_boot=300)
    assert r["delta"] == pytest.approx(0.0, abs=1e-12)
    assert r["ci_low"] <= 0 <= r["ci_high"]
    assert not r["significant"]


def test_clearly_better_model_is_detected(toy):
    """A model given the labels themselves must beat noise, significantly."""
    users, labels = toy
    good = per_user_stats(users, labels, labels + 0.01 * np.random.default_rng(3).normal(size=len(labels)))
    bad = per_user_stats(users, labels, np.random.default_rng(4).normal(size=len(labels)))
    r = paired_bootstrap(good, bad, n_boot=300)
    assert r["delta"] > 0 and r["significant"]
    assert r["ci_low"] > 0


def test_pairing_is_enforced(toy):
    """Unpaired inputs must raise rather than silently produce a wrong interval."""
    users, labels = toy
    a = per_user_stats(users, labels, np.zeros(len(labels)))
    b = per_user_stats([u + "x" for u in users], labels, np.zeros(len(labels)))
    with pytest.raises(ValueError, match="same users"):
        paired_bootstrap(a, b, n_boot=10)


def test_paired_interval_is_tighter_than_unpaired(toy):
    """The whole point of pairing: user difficulty cancels in the difference."""
    users, labels = toy
    rng = np.random.default_rng(5)
    base = rng.normal(size=len(labels))
    a = per_user_stats(users, labels, base)
    b = per_user_stats(users, labels, base + 0.05 * rng.normal(size=len(labels)))
    paired = paired_bootstrap(a, b, n_boot=400)["se"]
    unpaired = bootstrap_ci(a, n_boot=400)["se"] + bootstrap_ci(b, n_boot=400)["se"]
    assert paired < unpaired


# --- power -----------------------------------------------------------------

def test_seeds_needed_matches_hand_calculation():
    # delta 0.0015, sigma 0.0006 -> ~2.6 seeds per arm
    assert 2.0 < seeds_needed(0.0015, 0.0006) < 3.5


def test_seeds_scale_inversely_with_squared_effect():
    assert seeds_needed(0.001) == pytest.approx(4 * seeds_needed(0.002), rel=1e-9)


def test_tiny_effects_are_flagged_undetectable():
    """The 5->15 ensemble step needs ~144 seeds/arm — it must not be recommended."""
    r = is_detectable(0.0002, 0.0006, max_seeds=10)
    assert not r["detectable"]
    assert r["seeds_needed"] > 100


def test_asymptotic_formula_overshoots_at_small_n():
    """The error that inflated this project's own headline statistic by 24%."""
    for n in (3, 10, 50):
        assert expected_max_asymptotic(n) > expected_max(n), n
    assert expected_max_asymptotic(50) / expected_max(50) > 1.2


def test_corrected_selection_inflation_value():
    """Best-of-50 at one seed manufactures ~0.00135, not the 0.00168 first published."""
    r = selection_inflation(50, sigma=0.0006, seeds_per_candidate=1)
    assert r["expected_inflation"] == pytest.approx(0.00135, abs=5e-5)
    assert 0.5 < r["share_of_epsilon"] < 0.8
    assert r["asymptotic_overstatement_pct"] > 20


def test_more_seeds_reduce_selection_inflation():
    one = selection_inflation(50, seeds_per_candidate=1)["expected_inflation"]
    three = selection_inflation(50, seeds_per_candidate=3)["expected_inflation"]
    assert three < one


def test_recommend_seeds_returns_feasible_count():
    r = recommend_seeds(0.0015, n_candidates=50)
    assert 1 <= r["seeds"] <= 5


# --- transfer --------------------------------------------------------------

def test_slope_recovered_on_synthetic_data():
    rng = np.random.default_rng(11)
    gv = np.linspace(0, 0.02, 25)
    gt = 0.4 * gv + rng.normal(0, 1e-5, size=25)
    r = transfer_slope(0.60 + gv, 0.59 + gt, baseline_valid=0.60, baseline_test=0.59,
                       n_boot=400)
    assert r["slope"] == pytest.approx(0.4, abs=0.05)
    assert r["ci_low"] < 0.4 < r["ci_high"]


def test_shrinkage_is_reported_when_slope_below_one():
    rng = np.random.default_rng(12)
    gv = np.linspace(0, 0.02, 20)
    gt = 0.3 * gv + rng.normal(0, 1e-5, size=20)
    r = transfer_slope(0.60 + gv, 0.59 + gt, baseline_valid=0.60, baseline_test=0.59,
                       n_boot=400)
    assert "below 1" in r["note"]


def test_too_few_models_is_refused_or_flagged():
    with pytest.raises(ValueError, match="at least 3"):
        transfer_slope([0.60, 0.61], [0.59, 0.60])
    r = transfer_slope([0.60, 0.61, 0.62], [0.59, 0.60, 0.61], n_boot=100)
    assert not r["reliable"]


def test_target_rescaling_round_trips():
    assert rescale_target(0.0031, 0.5) == pytest.approx(0.00155)
    assert required_valid_gain(0.002, 0.5) == pytest.approx(0.004)


# --- selection -------------------------------------------------------------

def test_rank_within_user_is_normalised_per_user(toy):
    users, _ = toy
    r = rank_within_user(np.random.default_rng(13).normal(size=len(users)), users)
    assert r.min() >= 0.0 and r.max() <= 1.0


def test_rank_average_is_scale_invariant(toy):
    """Ranks are the only defensible common currency across differently-scaled models."""
    users, _ = toy
    s = np.random.default_rng(14).normal(size=len(users))
    assert np.allclose(rank_average([s], users), rank_average([100 * s + 5], users))


def test_topk_selects_and_combines(toy):
    users, labels = toy
    rng = np.random.default_rng(15)
    cands = {f"m{i}": labels * (0.5 + 0.1 * i) + rng.normal(0, 1, len(labels))
             for i in range(5)}
    r = topk_selection(cands, users, labels, k=3)
    assert len(r.members) == 3
    assert len(r.scores) == len(labels)


def test_topk_clamps_to_available_candidates(toy):
    users, labels = toy
    r = topk_selection({"only": np.zeros(len(labels))}, users, labels, k=5)
    assert len(r.members) == 1


def test_argmax_comparison_reports_selection_bias(toy):
    users, labels = toy
    rng = np.random.default_rng(16)
    cands = {f"m{i}": labels * 0.5 + rng.normal(0, 1, len(labels)) for i in range(6)}
    r = compare_argmax_vs_topk(cands, users, labels, k=3, n_boot=200)
    assert r["argmax_model"] in cands
    assert "selection noise" in r["note"]


# --- degenerate-split guard ------------------------------------------------

def test_degenerate_split_is_detected():
    """A split with no discriminative users makes every comparison meaningless."""
    users = [f"u{i}" for i in range(10) for _ in range(4)]
    labels = np.array([0.0] * 20 + [1.0] * 20)   # 5 all-neg users, 5 all-pos
    d = diagnose_eval_split(users, labels)
    assert d["discriminative"] == 0
    assert not d["metric_can_move"]
    assert "DEGENERATE" in d["warning"]


def test_healthy_split_passes_the_guard(toy):
    users, labels = toy
    d = diagnose_eval_split(users, labels)
    assert d["metric_can_move"] and d["discriminative"] > 0
    assert d["warning"] is None


def test_repo_mini_data_valid_split_is_degenerate():
    """Documents a real defect in tests/_mini_data.py: its valid split cannot
    move the metric, so any test using it to assert score response is vacuous."""
    import sys, tempfile
    from pathlib import Path
    sys.path.insert(0, "tests")
    from _mini_data import make_mini_data
    from tippytop.data.dataset import load_dataset

    d = load_dataset(make_mini_data(Path(tempfile.mkdtemp()) / "data"))
    diag = diagnose_eval_split(d.users("valid"), d.y("valid"))
    assert diag["discriminative"] == 0, (
        "fixture changed — if it now has discriminative users, this guard test "
        "should be updated and the fixture defect considered fixed")


def test_float32_labels_do_not_trip_the_kit_check():
    """Regression: Dataset.y() is float32, and the kit accumulates its nDCG sum
    in float32, so it disagrees with a float64 decomposition at ~1e-8. A fixed
    1e-9 tolerance passed on float64 test fixtures and failed on every real
    dataset — found only on the first full-scale run."""
    rng = np.random.default_rng(21)
    users = [f"u{i}" for i in range(400) for _ in range(6)]
    labels32 = (rng.random(2400) > 0.5).astype(np.float32)
    scores32 = rng.normal(size=2400).astype(np.float32)
    verify_against_kit(users, labels32, scores32)          # must not raise
    verify_against_kit(users, labels32.astype(np.float64), scores32)


def test_tight_tolerance_still_available_for_float64():
    rng = np.random.default_rng(22)
    users = [f"u{i}" for i in range(200) for _ in range(5)]
    labels = (rng.random(1000) > 0.5).astype(np.float64)
    verify_against_kit(users, labels, rng.normal(size=1000), tol=1e-9)
