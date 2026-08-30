"""Power and selection-noise calculations — decide whether an experiment is worth running.

Two questions this answers before any compute is spent:

1. **How many seeds does this comparison need?** An experiment with too few seeds
   to resolve its own effect is wasted budget, and worse, it produces a number
   that looks like evidence.

2. **How much of an apparent gain is selection artifact?** Picking the best of N
   candidates on a noisy score inflates the winner even when every candidate is
   truly identical. On this benchmark that inflation is comparable in size to
   every real effect measured so far, which makes it the dominant failure mode.

A correction worth stating, because it bit this project's own analysis
--------------------------------------------------------------------
The familiar approximation for the expected maximum of N draws is
sigma*sqrt(2 ln N). That is the **asymptotic leading term** and it overshoots
badly at the small N that actually applies here — by 24% at N = 50 and by 75% at
N = 3. This module computes the expected maximum by numerical integration
instead, so the number it reports is the real one.
"""
from __future__ import annotations

import math
from statistics import NormalDist

_ND = NormalDist()

#: Measured on the FM baseline over 10 seeds. Other model classes may differ —
#: measure rather than assume (see ``seeds_needed`` docstring).
DEFAULT_SIGMA_VALID = 0.0006
DEFAULT_SIGMA_TEST = 0.0008

#: The organizers' convergence threshold, and the bar a claim must clear.
EPSILON = 0.002


def seeds_needed(delta: float, sigma: float = DEFAULT_SIGMA_VALID,
                 power: float = 0.80, alpha: float = 0.05) -> float:
    """Seeds per arm to detect an effect of ``delta`` at the given power.

    Standard two-sample normal formula: n = 2 (z_{1-a/2} + z_{power})^2 sigma^2 / delta^2.

    ``sigma`` should be the seed standard deviation *for the model class being
    tested*. The 0.0006 default was measured on the FM baseline; a GBDT or an
    ensemble may be more or less stable, and assuming homoscedasticity across
    model classes is exactly the kind of unexamined assumption this module exists
    to discourage.
    """
    if delta <= 0:
        raise ValueError("delta must be positive; pass the absolute effect size")
    z_a = _ND.inv_cdf(1 - alpha / 2)
    z_b = _ND.inv_cdf(power)
    return 2.0 * (z_a + z_b) ** 2 * sigma ** 2 / delta ** 2


def is_detectable(delta: float, sigma: float = DEFAULT_SIGMA_VALID,
                  max_seeds: int = 10, **kw) -> dict:
    """Guard: can this effect be resolved within a feasible seed budget?

    Use before committing an iteration. An effect needing more seeds than the
    budget allows is not a small result — it is an unmeasurable one, and running
    it produces a number that cannot be interpreted either way.
    """
    n = seeds_needed(delta, sigma, **kw)
    return {
        "delta": delta,
        "seeds_needed": n,
        "max_seeds": max_seeds,
        "detectable": bool(n <= max_seeds),
        "verdict": ("run it" if n <= max_seeds else
                    f"NOT detectable within {max_seeds} seeds — needs {n:.0f}"),
    }


def expected_max(n: int, sigma: float = 1.0, _steps: int = 200_000) -> float:
    """E[max of n iid N(0, sigma)], computed exactly rather than asymptotically.

    Uses the survival-function form
    ``E[max] = INT_0^inf (1 - F(x)^n) dx - INT_-inf^0 F(x)^n dx``,
    which is stable and avoids the overshoot of the sqrt(2 ln n) approximation.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return 0.0
    lo, hi = -6.0, 8.0
    h = (hi - lo) / _steps
    total = 0.0
    for i in range(_steps):
        x = lo + (i + 0.5) * h
        f_n = _ND.cdf(x) ** n
        total += (1.0 - f_n) * h if x > 0 else -f_n * h
    return sigma * total


def expected_max_asymptotic(n: int, sigma: float = 1.0) -> float:
    """The sqrt(2 ln n) approximation — provided only for comparison.

    Kept so ``selection_inflation`` can report how badly the familiar shortcut
    overstates the real value at the small N that applies here.
    """
    return sigma * math.sqrt(2 * math.log(n)) if n > 1 else 0.0


def selection_inflation(n_candidates: int, sigma: float = DEFAULT_SIGMA_VALID,
                        seeds_per_candidate: int = 1) -> dict:
    """Apparent gain produced purely by picking the best of N noisy candidates.

    Models the worst case honestly: every candidate is *truly identical* to the
    incumbent, so any reported improvement is entirely selection artifact.
    Averaging over ``seeds_per_candidate`` shrinks the standard error by sqrt(m)
    and the inflation with it.
    """
    se = sigma / math.sqrt(max(seeds_per_candidate, 1))
    exact = expected_max(n_candidates, se)
    naive = expected_max_asymptotic(n_candidates, se)
    return {
        "n_candidates": n_candidates,
        "seeds_per_candidate": seeds_per_candidate,
        "se": se,
        "expected_inflation": exact,
        "asymptotic_overstatement_pct": (100 * (naive / exact - 1)) if exact > 0 else 0.0,
        "share_of_epsilon": exact / EPSILON,
        "warning": (
            f"best-of-{n_candidates} at {seeds_per_candidate} seed(s) manufactures "
            f"{exact:.5f} ({100 * exact / EPSILON:.0f}% of the {EPSILON} bar) from noise alone"
        ),
    }


def recommend_seeds(delta: float, n_candidates: int = 50,
                    sigma: float = DEFAULT_SIGMA_VALID,
                    max_seeds: int = 5) -> dict:
    """Smallest seed count that both resolves ``delta`` and tames selection noise.

    Two constraints, and the binding one is usually the second: enough power to
    detect the effect, *and* enough averaging that best-of-N inflation stays well
    under the effect being claimed.
    """
    for m in range(1, max_seeds + 1):
        power_ok = seeds_needed(delta, sigma) <= m
        infl = selection_inflation(n_candidates, sigma, m)["expected_inflation"]
        if power_ok and infl < delta / 2:
            return {"seeds": m, "reason": "power and selection noise both satisfied",
                    "inflation": infl, "delta": delta}
    infl = selection_inflation(n_candidates, sigma, max_seeds)["expected_inflation"]
    return {
        "seeds": max_seeds,
        "reason": f"capped at {max_seeds}; selection noise ({infl:.5f}) is still "
                  f"large relative to delta ({delta:.5f}) — treat results as provisional",
        "inflation": infl,
        "delta": delta,
    }
