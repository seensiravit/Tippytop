"""Validation-to-test transfer — the gating measurement nobody has taken.

The whole workflow optimizes validation primary. The grade is the *test* delta,
scored once. The relationship between the two has never been measured, and every
plan in this project silently assumes it is 1:1.

The available evidence does not support that assumption — nor its opposite. The
one observation on record is a seed-averaged ensemble gaining +0.0011 on
validation and +0.0003 on test, a ratio of 0.27. But test sigma is 0.0008, so a
+0.0003 test movement is entirely inside noise. Transfer is therefore **neither
1:1 nor 0.27x — it is unmeasured**, and that is the largest single risk to every
target anyone has set.

This module measures it from models that have *already been trained*, so it costs
no compute at all. If the slope comes back reliably below 1, every validation
target in the project needs dividing by it before it means anything.

    slope, and what it implies for a plan that targets +0.0031 on validation:

        slope 1.00  ->  +0.0031 on test   clears the 0.002 bar
        slope 0.50  ->  +0.0016 on test   misses
        slope 0.27  ->  +0.0008 on test   misses badly
"""
from __future__ import annotations

import numpy as np


def transfer_slope(valid: np.ndarray | list, test: np.ndarray | list, *,
                   baseline_valid: float | None = None,
                   baseline_test: float | None = None,
                   n_boot: int = 5000, seed: int = 0,
                   alpha: float = 0.05) -> dict:
    """Regress test gain on validation gain across already-trained models.

    Pass the raw primary scores of every model you have. Gains are taken relative
    to ``baseline_valid`` / ``baseline_test`` when given, otherwise relative to
    the weakest model in each list.

    The CI comes from bootstrapping over *models*, which is the right resampling
    unit: the uncertainty being quantified is "would this slope hold for the next
    model we train", not "for the next user we score".
    """
    v = np.asarray(valid, dtype=np.float64)
    t = np.asarray(test, dtype=np.float64)
    if v.shape != t.shape:
        raise ValueError(f"valid {v.shape} and test {t.shape} must have equal length")
    if len(v) < 3:
        raise ValueError(f"need at least 3 models to fit a slope; got {len(v)}")

    bv = float(v.min()) if baseline_valid is None else baseline_valid
    bt = float(t.min()) if baseline_test is None else baseline_test
    gv, gt = v - bv, t - bt

    slope, intercept = np.polyfit(gv, gt, 1)

    rng = np.random.default_rng(seed)
    n = len(gv)
    slopes = np.empty(n_boot)
    ok = 0
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # A resample with no spread in x has no defined slope; skip it.
        if np.ptp(gv[idx]) < 1e-12:
            continue
        slopes[ok] = np.polyfit(gv[idx], gt[idx], 1)[0]
        ok += 1
    slopes = slopes[:ok]

    lo, hi = (np.percentile(slopes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if ok > 10 else (float("nan"), float("nan")))

    r = float(np.corrcoef(gv, gt)[0, 1]) if np.ptp(gv) > 0 and np.ptp(gt) > 0 else 0.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "r": r,
        "r_squared": r ** 2,
        "n_models": n,
        "baseline_valid": bv,
        "baseline_test": bt,
        "reliable": bool(n >= 5 and ok > 10 and not np.isnan(lo) and (hi - lo) < 1.0),
        "note": _interpret(slope, lo, hi, n),
    }


def _interpret(slope: float, lo: float, hi: float, n: int) -> str:
    if n < 5:
        return (f"only {n} models — the slope is not yet trustworthy. Train a few "
                "more variants before acting on it.")
    if np.isnan(lo):
        return "bootstrap failed to resolve a slope; too little spread in validation gains."
    if hi < 1.0:
        return (f"slope is reliably below 1 (CI upper bound {hi:.2f}) — validation "
                "gains shrink on test. Divide every validation target by this slope.")
    if lo > 1.0:
        return f"slope is reliably above 1 (CI lower bound {lo:.2f}) — unusual; check for leakage."
    return ("CI spans 1.0, so 1:1 transfer cannot be ruled out — but neither can "
            "substantial shrinkage. Treat validation targets as optimistic.")


def rescale_target(valid_target: float, slope: float) -> float:
    """Expected test gain for a plan that achieves ``valid_target`` on validation."""
    return valid_target * slope


def required_valid_gain(test_target: float, slope: float) -> float:
    """Validation gain needed to land ``test_target`` on test.

    This is the number a plan should actually be built around once the slope is
    known — targeting the test bar directly, rather than hoping validation
    transfers.
    """
    if slope <= 0:
        raise ValueError("slope must be positive to invert the target")
    return test_target / slope
