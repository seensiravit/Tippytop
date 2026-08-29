"""Human-readable final run report rendering."""

from __future__ import annotations

from typing import Any


OFFICIAL_VALID_PRIMARY = 0.6016
OFFICIAL_TEST_PRIMARY = 0.5946
OFFICIAL_TEST_STD = 0.0008
ORACLE_TEST_PRIMARY = 0.8645


def render_report(results: dict[str, Any]) -> str:
    baseline = results.get("baseline_valid") or {}
    valid = results["best_validation"]
    test = results["test"]
    reproduced_delta = float(valid["primary"]) - float(baseline.get("primary", OFFICIAL_VALID_PRIMARY))
    official_valid_delta = float(valid["primary"]) - OFFICIAL_VALID_PRIMARY
    official_test_delta = float(test["primary"]) - OFFICIAL_TEST_PRIMARY
    oracle_gap = ORACLE_TEST_PRIMARY - float(test["primary"])
    return f"""# Tippytop Run Report

## Outcome

| Result | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Reproduced FM validation | {baseline.get('GAUC', 0):.4f} | {baseline.get('nDCG@5', 0):.4f} | {baseline.get('primary', 0):.4f} |
| Best validation | {valid['GAUC']:.4f} | {valid['nDCG@5']:.4f} | {valid['primary']:.4f} |
| One-time test | {test['GAUC']:.4f} | {test['nDCG@5']:.4f} | {test['primary']:.4f} |

Validation delta over reproduced FM: **{reproduced_delta:+.4f}**

Validation delta over official FM ({OFFICIAL_VALID_PRIMARY:.4f}): **{official_valid_delta:+.4f}**

Test delta over published FM mean ({OFFICIAL_TEST_PRIMARY:.4f} ± {OFFICIAL_TEST_STD:.4f}): **{official_test_delta:+.4f}**

Gap to oracle ({ORACLE_TEST_PRIMARY:.4f}): **{oracle_gap:.4f}**

## Run Summary

- Best experiment: `{results['best_experiment']}`
- Stopping reason: `{results['stopping_reason']}`
- Iterations: {results['iterations']}
- Agent wall-clock: {results['elapsed_seconds']:.1f} seconds
- Search wall-clock: {results['search_elapsed_seconds']:.1f} seconds
- Finalization wall-clock: {results['finalization_elapsed_seconds']:.1f} seconds
- LLM tokens: {results['llm_usage'].get('total_tokens', 0)}
- GPU-hours: 0
- Manual interventions: {results['manual_interventions']}
- Submission: `{results['submission']}`

See `iterations.jsonl` and `events.jsonl` for the hypothesis, configuration diff, metrics, and
recovery record for every iteration.
"""
