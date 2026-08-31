"""stderr must not carry hidden-test signal into the model's context.

The loop feeds a failing solution's stderr into the next DEBUG prompt so the
model can fix its own bug. That is the right design and it is also the one
unsealed channel: a generated solution that printed a test score and then
crashed would put that number in front of the model.
"""
from __future__ import annotations

import pytest

from tippytop.runlog.redact import scrub, contains_test_signal, REDACTION


@pytest.mark.parametrize("line", [
    "  test   GAUC 0.6610 | nDCG@5 0.5282 | primary 0.5946",
    "test primary 0.5946",
    "TEST GAUC: 0.6610",
    "primary (test) = 0.5946",
    "evaluated split='test' -> 0.5946",
    "{'test': {'GAUC': 0.661, 'primary': 0.5946}}",
])
def test_test_metrics_are_removed(line):
    out = scrub(f"Traceback (most recent call last):\n{line}\nValueError: boom")
    assert "0.5946" not in out or REDACTION in out
    assert not contains_test_signal(out)


def test_ordinary_traceback_survives():
    """Redaction must not destroy the debugging context the model needs."""
    err = ("Traceback (most recent call last):\n"
           '  File "solution.py", line 42, in main\n'
           "    scores = model.predict(data, args.split)\n"
           "AttributeError: 'NoneType' object has no attribute 'predict'")
    out = scrub(err)
    assert "AttributeError" in out
    assert "line 42" in out
    assert REDACTION not in out


def test_valid_metrics_are_not_redacted():
    """Validation feedback is the loop's whole signal — it must pass through."""
    out = scrub("valid GAUC 0.6674 | nDCG@5 0.5357 | primary 0.6016")
    assert "0.6016" in out
    assert REDACTION not in out


def test_empty_and_none_are_safe():
    assert scrub(None) == ""
    assert scrub("") == ""


def test_long_output_is_capped():
    out = scrub("x" * 20000, limit=1000)
    assert len(out) < 2000
    assert "elided" in out


def test_detector_agrees_with_scrubber():
    dirty = "test primary 0.5946"
    assert contains_test_signal(dirty)
    assert not contains_test_signal(scrub(dirty))
