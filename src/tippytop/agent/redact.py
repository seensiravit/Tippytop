"""Redact hidden-test signal from anything that reaches the LLM's context.

The architecture already keeps test scores away from the model: the loop scores
the valid split only, and ``_finalize`` touches test exactly once after the loop
ends. One channel was left unsealed, though — ``stderr``.

The orchestrator captures a failing solution's stderr and embeds it verbatim in
the next DEBUG prompt so the model can fix its own bug. That is the right design,
but stderr is attacker-adjacent text: a generated solution that computed a test
score, printed it, and *then* crashed would put that number straight into the
model's context. The generated code is not adversarial, but the run log is a
graded artifact, and a judge who finds a test metric inside a prompt cannot tell
the difference between an accident and a cheat.

This module scrubs stderr before it is used. It is deliberately conservative:
false positives cost a few characters of debugging context, false negatives cost
the integrity of the submission.

Note this is defence in depth, not the wall itself. The wall is that the
harness never evaluates test during the loop.
"""
from __future__ import annotations

import re

REDACTION = "[redacted: possible hidden-test signal]"

# A metric name adjacent to the word "test", in either order, on one line.
# Catches "test primary 0.5946", "primary (test) = 0.59", "TEST GAUC: 0.66".
_METRIC = r"(?:primary|GAUC|nDCG(?:@\d+)?|ndcg(?:@\d+)?)"
_PATTERNS = [
    re.compile(rf"^.*\btest\b.*{_METRIC}.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(rf"^.*{_METRIC}.*\btest\b.*$", re.IGNORECASE | re.MULTILINE),
    # A dict-ish test block: {'test': {...}} or "test": {...}
    re.compile(r"['\"]test['\"]\s*:\s*\{[^}]*\}", re.IGNORECASE),
    # An explicit split=test evaluation line carrying any float.
    re.compile(r"^.*split\s*=\s*['\"]?test['\"]?.*\d\.\d+.*$", re.IGNORECASE | re.MULTILINE),
]


def scrub(text: str | None, *, limit: int | None = 4000) -> str:
    """Remove any line that pairs the test split with a metric or a score.

    ``limit`` also caps the length, since an unbounded stderr tail is both a
    prompt-cost problem and a place for signal to hide.
    """
    if not text:
        return ""
    for pat in _PATTERNS:
        text = pat.sub(REDACTION, text)
    if limit is not None and len(text) > limit:
        head = text[: limit // 2]
        tail = text[-limit // 2:]
        text = f"{head}\n[... {len(text) - limit} characters elided ...]\n{tail}"
    return text


def contains_test_signal(text: str | None) -> bool:
    """True if anything in ``text`` still looks like hidden-test signal.

    Used by the tests to assert that no prompt ever carries a test metric.
    """
    if not text:
        return False
    return any(pat.search(text) for pat in _PATTERNS)
