"""Validated text edits for focused LLM-authored runtime repairs."""

from __future__ import annotations

from typing import Any

from ..generated import GeneratedExperiment


MAX_REPAIR_EDITS = 20


def apply_repair_payload(
    failed: GeneratedExperiment,
    payload: dict[str, Any],
) -> GeneratedExperiment:
    """Apply exact, unique LLM-authored replacements and revalidate the module."""

    if set(payload) != {"edits"}:
        raise ValueError("repair response must contain exactly one 'edits' field")
    edits = payload["edits"]
    if not isinstance(edits, list) or not edits or len(edits) > MAX_REPAIR_EDITS:
        raise ValueError(f"edits must be a non-empty list with at most {MAX_REPAIR_EDITS} items")

    source = failed.source
    for index, edit in enumerate(edits, start=1):
        if not isinstance(edit, dict) or set(edit) != {"old", "new"}:
            raise ValueError(f"edit {index} must contain exactly 'old' and 'new'")
        old = edit["old"]
        new = edit["new"]
        if not isinstance(old, str) or not old:
            raise ValueError(f"edit {index} old text must be non-empty")
        if not isinstance(new, str):
            raise ValueError(f"edit {index} new text must be a string")
        occurrences = source.count(old)
        if occurrences != 1:
            raise ValueError(
                f"edit {index} old text must occur exactly once, found {occurrences} occurrences"
            )
        source = source.replace(old, new, 1)

    return GeneratedExperiment(
        hypothesis=failed.hypothesis,
        expected_effect=failed.expected_effect,
        source=source,
    )
