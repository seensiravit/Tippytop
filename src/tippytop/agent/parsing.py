"""Parse an LLM reply into (hypothesis, code)."""
from __future__ import annotations
from dataclasses import dataclass
import re

# ```python ... ```  (also tolerate ```py and a bare ``` fence)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ParsedProposal:
    hypothesis: str
    code: str
    parse_ok: bool
    parse_error: str | None = None


def parse_response(text: str) -> ParsedProposal:
    """Hypothesis = text before the first fence; code = the LAST fenced block.

    Using the last fence means an illustrative snippet earlier in the reply never
    wins over the final full solution.
    """
    matches = list(_FENCE.finditer(text or ""))
    if not matches:
        return ParsedProposal(hypothesis=(text or "").strip(), code="",
                              parse_ok=False,
                              parse_error="no ```python code fence found")
    code = matches[-1].group(1).strip()
    head = text[: matches[0].start()].strip()
    hypothesis = _clean_hypothesis(head)
    if not code:
        return ParsedProposal(hypothesis=hypothesis, code="", parse_ok=False,
                              parse_error="empty code fence")
    return ParsedProposal(hypothesis=hypothesis, code=code, parse_ok=True)


def _clean_hypothesis(head: str) -> str:
    # Drop a leading "Hypothesis:" label if present; keep it a single tidy line-ish.
    m = re.match(r"(?is)^\s*hypothesis\s*:?\s*(.*)$", head)
    if m:
        head = m.group(1).strip()
    return head or "(no hypothesis given)"
