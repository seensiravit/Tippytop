"""Offline test of the Gemini response parser — no network."""
import pytest
from tippytop.agent.llm.gemini import _parse_gemini_response
from tippytop.agent.llm.base import LLMError

GOOD = {
    "candidates": [{
        "content": {"parts": [{"text": "Hypothesis: x\n```python\nprint(1)\n```"}]},
        "finishReason": "STOP",
    }],
    "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 40,
                      "totalTokenCount": 160},
}


def test_parses_text_and_tokens():
    r = _parse_gemini_response(GOOD)
    assert "print(1)" in r.text
    assert r.prompt_tokens == 120 and r.completion_tokens == 40 and r.total_tokens == 160


def test_blocked_response_raises():
    with pytest.raises(LLMError):
        _parse_gemini_response({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})


def test_empty_text_raises():
    bad = {"candidates": [{"content": {"parts": [{"text": ""}]},
                           "finishReason": "MAX_TOKENS"}]}
    with pytest.raises(LLMError):
        _parse_gemini_response(bad)
