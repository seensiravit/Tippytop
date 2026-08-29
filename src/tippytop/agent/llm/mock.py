"""A scripted, offline LLM client — drives the whole loop with zero network.

Give it either a list of assistant-message strings (returned in order, clamped at
the last) or a callable that receives (call_index, system, user) and returns the
message. The callable form lets a test branch on whether the prompt is a DEBUG
prompt (contains a traceback) so the mock can "fix" its own broken script.
"""
from __future__ import annotations
from typing import Callable, Sequence, Union

from .base import BaseLLMClient, LLMResponse

Script = Union[Sequence[str], Callable[[int, str, str], str]]


class MockLLMClient(BaseLLMClient):
    def __init__(self, scripts: Script, *, model: str = "mock",
                 tokens_per_call: tuple[int, int] = (1000, 1500)):
        self._scripts = scripts
        self.model = model
        self._pin, self._pout = tokens_per_call
        self.calls: list[dict] = []          # spy log: every (system, user) seen

    def generate(self, system: str, user: str, *, temperature: float = 0.4,
                 max_output_tokens: int = 8192) -> LLMResponse:
        idx = len(self.calls)
        self.calls.append({"system": system, "user": user})
        if callable(self._scripts):
            text = self._scripts(idx, system, user)
        else:
            seq = list(self._scripts)
            text = seq[idx] if idx < len(seq) else (seq[-1] if seq else "")
        return LLMResponse(text=text, prompt_tokens=self._pin,
                           completion_tokens=self._pout,
                           total_tokens=self._pin + self._pout,
                           raw={"mock": True, "call": idx})
