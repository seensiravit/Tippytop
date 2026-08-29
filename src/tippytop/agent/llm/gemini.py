"""Gemini client over the Google AI Studio REST API — stdlib only (no SDK).

Python 3.14 wheels for the google SDKs may be unavailable, so we speak the REST
API directly with urllib. The API key comes from the GEMINI_API_KEY env var.
"""
from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request

from .base import BaseLLMClient, LLMResponse, LLMError

_ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "{model}:generateContent?key={key}")
_RETRY_STATUS = {429, 500, 502, 503}


def _parse_gemini_response(js: dict) -> LLMResponse:
    """Extract text + token usage from a generateContent response."""
    cands = js.get("candidates") or []
    if not cands:
        reason = (js.get("promptFeedback") or {}).get("blockReason")
        raise LLMError(f"no candidates returned (blockReason={reason})")
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    if not text.strip():
        raise LLMError(f"empty text (finishReason={cands[0].get('finishReason')})")
    um = js.get("usageMetadata", {})
    return LLMResponse(
        text=text,
        prompt_tokens=um.get("promptTokenCount", 0),
        completion_tokens=um.get("candidatesTokenCount", 0),
        total_tokens=um.get("totalTokenCount", 0),
        raw=js,
    )


class GeminiClient(BaseLLMClient):
    def __init__(self, model: str = "gemini-2.5-flash-lite", *,
                 api_key: str | None = None, max_retries: int = 3,
                 timeout_s: int = 120):
        self.model = model
        self._key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._key:
            raise LLMError("GEMINI_API_KEY is not set (export it before --llm gemini)")
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    def _post(self, body: dict) -> dict:
        url = _ENDPOINT.format(model=self.model, key=self._key)
        data = json.dumps(body).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body_txt = e.read().decode("utf-8", "replace")[:500]
                if e.code in _RETRY_STATUS and attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    last_err = LLMError(f"HTTP {e.code}: {body_txt}")
                    continue
                raise LLMError(f"HTTP {e.code}: {body_txt}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    last_err = e
                    continue
                raise LLMError(f"network error: {e}") from e
        raise LLMError(f"exhausted retries: {last_err}")

    def generate(self, system: str, user: str, *, temperature: float = 0.4,
                 max_output_tokens: int = 8192) -> LLMResponse:
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_output_tokens},
        }
        return _parse_gemini_response(self._post(body))
