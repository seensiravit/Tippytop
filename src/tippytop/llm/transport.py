"""Minimal OpenAI-compatible HTTP transport."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from ..config import RunConfig
from .protocol import LLMResult


class OpenAITransport:
    def __init__(self, config: RunConfig):
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.api_key = config.api_key
        self.timeout = config.llm_timeout

    def list_models(self) -> list[str]:
        request = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise ConnectionError(f"cannot query LLM models endpoint: {error}") from error
        return [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        attempts: int,
        json_mode: bool,
        timeout: int | None,
    ) -> LLMResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                payload = self._post("/chat/completions", body, timeout=timeout)
                choice = payload["choices"][0]
                raw_usage = payload.get("usage", {})
                return LLMResult(
                    str(choice["message"]["content"]),
                    {
                        "prompt_tokens": int(raw_usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(raw_usage.get("completion_tokens", 0)),
                        "total_tokens": int(raw_usage.get("total_tokens", 0)),
                    },
                    requested_model=self.model,
                    returned_model=str(payload.get("model", "")),
                    response_id=str(payload.get("id", "")),
                    finish_reason=str(choice.get("finish_reason", "")),
                )
            except (KeyError, IndexError, TypeError, ValueError, ConnectionError) as error:
                last_error = error
                if attempt + 1 < attempts:
                    time.sleep(2**attempt)
        raise ConnectionError(f"LLM request failed after {attempts} attempts: {last_error}")

    def _post(
        self,
        endpoint: str,
        body: dict[str, Any],
        *,
        timeout: int | None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            raise ConnectionError(f"LLM HTTP {error.code}: {detail}") from error
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise ConnectionError(f"LLM request failed: {error}") from error

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
