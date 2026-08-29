"""Small OpenAI-compatible client for generated experiments and reflection."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import RunConfig
from .generated import GeneratedExperiment, executable_fingerprint, parse_json_object
from .llm_prompts import (
    generation_messages,
    reflection_messages,
    repair_messages,
    research_messages,
    review_messages,
)
from .llm_protocol import (
    REVIEW_CHECKS,
    ExperimentReview,
    GenerationFailure,
    LLMDeadlineExceeded,
    LLMResult,
    LLMTransportFailure,
    parse_review,
)
from .research_plan import ResearchPlan


class LLMClient:
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
        temperature: float = 0.2,
        max_tokens: int = 1200,
        attempts: int = 3,
        json_mode: bool = False,
        timeout: int | None = None,
    ) -> LLMResult:
        body = {
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
                content = choice["message"]["content"]
                raw_usage = payload.get("usage", {})
                usage = {
                    "prompt_tokens": int(raw_usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(raw_usage.get("completion_tokens", 0)),
                    "total_tokens": int(raw_usage.get("total_tokens", 0)),
                }
                return LLMResult(
                    str(content),
                    usage,
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

    def generate(
        self,
        context: dict[str, Any],
        plan: ResearchPlan,
        *,
        deadline: float | None = None,
    ) -> tuple[GeneratedExperiment, list[LLMResult]]:
        messages = generation_messages(context, plan)
        responses: list[LLMResult] = []
        correction_messages = list(messages)
        last_error: Exception | None = None
        for correction in range(3):
            try:
                result = self.complete(
                    correction_messages,
                    temperature=0.2 if correction == 0 else 0.0,
                    max_tokens=6500,
                    attempts=1,
                    json_mode=True,
                    timeout=self._deadline_timeout(deadline),
                )
            except ConnectionError as error:
                raise LLMTransportFailure(str(error), responses) from error
            responses.append(result)
            try:
                return GeneratedExperiment.from_dict(parse_json_object(result.content)), responses
            except ValueError as error:
                last_error = error
                correction_messages.extend(
                    [
                        {"role": "assistant", "content": result.content},
                        {
                            "role": "user",
                            "content": (
                                f"The response violates the experiment contract: {error}. "
                                "Return one corrected JSON object only, with the complete module "
                                "and both top-level entry points."
                            ),
                        },
                    ]
                )
        raise GenerationFailure(str(last_error), responses) from last_error

    def research(
        self,
        context: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> tuple[ResearchPlan, list[LLMResult]]:
        """Choose one substantive scientific direction before any code is written."""

        messages = research_messages(context)
        responses: list[LLMResult] = []
        last_error: Exception | None = None
        for correction in range(2):
            try:
                result = self.complete(
                    messages,
                    temperature=0.65 if correction == 0 else 0.1,
                    max_tokens=2400,
                    attempts=1,
                    json_mode=True,
                    timeout=self._deadline_timeout(deadline),
                )
            except ConnectionError as error:
                raise LLMTransportFailure(str(error), responses) from error
            responses.append(result)
            try:
                return ResearchPlan.from_dict(parse_json_object(result.content)), responses
            except ValueError as error:
                last_error = error
                if correction == 0:
                    messages.extend(
                        [
                            {"role": "assistant", "content": result.content},
                            {
                                "role": "user",
                                "content": (
                                    f"The plan violates the schema: {error}. Return one corrected, "
                                    "complete research plan JSON object and no code."
                                ),
                            },
                        ]
                    )
        raise GenerationFailure(str(last_error), responses) from last_error

    def repair(
        self,
        context: dict[str, Any],
        failed: GeneratedExperiment,
        error: str,
        plan: ResearchPlan | None = None,
        *,
        deadline: float | None = None,
    ) -> tuple[GeneratedExperiment, list[LLMResult]]:
        messages = repair_messages(context, failed, error, plan)
        responses: list[LLMResult] = []
        last_error: Exception | None = None
        failed_fingerprint = executable_fingerprint(failed.source)
        for correction in range(2):
            try:
                result = self.complete(
                    messages,
                    temperature=0.0,
                    max_tokens=6500,
                    attempts=1,
                    json_mode=True,
                    timeout=self._deadline_timeout(deadline),
                )
            except ConnectionError as request_error:
                raise LLMTransportFailure(str(request_error), responses) from request_error
            responses.append(result)
            try:
                repaired = GeneratedExperiment.from_dict(parse_json_object(result.content))
                if executable_fingerprint(repaired.source) == failed_fingerprint:
                    raise ValueError("repair did not change executable code")
                return repaired, responses
            except ValueError as repair_error:
                last_error = repair_error
                if correction == 0:
                    messages.extend(
                        [
                            {"role": "assistant", "content": result.content},
                            {
                                "role": "user",
                                "content": (
                                    f"The repair is still invalid: {repair_error}. Return a complete "
                                    "corrected module that changes the failing executable statements."
                                ),
                            },
                        ]
                    )
        raise GenerationFailure(str(last_error), responses) from last_error

    def review(
        self,
        context: dict[str, Any],
        plan: ResearchPlan,
        proposed: GeneratedExperiment,
        *,
        deadline: float | None = None,
    ) -> tuple[ExperimentReview, list[LLMResult]]:
        """Critique and, when needed, rewrite generated source before execution."""

        messages = review_messages(context, plan, proposed)
        responses: list[LLMResult] = []
        last_error: Exception | None = None
        for correction in range(2):
            try:
                result = self.complete(
                    messages,
                    temperature=0.1 if correction == 0 else 0.0,
                    max_tokens=6500,
                    attempts=1,
                    json_mode=True,
                    timeout=self._deadline_timeout(deadline),
                )
            except ConnectionError as request_error:
                raise LLMTransportFailure(str(request_error), responses) from request_error
            responses.append(result)
            try:
                review = parse_review(result.content)
                changed = (
                    executable_fingerprint(review.experiment.source)
                    != executable_fingerprint(proposed.source)
                )
                if review.verdict == "revise" and not changed:
                    raise ValueError("review requested revision but did not change executable code")
                if review.verdict == "pass" and changed:
                    raise ValueError("review changed executable code but used a pass verdict")
                return review, responses
            except ValueError as review_error:
                last_error = review_error
                if correction == 0:
                    messages.extend(
                        [
                            {"role": "assistant", "content": result.content},
                            {
                                "role": "user",
                                "content": (
                                    f"The review response is invalid: {review_error}. Return every "
                                    "required field and all six checks. A revise verdict must include a substantively "
                                    "corrected complete module; pass must preserve executable behavior."
                                ),
                            },
                        ]
                    )
        raise GenerationFailure(str(last_error), responses) from last_error

    def reflect(self, context: dict[str, Any], *, deadline: float | None = None) -> LLMResult:
        return self.complete(
            reflection_messages(context),
            temperature=0.2,
            max_tokens=300,
            attempts=1,
            timeout=self._deadline_timeout(deadline),
        )

    def _post(
        self,
        endpoint: str,
        body: dict[str, Any],
        *,
        timeout: int | None = None,
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

    def _deadline_timeout(self, deadline: float | None) -> int:
        if deadline is None:
            return self.timeout
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise LLMDeadlineExceeded("run wall-clock limit reached")
        return max(1, min(self.timeout, remaining))
