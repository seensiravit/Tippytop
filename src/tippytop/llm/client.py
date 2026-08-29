"""Small OpenAI-compatible client for generated experiments and reflection."""

from __future__ import annotations

import time
from typing import Any

from ..config import RunConfig
from ..generated import GeneratedExperiment, executable_fingerprint, parse_json_object
from ..research import ResearchPlan
from .api_validation import validate_installed_api_calls
from .prompts import (
    generation_messages,
    reflection_messages,
    repair_messages,
    research_messages,
)
from .patches import apply_repair_payload
from .protocol import (
    GenerationFailure,
    LLMDeadlineExceeded,
    LLMResult,
    LLMTransportFailure,
)
from .semantic_validation import validate_prediction_paths
from .transport import OpenAITransport


REPAIR_MAX_TOKENS = 7000


class LLMClient:
    def __init__(self, config: RunConfig):
        self.transport = OpenAITransport(config)
        self.timeout = config.llm_timeout

    def list_models(self) -> list[str]:
        return self.transport.list_models()

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
        return self.transport.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            attempts=attempts,
            json_mode=json_mode,
            timeout=timeout,
        )

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
        failure_history: list[dict[str, Any]] | None = None,
        *,
        deadline: float | None = None,
    ) -> tuple[GeneratedExperiment, list[LLMResult]]:
        messages = repair_messages(context, failed, error, plan, failure_history)
        responses: list[LLMResult] = []
        last_error: Exception | None = None
        failed_fingerprint = executable_fingerprint(failed.source)
        for correction in range(3):
            try:
                result = self.complete(
                    messages,
                    temperature=0.0,
                    max_tokens=REPAIR_MAX_TOKENS,
                    attempts=1,
                    json_mode=True,
                    timeout=self._deadline_timeout(deadline),
                )
            except ConnectionError as request_error:
                raise LLMTransportFailure(str(request_error), responses) from request_error
            responses.append(result)
            try:
                repaired = apply_repair_payload(failed, parse_json_object(result.content))
                validate_installed_api_calls(repaired.source)
                validate_prediction_paths(repaired.source)
                if executable_fingerprint(repaired.source) == failed_fingerprint:
                    raise ValueError("repair did not change executable code")
                return repaired, responses
            except ValueError as repair_error:
                last_error = repair_error
                if correction < 2:
                    messages.extend(
                        [
                            {"role": "assistant", "content": result.content},
                            {
                                "role": "user",
                                "content": (
                                    f"The repair is still invalid: {repair_error}. Return a replacement "
                                    "edits array applied to the original failed source. Fix every listed "
                                    "API mismatch together using exact unique substrings."
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

    def _deadline_timeout(self, deadline: float | None) -> int:
        if deadline is None:
            return self.timeout
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise LLMDeadlineExceeded("run wall-clock limit reached")
        return max(1, min(self.timeout, remaining))
