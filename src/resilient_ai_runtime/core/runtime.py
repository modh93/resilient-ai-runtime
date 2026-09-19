"""Application-facing orchestration for the Runtime v0.1 happy path."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from resilient_ai_runtime.providers import ProviderAdapter, create_provider_adapter
from resilient_ai_runtime.routing import CandidateRouter, ModelRegistry, PolicyResolver

from .config import (
    ModelConfig,
    ProviderConfig,
    RuntimeConfiguration,
    load_runtime_config,
)
from .cost import estimate_request_cost
from .errors import AllCandidatesFailedError, ConfigurationError, ProviderError
from .logging import emit_execution_record
from .request import GenerationRequest
from .response import AttemptRecord, GenerationResponse


class AIRuntime:
    """The provider-agnostic generation interface exposed to applications."""

    def __init__(
        self,
        configuration: RuntimeConfiguration,
        *,
        adapters: dict[str, ProviderAdapter] | None = None,
        adapter_factory: Callable[
            [ProviderConfig], ProviderAdapter
        ] = create_provider_adapter,
        execution_logger: Callable[[dict[str, Any]], None] = emit_execution_record,
    ) -> None:
        self._configuration = configuration
        self._adapters = dict(adapters or {})
        self._adapter_factory = adapter_factory
        self._execution_logger = execution_logger
        self._policy_resolver = PolicyResolver(configuration.policies)
        self._router = CandidateRouter(ModelRegistry(configuration.models))

    @classmethod
    def from_config(cls, path: str) -> AIRuntime:
        """Create a runtime from one validated YAML configuration file."""

        return cls(load_runtime_config(path))

    def generate(
        self,
        *,
        prompt: str,
        policy: str,
        system_prompt: str | None = None,
        constraints: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        """Generate text through the first candidate eligible under a policy."""

        request_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        started_at = perf_counter()
        attempts: list[AttemptRecord] = []
        try:
            request = GenerationRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                policy=policy,
                constraints=constraints,
                metadata=metadata,
            )
            effective_policy = self._policy_resolver.resolve(policy, constraints)
            candidates = self._router.get_candidates(
                policy=effective_policy,
                capability="text",
            )
            deadline = _deadline(started_at, effective_policy.max_latency_ms)

            for candidate_index, candidate in enumerate(candidates):
                adapter = self._adapter_for(candidate.provider)
                for retry_index in range(
                    self._configuration.runtime.retries_per_candidate + 1
                ):
                    timeout_s = _attempt_timeout(
                        _resolve_timeout(candidate, self._configuration), deadline
                    )
                    if timeout_s is None:
                        raise AllCandidatesFailedError(
                            attempts, "request latency budget exhausted"
                        )

                    attempt_started_at = perf_counter()
                    try:
                        result = adapter.generate(
                            model=candidate.model,
                            prompt=request.prompt,
                            system_prompt=request.system_prompt,
                            timeout_s=timeout_s,
                        )
                    except ProviderError as error:
                        attempts.append(
                            AttemptRecord(
                                provider=candidate.provider,
                                model=candidate.model,
                                status="failed",
                                latency_ms=_elapsed_ms(attempt_started_at),
                                error_code=_error_code(error),
                                zone=candidate.zone,
                            )
                        )
                        error.attempts = list(attempts)
                        if (
                            error.retryable
                            and retry_index
                            < self._configuration.runtime.retries_per_candidate
                        ):
                            continue
                        if (
                            effective_policy.fallback
                            and error.fallback_eligible
                            and candidate_index + 1 < len(candidates)
                        ):
                            break
                        if candidate_index + 1 == len(candidates):
                            raise AllCandidatesFailedError(attempts) from error
                        raise

                    latency_ms = _elapsed_ms(started_at)
                    attempts.append(
                        AttemptRecord(
                            provider=candidate.provider,
                            model=candidate.model,
                            status="success",
                            latency_ms=_elapsed_ms(attempt_started_at),
                            zone=candidate.zone,
                        )
                    )
                    estimated_cost_usd = estimate_request_cost(candidate, result)
                    response = GenerationResponse(
                        text=result.text,
                        provider=candidate.provider,
                        model=candidate.model,
                        latency_ms=latency_ms,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        estimated_cost_usd=estimated_cost_usd,
                        attempts=attempts,
                        request_id=request_id,
                    )
                    self._emit_record(
                        request_id=request_id,
                        timestamp=timestamp,
                        policy=policy,
                        response=response,
                        status="success",
                    )
                    return response

            raise AllCandidatesFailedError(attempts)
        except Exception as error:
            failure_attempts = getattr(error, "attempts", attempts)
            self._emit_record(
                request_id=request_id,
                timestamp=timestamp,
                policy=policy,
                attempts=failure_attempts,
                status="failed",
                error_code=_terminal_error_code(error, failure_attempts),
                latency_ms=_elapsed_ms(started_at),
            )
            raise

    def _adapter_for(self, provider_name: str) -> ProviderAdapter:
        if adapter := self._adapters.get(provider_name):
            return adapter
        try:
            provider_config = self._configuration.providers[provider_name]
        except KeyError as error:
            raise ConfigurationError(
                f"no configuration found for provider {provider_name!r}"
            ) from error
        adapter = self._adapter_factory(provider_config)
        self._adapters[provider_name] = adapter
        return adapter

    def _emit_record(
        self,
        *,
        request_id: str,
        timestamp: str,
        policy: str,
        status: str,
        response: GenerationResponse | None = None,
        attempts: list[AttemptRecord] | None = None,
        error_code: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        records = response.attempts if response is not None else attempts or []
        selected_attempt = records[-1] if records else None
        record: dict[str, Any] = {
            "request_id": request_id,
            "timestamp": timestamp,
            "policy": policy,
            "selected_provider": response.provider
            if response
            else _attempt_value(selected_attempt, "provider"),
            "selected_model": response.model
            if response
            else _attempt_value(selected_attempt, "model"),
            "attempt_count": len(records),
            "latency_ms": response.latency_ms if response else latency_ms,
            "input_tokens": response.input_tokens if response else None,
            "output_tokens": response.output_tokens if response else None,
            "estimated_cost_usd": response.estimated_cost_usd if response else None,
            "status": status,
            "attempts": [_attempt_record(attempt) for attempt in records],
        }
        if error_code:
            record["error_code"] = error_code
        try:
            self._execution_logger(record)
        except Exception:
            return


def _deadline(started_at: float, max_latency_ms: int | None) -> float | None:
    """Return an absolute deadline for one request's global latency budget."""

    if max_latency_ms is None:
        return None
    return started_at + max_latency_ms / 1_000


def _attempt_timeout(runtime_timeout_s: float, deadline: float | None) -> float | None:
    """Return the remaining global budget capped by the runtime attempt bound."""

    if deadline is None:
        return runtime_timeout_s
    remaining_s = deadline - perf_counter()
    if remaining_s <= 0:
        return None
    return min(runtime_timeout_s, remaining_s)


def _resolve_timeout(
    candidate: ModelConfig, configuration: RuntimeConfiguration
) -> float:
    """Resolve the per-attempt timeout while preserving policy budgets.

    ``_attempt_timeout`` subsequently caps this value by the policy's remaining
    global latency budget, when one is configured.
    """

    if candidate.timeout_s is not None:
        return candidate.timeout_s
    provider_timeout_s = configuration.providers[candidate.provider].timeout_s
    if provider_timeout_s is not None:
        return provider_timeout_s
    return configuration.runtime.timeout_s


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1_000)


def _error_code(error: ProviderError) -> str:
    return {
        "ProviderAuthenticationError": "provider_authentication",
        "ProviderRateLimitError": "provider_rate_limit",
        "ProviderRequestError": "provider_request",
        "ProviderTimeoutError": "provider_timeout",
        "ProviderUnavailableError": "provider_unavailable",
    }.get(type(error).__name__, "provider_error")


def _terminal_error_code(error: Exception, attempts: list[AttemptRecord]) -> str | None:
    if isinstance(error, ProviderError):
        return _error_code(error)
    if attempts:
        return attempts[-1].error_code
    return None


def _attempt_value(attempt: AttemptRecord | None, field_name: str) -> str | None:
    return getattr(attempt, field_name) if attempt is not None else None


def _attempt_record(attempt: AttemptRecord) -> dict[str, Any]:
    return {
        "provider": attempt.provider,
        "model": attempt.model,
        "status": attempt.status,
        "latency_ms": attempt.latency_ms,
        "error_code": attempt.error_code,
        "zone": attempt.zone,
    }
