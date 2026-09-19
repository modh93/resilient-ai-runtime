"""End-to-end unit tests for the Runtime v0.1 happy path."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest

from resilient_ai_runtime import AIRuntime
from resilient_ai_runtime.core import (
    AllCandidatesFailedError,
    ModelConfig,
    NoEligibleModelError,
    PolicyConfig,
    PolicyViolationError,
    ProviderConfig,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResult,
    ProviderTimeoutError,
    ProviderUnavailableError,
    PricingConfig,
    RuntimeConfiguration,
    RuntimeDefaults,
)


@dataclass
class _FakeAdapter:
    name: str
    result: ProviderResult = field(
        default_factory=lambda: ProviderResult(
            text="Generated text", input_tokens=12, output_tokens=4
        )
    )
    calls: list[dict[str, object]] = field(default_factory=list)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: str | None,
        timeout_s: float,
    ) -> ProviderResult:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "timeout_s": timeout_s,
            }
        )
        return self.result


@dataclass
class _SequenceAdapter:
    name: str
    outcomes: list[ProviderResult | Exception]
    calls: list[dict[str, object]] = field(default_factory=list)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: str | None,
        timeout_s: float,
    ) -> ProviderResult:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "timeout_s": timeout_s,
            }
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def configuration() -> RuntimeConfiguration:
    return RuntimeConfiguration(
        runtime=RuntimeDefaults(timeout_s=30, retries_per_candidate=1),
        providers={
            "ollama": ProviderConfig(name="ollama"),
            "mistral": ProviderConfig(name="mistral", credential_env="MISTRAL_API_KEY"),
        },
        models={
            "local-default": ModelConfig(
                id="local-default",
                provider="ollama",
                model="qwen3:8b",
                zone="local",
                enabled=True,
                capabilities=("text",),
            ),
            "mistral-default": ModelConfig(
                id="mistral-default",
                provider="mistral",
                model="mistral-small-latest",
                zone="eu-cloud",
                enabled=True,
                capabilities=("text",),
            ),
            "vision-only": ModelConfig(
                id="vision-only",
                provider="ollama",
                model="vision-model",
                zone="local",
                enabled=True,
                capabilities=("vision",),
            ),
        },
        policies={
            "local": PolicyConfig(
                name="local",
                model_priority=("local-default", "mistral-default"),
                fallback=True,
                allowed_zones=("local",),
            ),
            "cloud": PolicyConfig(
                name="cloud",
                model_priority=("mistral-default", "local-default"),
                fallback=True,
                allowed_zones=("eu-cloud",),
            ),
            "fast": PolicyConfig(
                name="fast",
                model_priority=("local-default",),
                fallback=True,
                max_latency_ms=5_000,
            ),
            "runtime-bound": PolicyConfig(
                name="runtime-bound",
                model_priority=("local-default",),
                fallback=True,
                max_latency_ms=60_000,
            ),
            "resilient": PolicyConfig(
                name="resilient",
                model_priority=("local-default", "mistral-default"),
                fallback=True,
            ),
            "no-fallback": PolicyConfig(
                name="no-fallback",
                model_priority=("local-default", "mistral-default"),
                fallback=False,
            ),
            "budgeted": PolicyConfig(
                name="budgeted",
                model_priority=("local-default", "mistral-default"),
                fallback=True,
                max_latency_ms=10_000,
            ),
            "short-budget": PolicyConfig(
                name="short-budget",
                model_priority=("local-default", "mistral-default"),
                fallback=True,
                max_latency_ms=1_000,
            ),
            "zero-latency": PolicyConfig(
                name="zero-latency",
                model_priority=("local-default",),
                fallback=True,
                max_latency_ms=0,
            ),
            "none": PolicyConfig(
                name="none",
                model_priority=("vision-only",),
                fallback=True,
            ),
        },
    )


def test_runtime_from_config_loads_a_usable_runtime() -> None:
    config_path = Path(__file__).parents[2] / "examples" / "runtime.example.yaml"

    runtime = AIRuntime.from_config(str(config_path))

    assert isinstance(runtime, AIRuntime)


def test_runtime_executes_local_candidate_and_normalizes_response(
    configuration: RuntimeConfiguration,
) -> None:
    local_adapter = _FakeAdapter(name="ollama")
    runtime = AIRuntime(configuration, adapters={"ollama": local_adapter})

    response = runtime.generate(
        prompt="Summarize this.", policy="local", system_prompt="Be concise."
    )

    assert local_adapter.calls == [
        {
            "model": "qwen3:8b",
            "prompt": "Summarize this.",
            "system_prompt": "Be concise.",
            "timeout_s": 30,
        }
    ]
    assert response.text == "Generated text"
    assert response.provider == "ollama"
    assert response.model == "qwen3:8b"
    assert response.input_tokens == 12
    assert response.output_tokens == 4
    assert response.estimated_cost_usd is None
    assert response.attempts[0].status == "success"
    assert response.attempts[0].zone == "local"
    UUID(response.request_id)


def test_runtime_uses_same_public_api_for_cloud_candidate(
    configuration: RuntimeConfiguration,
) -> None:
    cloud_adapter = _FakeAdapter(name="mistral")
    runtime = AIRuntime(configuration, adapters={"mistral": cloud_adapter})

    response = runtime.generate(prompt="Summarize this.", policy="cloud")

    assert cloud_adapter.calls[0]["model"] == "mistral-small-latest"
    assert response.provider == "mistral"
    assert response.model == "mistral-small-latest"


def test_runtime_rejects_unknown_policy(configuration: RuntimeConfiguration) -> None:
    runtime = AIRuntime(configuration)

    with pytest.raises(PolicyViolationError, match="unknown policy"):
        runtime.generate(prompt="Hello", policy="unknown")


def test_runtime_propagates_no_eligible_model(
    configuration: RuntimeConfiguration,
) -> None:
    runtime = AIRuntime(configuration)

    with pytest.raises(NoEligibleModelError, match="no eligible model"):
        runtime.generate(prompt="Hello", policy="none")


def test_policy_latency_limit_tightens_adapter_timeout(
    configuration: RuntimeConfiguration,
) -> None:
    adapter = _FakeAdapter(name="ollama")
    runtime = AIRuntime(configuration, adapters={"ollama": adapter})

    runtime.generate(prompt="Hello", policy="fast")

    assert 0 < adapter.calls[0]["timeout_s"] <= 5


def test_runtime_timeout_remains_bound_when_stricter_than_policy(
    configuration: RuntimeConfiguration,
) -> None:
    adapter = _FakeAdapter(name="ollama")
    runtime = AIRuntime(configuration, adapters={"ollama": adapter})

    runtime.generate(prompt="Hello", policy="runtime-bound")

    assert adapter.calls[0]["timeout_s"] == 30


def test_model_timeout_override_wins_over_provider_and_runtime_defaults(
    configuration: RuntimeConfiguration,
) -> None:
    configuration = replace(
        configuration,
        providers={
            **configuration.providers,
            "ollama": replace(configuration.providers["ollama"], timeout_s=60),
        },
        models={
            **configuration.models,
            "local-default": replace(
                configuration.models["local-default"], timeout_s=45
            ),
        },
    )
    adapter = _FakeAdapter(name="ollama")
    runtime = AIRuntime(configuration, adapters={"ollama": adapter})

    runtime.generate(prompt="Hello", policy="local")

    assert adapter.calls[0]["timeout_s"] == 45


def test_provider_timeout_override_wins_over_runtime_default(
    configuration: RuntimeConfiguration,
) -> None:
    configuration = replace(
        configuration,
        providers={
            **configuration.providers,
            "ollama": replace(configuration.providers["ollama"], timeout_s=60),
        },
    )
    adapter = _FakeAdapter(name="ollama")
    runtime = AIRuntime(configuration, adapters={"ollama": adapter})

    runtime.generate(prompt="Hello", policy="local")

    assert adapter.calls[0]["timeout_s"] == 60


def test_policy_latency_budget_caps_resolved_timeout_override(
    configuration: RuntimeConfiguration,
) -> None:
    configuration = replace(
        configuration,
        models={
            **configuration.models,
            "local-default": replace(
                configuration.models["local-default"], timeout_s=90
            ),
        },
    )
    adapter = _FakeAdapter(name="ollama")
    runtime = AIRuntime(configuration, adapters={"ollama": adapter})

    runtime.generate(prompt="Hello", policy="fast")

    assert 0 < adapter.calls[0]["timeout_s"] <= 5


def test_timeout_retries_the_same_candidate_with_exact_bound(
    configuration: RuntimeConfiguration,
) -> None:
    adapter = _SequenceAdapter(
        name="ollama",
        outcomes=[ProviderTimeoutError("timed out"), ProviderResult(text="Recovered")],
    )
    runtime = AIRuntime(configuration, adapters={"ollama": adapter})

    response = runtime.generate(prompt="Hello", policy="local")

    assert len(adapter.calls) == 2
    assert [attempt.status for attempt in response.attempts] == ["failed", "success"]
    assert response.attempts[0].error_code == "provider_timeout"


def test_rate_limit_retries_before_falling_back(
    configuration: RuntimeConfiguration,
) -> None:
    local = _SequenceAdapter(
        name="ollama",
        outcomes=[ProviderRateLimitError("limited"), ProviderRateLimitError("limited")],
    )
    cloud = _SequenceAdapter(name="mistral", outcomes=[ProviderResult(text="Cloud")])
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local, "mistral": cloud},
    )

    response = runtime.generate(prompt="Hello", policy="resilient")

    assert len(local.calls) == 2
    assert len(cloud.calls) == 1
    assert response.provider == "mistral"
    assert [attempt.status for attempt in response.attempts] == [
        "failed",
        "failed",
        "success",
    ]


def test_unavailable_provider_falls_back_to_next_candidate(
    configuration: RuntimeConfiguration,
) -> None:
    configuration = replace(
        configuration,
        runtime=RuntimeDefaults(timeout_s=30, retries_per_candidate=0),
    )
    local = _SequenceAdapter(name="ollama", outcomes=[ProviderUnavailableError("down")])
    cloud = _SequenceAdapter(name="mistral", outcomes=[ProviderResult(text="Cloud")])
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local, "mistral": cloud},
    )

    response = runtime.generate(prompt="Hello", policy="resilient")

    assert response.provider == "mistral"
    assert [attempt.provider for attempt in response.attempts] == ["ollama", "mistral"]
    assert [attempt.zone for attempt in response.attempts] == ["local", "eu-cloud"]


def test_timeout_falls_back_to_next_candidate(
    configuration: RuntimeConfiguration,
) -> None:
    configuration = replace(
        configuration,
        runtime=RuntimeDefaults(timeout_s=30, retries_per_candidate=0),
    )
    local = _SequenceAdapter(name="ollama", outcomes=[ProviderTimeoutError("timeout")])
    cloud = _SequenceAdapter(name="mistral", outcomes=[ProviderResult(text="Cloud")])
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local, "mistral": cloud},
    )

    response = runtime.generate(prompt="Hello", policy="resilient")

    assert response.provider == "mistral"
    assert [attempt.provider for attempt in response.attempts] == ["ollama", "mistral"]
    assert response.attempts[0].error_code == "provider_timeout"


@pytest.mark.parametrize(
    "error_type", [ProviderAuthenticationError, ProviderRequestError]
)
def test_non_fallback_eligible_errors_stop_without_fallback(
    configuration: RuntimeConfiguration,
    error_type: type[Exception],
) -> None:
    local = _SequenceAdapter(name="ollama", outcomes=[error_type("permanent")])
    cloud = _SequenceAdapter(name="mistral", outcomes=[ProviderResult(text="Cloud")])
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local, "mistral": cloud},
    )

    with pytest.raises(error_type) as error_info:
        runtime.generate(prompt="Hello", policy="resilient")

    assert len(local.calls) == 1
    assert cloud.calls == []
    assert len(error_info.value.attempts) == 1


def test_policy_without_fallback_stops_after_current_candidate(
    configuration: RuntimeConfiguration,
) -> None:
    local = _SequenceAdapter(
        name="ollama",
        outcomes=[ProviderUnavailableError("down"), ProviderUnavailableError("down")],
    )
    cloud = _SequenceAdapter(name="mistral", outcomes=[ProviderResult(text="Cloud")])
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local, "mistral": cloud},
    )

    with pytest.raises(ProviderUnavailableError):
        runtime.generate(prompt="Hello", policy="no-fallback")

    assert len(local.calls) == 2
    assert cloud.calls == []


def test_all_candidates_failing_preserves_complete_attempt_history(
    configuration: RuntimeConfiguration,
) -> None:
    configuration = replace(
        configuration,
        runtime=RuntimeDefaults(timeout_s=30, retries_per_candidate=0),
    )
    local = _SequenceAdapter(name="ollama", outcomes=[ProviderUnavailableError("down")])
    cloud = _SequenceAdapter(
        name="mistral", outcomes=[ProviderUnavailableError("down")]
    )
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local, "mistral": cloud},
    )

    with pytest.raises(AllCandidatesFailedError) as error_info:
        runtime.generate(prompt="Hello", policy="resilient")

    assert [attempt.provider for attempt in error_info.value.attempts] == [
        "ollama",
        "mistral",
    ]


def test_global_latency_budget_decreases_across_retry_and_fallback(
    configuration: RuntimeConfiguration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
    monkeypatch.setattr(
        "resilient_ai_runtime.core.runtime.perf_counter", lambda: next(clock)
    )
    local = _SequenceAdapter(
        name="ollama",
        outcomes=[ProviderTimeoutError("timeout"), ProviderTimeoutError("timeout")],
    )
    cloud = _SequenceAdapter(name="mistral", outcomes=[ProviderResult(text="Cloud")])
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local, "mistral": cloud},
    )

    runtime.generate(prompt="Hello", policy="budgeted")

    assert [call["timeout_s"] for call in local.calls + cloud.calls] == [10, 9, 8]


def test_runtime_timeout_stays_per_attempt_bound_across_retry_and_fallback(
    configuration: RuntimeConfiguration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = replace(
        configuration,
        runtime=RuntimeDefaults(timeout_s=2, retries_per_candidate=1),
    )
    clock = iter([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
    monkeypatch.setattr(
        "resilient_ai_runtime.core.runtime.perf_counter", lambda: next(clock)
    )
    local = _SequenceAdapter(
        name="ollama",
        outcomes=[ProviderTimeoutError("timeout"), ProviderTimeoutError("timeout")],
    )
    cloud = _SequenceAdapter(name="mistral", outcomes=[ProviderResult(text="Cloud")])
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local, "mistral": cloud},
    )

    runtime.generate(prompt="Hello", policy="budgeted")

    assert [call["timeout_s"] for call in local.calls + cloud.calls] == [2, 2, 2]


def test_exhausted_global_budget_prevents_fallback_attempt(
    configuration: RuntimeConfiguration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = replace(
        configuration,
        runtime=RuntimeDefaults(timeout_s=30, retries_per_candidate=0),
    )
    clock = iter([0, 0, 0, 1.1, 1.1, 1.1])
    monkeypatch.setattr(
        "resilient_ai_runtime.core.runtime.perf_counter", lambda: next(clock)
    )
    local = _SequenceAdapter(name="ollama", outcomes=[ProviderUnavailableError("down")])
    cloud = _SequenceAdapter(name="mistral", outcomes=[ProviderResult(text="Cloud")])
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local, "mistral": cloud},
    )

    with pytest.raises(AllCandidatesFailedError, match="latency budget exhausted"):
        runtime.generate(prompt="Hello", policy="short-budget")

    assert len(local.calls) == 1
    assert cloud.calls == []


def test_zero_latency_limit_prevents_execution(
    configuration: RuntimeConfiguration,
) -> None:
    adapter = _FakeAdapter(name="ollama")
    runtime = AIRuntime(configuration, adapters={"ollama": adapter})

    with pytest.raises(AllCandidatesFailedError, match="latency budget exhausted"):
        runtime.generate(prompt="Hello", policy="zero-latency")

    assert adapter.calls == []


def test_successful_request_emits_safe_structured_execution_record(
    configuration: RuntimeConfiguration,
) -> None:
    records: list[dict[str, object]] = []
    priced_local = replace(
        configuration.models["local-default"],
        pricing=PricingConfig(
            input_price_per_token=0.0002,
            output_price_per_token=0.0004,
        ),
    )
    configuration = replace(
        configuration,
        models={**configuration.models, "local-default": priced_local},
    )
    adapter = _FakeAdapter(name="ollama")
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": adapter},
        execution_logger=records.append,
    )

    response = runtime.generate(
        prompt="sensitive prompt",
        system_prompt="sensitive system prompt",
        policy="local",
        metadata={"content": "sensitive metadata"},
    )

    record = records[0]
    assert record["request_id"] == response.request_id
    assert record["policy"] == "local"
    assert record["selected_provider"] == "ollama"
    assert record["selected_model"] == "qwen3:8b"
    assert record["attempt_count"] == 1
    assert record["input_tokens"] == 12
    assert record["output_tokens"] == 4
    assert record["estimated_cost_usd"] == pytest.approx(0.004)
    assert record["status"] == "success"
    assert record["attempts"][0]["zone"] == "local"
    datetime.fromisoformat(str(record["timestamp"]))
    assert "sensitive" not in json.dumps(record)


def test_fallback_execution_record_reports_final_provider_and_attempt_count(
    configuration: RuntimeConfiguration,
) -> None:
    records: list[dict[str, object]] = []
    local = _SequenceAdapter(
        name="ollama",
        outcomes=[ProviderRateLimitError("limited"), ProviderRateLimitError("limited")],
    )
    cloud = _SequenceAdapter(name="mistral", outcomes=[ProviderResult(text="Cloud")])
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local, "mistral": cloud},
        execution_logger=records.append,
    )

    runtime.generate(prompt="Hello", policy="resilient")

    record = records[0]
    assert record["selected_provider"] == "mistral"
    assert record["selected_model"] == "mistral-small-latest"
    assert record["attempt_count"] == 3
    assert record["status"] == "success"


def test_failed_request_emits_failure_execution_record(
    configuration: RuntimeConfiguration,
) -> None:
    records: list[dict[str, object]] = []
    local = _SequenceAdapter(name="ollama", outcomes=[ProviderRequestError("invalid")])
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local},
        execution_logger=records.append,
    )

    with pytest.raises(ProviderRequestError):
        runtime.generate(prompt="sensitive prompt", policy="resilient")

    record = records[0]
    assert record["status"] == "failed"
    assert record["selected_provider"] == "ollama"
    assert record["attempt_count"] == 1
    assert record["error_code"] == "provider_request"
    assert "sensitive" not in json.dumps(record)
