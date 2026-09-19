"""Regression tests for terminal failure observability."""

from dataclasses import dataclass, field

import pytest

from resilient_ai_runtime import AIRuntime
from resilient_ai_runtime.core import (
    AllCandidatesFailedError,
    ModelConfig,
    PolicyConfig,
    ProviderConfig,
    ProviderResult,
    ProviderUnavailableError,
    RuntimeConfiguration,
    RuntimeDefaults,
)


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
        self.calls.append({"model": model, "timeout_s": timeout_s})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_all_candidates_failed_record_keeps_terminal_provider_error_code() -> None:
    configuration = RuntimeConfiguration(
        runtime=RuntimeDefaults(timeout_s=30, retries_per_candidate=0),
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
        },
        policies={
            "resilient": PolicyConfig(
                name="resilient",
                model_priority=("local-default", "mistral-default"),
                fallback=True,
            )
        },
    )
    records: list[dict[str, object]] = []
    local = _SequenceAdapter(
        name="ollama", outcomes=[ProviderUnavailableError("local down")]
    )
    cloud = _SequenceAdapter(
        name="mistral", outcomes=[ProviderUnavailableError("cloud down")]
    )
    runtime = AIRuntime(
        configuration,
        adapters={"ollama": local, "mistral": cloud},
        execution_logger=records.append,
    )

    with pytest.raises(AllCandidatesFailedError):
        runtime.generate(prompt="Hello", policy="resilient")

    assert len(records) == 1
    record = records[0]
    assert record["status"] == "failed"
    assert record["selected_provider"] == "mistral"
    assert record["selected_model"] == "mistral-small-latest"
    assert record["attempt_count"] == 2
    assert record["error_code"] == "provider_unavailable"
