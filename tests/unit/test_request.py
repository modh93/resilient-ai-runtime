"""Tests for normalized generation requests."""

import pytest

from resilient_ai_runtime.core import GenerationRequest


@pytest.mark.parametrize("prompt", ["", "   ", "\t\n"])
def test_generation_request_rejects_empty_prompt(prompt: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        GenerationRequest(prompt=prompt)


def test_generation_request_accepts_minimal_prompt() -> None:
    request = GenerationRequest(prompt="Summarize this document")

    assert request.prompt == "Summarize this document"
    assert request.system_prompt is None
    assert request.policy is None


def test_generation_request_preserves_optional_fields() -> None:
    constraints = {"max_cost_usd": 0.03}
    metadata = {"project": "example-app"}

    request = GenerationRequest(
        prompt="Summarize this document",
        system_prompt="Be concise.",
        policy="confidential",
        constraints=constraints,
        metadata=metadata,
    )

    assert request.system_prompt == "Be concise."
    assert request.policy == "confidential"
    assert request.constraints == constraints
    assert request.metadata == metadata
