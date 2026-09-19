"""Tests for normalized runtime response models."""

from resilient_ai_runtime.core import AttemptRecord, GenerationResponse, ProviderResult


def test_response_supports_missing_usage_and_cost_metadata() -> None:
    response = GenerationResponse(
        text="Summary",
        provider="ollama",
        model="qwen3:8b",
        latency_ms=120,
        request_id="request-123",
    )

    assert response.input_tokens is None
    assert response.output_tokens is None
    assert response.estimated_cost_usd is None
    assert response.attempts == []


def test_attempt_records_capture_success_and_failure() -> None:
    failed_attempt = AttemptRecord(
        provider="openai",
        model="gpt-example",
        status="failed",
        latency_ms=100,
        error_code="provider_unavailable",
        zone="frontier",
    )
    successful_attempt = AttemptRecord(
        provider="mistral",
        model="mistral-small-latest",
        status="success",
        latency_ms=200,
        zone="eu-cloud",
    )

    response = GenerationResponse(
        text="Summary",
        provider="mistral",
        model="mistral-small-latest",
        latency_ms=300,
        request_id="request-123",
        attempts=[failed_attempt, successful_attempt],
    )

    assert response.attempts == [failed_attempt, successful_attempt]
    assert successful_attempt.error_code is None
    assert successful_attempt.zone == "eu-cloud"


def test_provider_result_supports_missing_token_usage() -> None:
    result = ProviderResult(text="Summary")

    assert result.input_tokens is None
    assert result.output_tokens is None
