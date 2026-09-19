"""Tests for best-effort cost estimation from model configuration."""

from resilient_ai_runtime.core import (
    ModelConfig,
    PricingConfig,
    ProviderResult,
    estimate_request_cost,
)


def test_cost_uses_configured_prices_and_provider_usage() -> None:
    model = _model(
        PricingConfig(input_price_per_token=0.0002, output_price_per_token=0.0004)
    )
    result = ProviderResult(text="Generated", input_tokens=12, output_tokens=4)

    assert estimate_request_cost(model, result) == 0.004


def test_cost_is_none_when_usage_is_missing() -> None:
    model = _model(
        PricingConfig(input_price_per_token=0.0002, output_price_per_token=0.0004)
    )

    assert estimate_request_cost(model, ProviderResult(text="Generated")) is None


def test_cost_is_none_when_pricing_is_missing_for_local_model() -> None:
    result = ProviderResult(text="Generated", input_tokens=12, output_tokens=4)

    assert estimate_request_cost(_model(None), result) is None


def _model(pricing: PricingConfig | None) -> ModelConfig:
    return ModelConfig(
        id="local-default",
        provider="ollama",
        model="qwen3:8b",
        zone="local",
        enabled=True,
        capabilities=("text",),
        pricing=pricing,
    )
