"""Best-effort request cost estimation from explicit model pricing."""

from .config import ModelConfig
from .response import ProviderResult


def estimate_request_cost(model: ModelConfig, result: ProviderResult) -> float | None:
    """Return an estimated cost only when pricing and both usage counts exist."""

    if (
        model.pricing is None
        or result.input_tokens is None
        or result.output_tokens is None
    ):
        return None
    return (
        result.input_tokens * model.pricing.input_price_per_token
        + result.output_tokens * model.pricing.output_price_per_token
    )
