"""Normalized result models shared by providers and the runtime."""

from dataclasses import dataclass, field


@dataclass
class AttemptRecord:
    """The normalized outcome of one provider/model execution attempt.

    ``zone`` records the configured execution boundary for the attempted
    candidate. It does not prove network-level payload delivery.
    """

    provider: str
    model: str
    status: str
    latency_ms: int
    error_code: str | None = None
    zone: str | None = None


@dataclass
class ProviderResult:
    """Provider-adapter output before runtime orchestration adds metadata."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class GenerationResponse:
    """Provider-agnostic response returned by the runtime."""

    text: str
    provider: str
    model: str
    latency_ms: int
    request_id: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
