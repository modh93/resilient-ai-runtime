"""Provider-agnostic adapter contract for Runtime v0."""

from typing import Protocol

from resilient_ai_runtime.core.response import ProviderResult


class ProviderAdapter(Protocol):
    """Translate normalized generation inputs to one provider API."""

    name: str

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: str | None,
        timeout_s: float,
    ) -> ProviderResult:
        """Generate text and return only the normalized provider result."""
