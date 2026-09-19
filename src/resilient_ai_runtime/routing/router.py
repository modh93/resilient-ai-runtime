"""Deterministic candidate selection for Runtime v0."""

from typing import Any

from resilient_ai_runtime.core.config import ModelConfig, PolicyConfig
from resilient_ai_runtime.core.errors import NoEligibleModelError

from .registry import ModelRegistry


class CandidateRouter:
    """Filter models by an effective policy while preserving explicit priority.

    Runtime v0 has no pre-call token estimate, so a cost limit does not exclude
    models merely because their request cost cannot be calculated yet. Cost is
    enforced only when a future caller supplies a calculable estimate.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def get_candidates(
        self,
        *,
        policy: PolicyConfig,
        capability: str = "text",
    ) -> list[ModelConfig]:
        """Return eligible models in an already-resolved policy's configured order."""

        candidates = [
            model
            for model in self._registry.in_priority_order(policy.model_priority)
            if _is_eligible(model, policy, capability)
        ]
        if not candidates:
            raise NoEligibleModelError(
                f"no eligible model for policy {policy.name!r} and capability {capability!r}"
            )
        return candidates


def _is_eligible(model: ModelConfig, policy: Any, capability: str) -> bool:
    if not model.enabled or capability not in model.capabilities:
        return False
    if policy.allowed_zones is not None and model.zone not in policy.allowed_zones:
        return False
    if (
        policy.allowed_providers is not None
        and model.provider not in policy.allowed_providers
    ):
        return False
    if policy.forbidden_providers and model.provider in policy.forbidden_providers:
        return False
    if policy.allowed_models is not None and model.id not in policy.allowed_models:
        return False
    return not (policy.forbidden_models and model.id in policy.forbidden_models)
