"""Tests for deterministic model registry, policies, and candidate routing."""

import pytest

from resilient_ai_runtime.core import (
    ModelConfig,
    NoEligibleModelError,
    PolicyConfig,
    PolicyViolationError,
)
from resilient_ai_runtime.routing import CandidateRouter, ModelRegistry, PolicyResolver


@pytest.fixture
def models() -> dict[str, ModelConfig]:
    return {
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
        "openai-default": ModelConfig(
            id="openai-default",
            provider="openai",
            model="gpt-example",
            zone="frontier",
            enabled=True,
            capabilities=("text",),
        ),
        "disabled-local": ModelConfig(
            id="disabled-local",
            provider="ollama",
            model="qwen3:4b",
            zone="local",
            enabled=False,
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
    }


@pytest.fixture
def policies() -> dict[str, PolicyConfig]:
    return {
        "balanced": PolicyConfig(
            name="balanced",
            model_priority=(
                "openai-default",
                "local-default",
                "mistral-default",
                "disabled-local",
                "vision-only",
            ),
            fallback=True,
            max_cost_usd=0.10,
            max_latency_ms=15_000,
        ),
        "confidential": PolicyConfig(
            name="confidential",
            model_priority=("local-default", "mistral-default", "openai-default"),
            fallback=True,
            allowed_zones=("local", "eu-cloud"),
            forbidden_providers=("openai",),
        ),
        "local-only": PolicyConfig(
            name="local-only",
            model_priority=("local-default", "mistral-default", "openai-default"),
            fallback=True,
            allowed_zones=("local",),
        ),
    }


@pytest.fixture
def router(models: dict[str, ModelConfig]) -> CandidateRouter:
    return CandidateRouter(ModelRegistry(models))


@pytest.fixture
def resolver(policies: dict[str, PolicyConfig]) -> PolicyResolver:
    return PolicyResolver(policies)


def test_forbidden_provider_is_never_selected(
    router: CandidateRouter,
    resolver: PolicyResolver,
) -> None:
    candidates = router.get_candidates(policy=resolver.resolve("confidential"))

    assert [candidate.provider for candidate in candidates] == ["ollama", "mistral"]


def test_forbidden_zone_is_never_selected(
    router: CandidateRouter,
    resolver: PolicyResolver,
) -> None:
    candidates = router.get_candidates(policy=resolver.resolve("local-only"))

    assert [candidate.zone for candidate in candidates] == ["local"]


def test_disabled_models_and_missing_capabilities_are_ignored(
    router: CandidateRouter,
    resolver: PolicyResolver,
) -> None:
    candidates = router.get_candidates(policy=resolver.resolve("balanced"))

    assert [candidate.id for candidate in candidates] == [
        "openai-default",
        "local-default",
        "mistral-default",
    ]


def test_request_constraint_tightens_policy_limit() -> None:
    resolver = PolicyResolver(
        {
            "balanced": PolicyConfig(
                name="balanced",
                model_priority=("local-default",),
                fallback=True,
                max_cost_usd=0.10,
                max_latency_ms=10_000,
            )
        }
    )

    policy = resolver.resolve(
        "balanced", {"max_cost_usd": 0.03, "max_latency_ms": 5_000}
    )

    assert policy.max_cost_usd == 0.03
    assert policy.max_latency_ms == 5_000


def test_request_constraint_cannot_weaken_allowed_zone(
    resolver: PolicyResolver,
) -> None:
    with pytest.raises(PolicyViolationError, match="allowed_zones cannot widen"):
        resolver.resolve(
            "local-only",
            {"allowed_zones": ["local", "frontier"]},
        )


def test_explicit_model_priority_is_preserved(
    router: CandidateRouter,
    resolver: PolicyResolver,
) -> None:
    candidates = router.get_candidates(policy=resolver.resolve("balanced"))

    assert [candidate.id for candidate in candidates] == [
        "openai-default",
        "local-default",
        "mistral-default",
    ]


def test_local_only_returns_only_local_candidates(
    router: CandidateRouter,
    resolver: PolicyResolver,
) -> None:
    candidates = router.get_candidates(policy=resolver.resolve("local-only"))

    assert [candidate.id for candidate in candidates] == ["local-default"]


def test_unknown_or_duplicate_priority_entries_do_not_create_candidates(
    models: dict[str, ModelConfig],
) -> None:
    router = CandidateRouter(ModelRegistry(models))
    policy = PolicyConfig(
        name="safe",
        model_priority=("unknown", "local-default", "local-default"),
        fallback=True,
    )

    candidates = router.get_candidates(policy=policy)

    assert [candidate.id for candidate in candidates] == ["local-default"]


def test_no_eligible_candidate_raises_normalized_error(
    router: CandidateRouter,
    resolver: PolicyResolver,
) -> None:
    with pytest.raises(NoEligibleModelError, match="no eligible model"):
        router.get_candidates(
            policy=resolver.resolve("local-only"),
            capability="vision",
        )
