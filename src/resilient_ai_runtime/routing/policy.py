"""Policy resolution and request-constraint tightening for Runtime v0."""

from dataclasses import replace
from typing import Any

from resilient_ai_runtime.core.config import PolicyConfig
from resilient_ai_runtime.core.errors import PolicyViolationError

_ALLOWED_CONSTRAINTS = frozenset(
    {
        "max_cost_usd",
        "max_latency_ms",
        "allowed_zones",
        "allowed_providers",
        "forbidden_providers",
        "allowed_models",
        "forbidden_models",
    }
)


class PolicyResolver:
    """Resolve named policies and apply only stricter request constraints."""

    def __init__(self, policies: dict[str, PolicyConfig]) -> None:
        self._policies = dict(policies)

    def resolve(
        self, name: str, constraints: dict[str, Any] | None = None
    ) -> PolicyConfig:
        """Return the effective policy for one request.

        Request constraints can lower cost or latency limits and narrow allow
        lists. They cannot widen an existing allow list or remove a forbidden
        provider/model.
        """

        try:
            policy = self._policies[name]
        except KeyError as error:
            raise PolicyViolationError(f"unknown policy {name!r}") from error

        if constraints is None:
            return policy
        if not isinstance(constraints, dict):
            raise PolicyViolationError("request constraints must be a mapping")

        unknown_constraints = set(constraints) - _ALLOWED_CONSTRAINTS
        if unknown_constraints:
            raise PolicyViolationError(
                "unsupported request constraints: "
                f"{', '.join(sorted(unknown_constraints))}"
            )

        return replace(
            policy,
            max_cost_usd=_tighten_limit(
                policy.max_cost_usd, constraints.get("max_cost_usd"), "max_cost_usd"
            ),
            max_latency_ms=_tighten_limit(
                policy.max_latency_ms,
                constraints.get("max_latency_ms"),
                "max_latency_ms",
            ),
            allowed_zones=_narrow_allow_list(
                policy.allowed_zones, constraints.get("allowed_zones"), "allowed_zones"
            ),
            allowed_providers=_narrow_allow_list(
                policy.allowed_providers,
                constraints.get("allowed_providers"),
                "allowed_providers",
            ),
            allowed_models=_narrow_allow_list(
                policy.allowed_models,
                constraints.get("allowed_models"),
                "allowed_models",
            ),
            forbidden_providers=_extend_deny_list(
                policy.forbidden_providers,
                constraints.get("forbidden_providers"),
                "forbidden_providers",
            ),
            forbidden_models=_extend_deny_list(
                policy.forbidden_models,
                constraints.get("forbidden_models"),
                "forbidden_models",
            ),
        )


def _tighten_limit(
    current: float | int | None, requested: Any, name: str
) -> float | int | None:
    if requested is None:
        return current
    if isinstance(requested, bool) or not isinstance(requested, int | float):
        raise PolicyViolationError(f"{name} must be a number")
    if requested < 0:
        raise PolicyViolationError(f"{name} must be zero or greater")
    return requested if current is None else min(current, requested)


def _narrow_allow_list(
    current: tuple[str, ...] | None, requested: Any, name: str
) -> tuple[str, ...] | None:
    if requested is None:
        return current
    requested_values = _string_tuple(requested, name)
    if current is not None and not set(requested_values).issubset(current):
        raise PolicyViolationError(f"{name} cannot widen the named policy")
    return requested_values


def _extend_deny_list(
    current: tuple[str, ...] | None, requested: Any, name: str
) -> tuple[str, ...] | None:
    if requested is None:
        return current
    requested_values = _string_tuple(requested, name)
    return tuple(dict.fromkeys((current or ()) + requested_values))


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise PolicyViolationError(f"{name} must be a list of non-empty strings")
    return tuple(value)
