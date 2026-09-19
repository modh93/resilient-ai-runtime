"""Typed loading and validation for Runtime v0 YAML configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError

VALID_ZONES = frozenset({"local", "eu-cloud", "frontier"})


@dataclass(frozen=True)
class RuntimeDefaults:
    """Runtime-wide execution defaults."""

    timeout_s: float
    retries_per_candidate: int


@dataclass(frozen=True)
class ProviderConfig:
    """Safe provider connection settings, with secrets kept in the environment."""

    name: str
    base_url: str | None = None
    base_url_env: str | None = None
    credential_env: str | None = None
    timeout_s: float | None = None

    def resolve_base_url(self) -> str | None:
        """Return the configured base URL, preferring its environment override."""

        if self.base_url_env:
            return os.getenv(self.base_url_env, self.base_url)
        return self.base_url

    def resolve_credential(self) -> str | None:
        """Resolve a credential only when a caller needs to use this provider."""

        return os.getenv(self.credential_env) if self.credential_env else None


@dataclass(frozen=True)
class PricingConfig:
    """Configured per-token prices used later for best-effort cost estimation."""

    input_price_per_token: float
    output_price_per_token: float


@dataclass(frozen=True)
class ModelConfig:
    """One logical model entry in the runtime registry."""

    id: str
    provider: str
    model: str
    zone: str
    enabled: bool
    capabilities: tuple[str, ...]
    pricing: PricingConfig | None = None
    timeout_s: float | None = None


@dataclass(frozen=True)
class PolicyConfig:
    """A named set of constraints that later routing code will enforce."""

    name: str
    model_priority: tuple[str, ...]
    fallback: bool
    allowed_zones: tuple[str, ...] | None = None
    allowed_providers: tuple[str, ...] | None = None
    forbidden_providers: tuple[str, ...] | None = None
    allowed_models: tuple[str, ...] | None = None
    forbidden_models: tuple[str, ...] | None = None
    max_cost_usd: float | None = None
    max_latency_ms: int | None = None


@dataclass(frozen=True)
class RuntimeConfiguration:
    """All validated configuration needed by Runtime v0."""

    runtime: RuntimeDefaults
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    policies: dict[str, PolicyConfig]


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_runtime_config(path: str | Path) -> RuntimeConfiguration:
    """Load and validate a Runtime v0 YAML configuration file.

    Credentials remain referenced by environment-variable name and are resolved
    only through :class:`ProviderConfig` when a provider is used.
    """

    config_path = Path(path)
    try:
        raw_config = yaml.load(
            config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except OSError as error:
        raise ConfigurationError(
            f"unable to read configuration file: {config_path}"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"invalid YAML configuration in {config_path}"
        ) from error

    root = _mapping(raw_config, "configuration")
    _validate_keys(
        root, {"runtime", "providers", "models", "policies"}, "configuration"
    )

    providers = _parse_providers(_required_mapping(root, "providers", "configuration"))
    models = _parse_models(
        _required_mapping(root, "models", "configuration"), providers
    )
    policies = _parse_policies(
        _required_mapping(root, "policies", "configuration"), models, providers
    )

    return RuntimeConfiguration(
        runtime=_parse_runtime(_required_mapping(root, "runtime", "configuration")),
        providers=providers,
        models=models,
        policies=policies,
    )


def _parse_runtime(raw: dict[str, Any]) -> RuntimeDefaults:
    _validate_keys(raw, {"timeout_s", "retries_per_candidate"}, "runtime")
    timeout_s = _required_number(raw, "timeout_s", "runtime")
    retries_per_candidate = _required_int(raw, "retries_per_candidate", "runtime")
    if timeout_s <= 0:
        raise ConfigurationError("runtime.timeout_s must be greater than zero")
    if retries_per_candidate < 0:
        raise ConfigurationError(
            "runtime.retries_per_candidate must be zero or greater"
        )
    return RuntimeDefaults(
        timeout_s=timeout_s, retries_per_candidate=retries_per_candidate
    )


def _parse_providers(raw: dict[str, Any]) -> dict[str, ProviderConfig]:
    providers: dict[str, ProviderConfig] = {}
    for name, value in raw.items():
        provider_name = _non_empty_string(name, "provider name")
        config = _mapping(value, f"providers.{provider_name}")
        _validate_keys(
            config,
            {"base_url", "base_url_env", "credential_env", "timeout_s"},
            f"providers.{provider_name}",
        )
        providers[provider_name] = ProviderConfig(
            name=provider_name,
            base_url=_optional_string(config, "base_url", f"providers.{provider_name}"),
            base_url_env=_optional_string(
                config, "base_url_env", f"providers.{provider_name}"
            ),
            credential_env=_optional_string(
                config, "credential_env", f"providers.{provider_name}"
            ),
            timeout_s=_optional_positive_number(
                config, "timeout_s", f"providers.{provider_name}"
            ),
        )
    return providers


def _parse_models(
    raw: dict[str, Any], providers: dict[str, ProviderConfig]
) -> dict[str, ModelConfig]:
    models: dict[str, ModelConfig] = {}
    for model_id, value in raw.items():
        logical_id = _non_empty_string(model_id, "model ID")
        config = _mapping(value, f"models.{logical_id}")
        _validate_keys(
            config,
            {
                "provider",
                "model",
                "zone",
                "enabled",
                "capabilities",
                "pricing",
                "timeout_s",
            },
            f"models.{logical_id}",
        )
        provider = _required_string(config, "provider", f"models.{logical_id}")
        if provider not in providers:
            raise ConfigurationError(
                f"models.{logical_id}.provider references unknown provider {provider!r}"
            )
        zone = _required_string(config, "zone", f"models.{logical_id}")
        if zone not in VALID_ZONES:
            raise ConfigurationError(
                f"models.{logical_id}.zone must be one of: {', '.join(sorted(VALID_ZONES))}"
            )
        models[logical_id] = ModelConfig(
            id=logical_id,
            provider=provider,
            model=_required_string(config, "model", f"models.{logical_id}"),
            zone=zone,
            enabled=_required_bool(config, "enabled", f"models.{logical_id}"),
            capabilities=_required_string_list(
                config, "capabilities", f"models.{logical_id}"
            ),
            pricing=_parse_pricing(
                config.get("pricing"), f"models.{logical_id}.pricing"
            ),
            timeout_s=_optional_positive_number(
                config, "timeout_s", f"models.{logical_id}"
            ),
        )
    return models


def _parse_pricing(raw: Any, location: str) -> PricingConfig | None:
    if raw is None:
        return None
    pricing = _mapping(raw, location)
    _validate_keys(
        pricing, {"input_price_per_token", "output_price_per_token"}, location
    )
    input_price = _required_number(pricing, "input_price_per_token", location)
    output_price = _required_number(pricing, "output_price_per_token", location)
    if input_price < 0 or output_price < 0:
        raise ConfigurationError(f"{location} values must be zero or greater")
    return PricingConfig(
        input_price_per_token=input_price,
        output_price_per_token=output_price,
    )


def _parse_policies(
    raw: dict[str, Any],
    models: dict[str, ModelConfig],
    providers: dict[str, ProviderConfig],
) -> dict[str, PolicyConfig]:
    policies: dict[str, PolicyConfig] = {}
    for policy_name, value in raw.items():
        name = _non_empty_string(policy_name, "policy name")
        config = _mapping(value, f"policies.{name}")
        _validate_keys(
            config,
            {
                "allowed_zones",
                "allowed_providers",
                "forbidden_providers",
                "allowed_models",
                "forbidden_models",
                "model_priority",
                "fallback",
                "max_cost_usd",
                "max_latency_ms",
            },
            f"policies.{name}",
        )
        allowed_zones = _optional_string_list(
            config, "allowed_zones", f"policies.{name}"
        )
        if allowed_zones and (invalid_zones := set(allowed_zones) - VALID_ZONES):
            raise ConfigurationError(
                f"policies.{name}.allowed_zones contains invalid zones: "
                f"{', '.join(sorted(invalid_zones))}"
            )
        policy = PolicyConfig(
            name=name,
            model_priority=_required_string_list(
                config, "model_priority", f"policies.{name}"
            ),
            fallback=_required_bool(config, "fallback", f"policies.{name}"),
            allowed_zones=allowed_zones,
            allowed_providers=_optional_string_list(
                config, "allowed_providers", f"policies.{name}"
            ),
            forbidden_providers=_optional_string_list(
                config, "forbidden_providers", f"policies.{name}"
            ),
            allowed_models=_optional_string_list(
                config, "allowed_models", f"policies.{name}"
            ),
            forbidden_models=_optional_string_list(
                config, "forbidden_models", f"policies.{name}"
            ),
            max_cost_usd=_optional_non_negative_number(
                config, "max_cost_usd", f"policies.{name}"
            ),
            max_latency_ms=_optional_non_negative_int(
                config, "max_latency_ms", f"policies.{name}"
            ),
        )
        _validate_model_references(policy, models)
        _validate_provider_references(policy, providers)
        policies[name] = policy
    return policies


def _validate_model_references(
    policy: PolicyConfig, models: dict[str, ModelConfig]
) -> None:
    for field_name, model_ids in (
        ("model_priority", policy.model_priority),
        ("allowed_models", policy.allowed_models),
        ("forbidden_models", policy.forbidden_models),
    ):
        for model_id in model_ids or ():
            if model_id not in models:
                raise ConfigurationError(
                    f"policies.{policy.name}.{field_name} references unknown model {model_id!r}"
                )


def _validate_provider_references(
    policy: PolicyConfig, providers: dict[str, ProviderConfig]
) -> None:
    for field_name, provider_names in (
        ("allowed_providers", policy.allowed_providers),
        ("forbidden_providers", policy.forbidden_providers),
    ):
        for provider_name in provider_names or ():
            if provider_name not in providers:
                raise ConfigurationError(
                    f"policies.{policy.name}.{field_name} references unknown provider "
                    f"{provider_name!r}"
                )


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{location} must be a mapping")
    return value


def _required_mapping(raw: dict[str, Any], key: str, location: str) -> dict[str, Any]:
    if key not in raw:
        raise ConfigurationError(f"{location}.{key} is required")
    return _mapping(raw[key], f"{location}.{key}")


def _validate_keys(raw: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown_keys = set(raw) - allowed
    if unknown_keys:
        raise ConfigurationError(
            f"{location} contains unknown fields: {', '.join(sorted(unknown_keys))}"
        )


def _required_string(raw: dict[str, Any], key: str, location: str) -> str:
    if key not in raw:
        raise ConfigurationError(f"{location}.{key} is required")
    return _non_empty_string(raw[key], f"{location}.{key}")


def _optional_string(raw: dict[str, Any], key: str, location: str) -> str | None:
    if key not in raw or raw[key] is None:
        return None
    return _non_empty_string(raw[key], f"{location}.{key}")


def _non_empty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{location} must be a non-empty string")
    return value


def _required_bool(raw: dict[str, Any], key: str, location: str) -> bool:
    if key not in raw:
        raise ConfigurationError(f"{location}.{key} is required")
    if not isinstance(raw[key], bool):
        raise ConfigurationError(f"{location}.{key} must be a boolean")
    return raw[key]


def _required_number(raw: dict[str, Any], key: str, location: str) -> float:
    if key not in raw:
        raise ConfigurationError(f"{location}.{key} is required")
    return _number(raw[key], f"{location}.{key}")


def _optional_non_negative_number(
    raw: dict[str, Any], key: str, location: str
) -> float | None:
    if key not in raw or raw[key] is None:
        return None
    value = _number(raw[key], f"{location}.{key}")
    if value < 0:
        raise ConfigurationError(f"{location}.{key} must be zero or greater")
    return value


def _optional_positive_number(
    raw: dict[str, Any], key: str, location: str
) -> float | None:
    if key not in raw or raw[key] is None:
        return None
    value = _number(raw[key], f"{location}.{key}")
    if value <= 0:
        raise ConfigurationError(f"{location}.{key} must be greater than zero")
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{location} must be a number")
    return float(value)


def _required_int(raw: dict[str, Any], key: str, location: str) -> int:
    if key not in raw:
        raise ConfigurationError(f"{location}.{key} is required")
    return _integer(raw[key], f"{location}.{key}")


def _optional_non_negative_int(
    raw: dict[str, Any], key: str, location: str
) -> int | None:
    if key not in raw or raw[key] is None:
        return None
    value = _integer(raw[key], f"{location}.{key}")
    if value < 0:
        raise ConfigurationError(f"{location}.{key} must be zero or greater")
    return value


def _integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{location} must be an integer")
    return value


def _required_string_list(
    raw: dict[str, Any], key: str, location: str
) -> tuple[str, ...]:
    if key not in raw:
        raise ConfigurationError(f"{location}.{key} is required")
    return _string_list(raw[key], f"{location}.{key}")


def _optional_string_list(
    raw: dict[str, Any], key: str, location: str
) -> tuple[str, ...] | None:
    if key not in raw or raw[key] is None:
        return None
    return _string_list(raw[key], f"{location}.{key}")


def _string_list(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{location} must be a list of strings")
    return tuple(_non_empty_string(item, location) for item in value)
