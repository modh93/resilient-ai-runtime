"""Tests for the validated Runtime v0 YAML configuration loader."""

from pathlib import Path

import pytest

from resilient_ai_runtime.core import ConfigurationError, load_runtime_config

EXAMPLE_CONFIG = Path(__file__).parents[2] / "examples" / "runtime.example.yaml"
BENCHMARK_EXAMPLE_CONFIG = (
    Path(__file__).parents[2] / "benchmarks" / "runtime.benchmark.example.yaml"
)


def test_example_configuration_loads_into_typed_objects() -> None:
    config = load_runtime_config(EXAMPLE_CONFIG)

    assert config.runtime.timeout_s == 30.0
    assert config.runtime.retries_per_candidate == 1
    assert config.models["local-default"].timeout_s is None
    assert config.models["local-default"].zone == "local"
    assert config.models["mistral-default"].pricing is not None
    assert config.policies["balanced"].model_priority == (
        "local-default",
        "mistral-default",
        "openai-default",
        "anthropic-default",
    )


def test_public_example_configuration_loads_with_required_policy_patterns() -> None:
    config = load_runtime_config(EXAMPLE_CONFIG)

    assert {"local-only", "balanced", "confidential"} <= set(config.policies)
    assert config.providers["ollama"].timeout_s == 90.0
    assert config.providers["mistral"].timeout_s == 10.0


def test_benchmark_example_disables_retries_and_fallback() -> None:
    config = load_runtime_config(BENCHMARK_EXAMPLE_CONFIG)

    assert config.runtime.retries_per_candidate == 0
    assert all(not policy.fallback for policy in config.policies.values())


def test_provider_and_model_timeouts_are_loaded_when_configured(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path,
        _config_with_model(
            "    provider: test\n"
            "    model: example-model\n"
            "    zone: local\n"
            "    enabled: true\n"
            "    capabilities: [text]\n"
            "    timeout_s: 45\n",
            provider_settings="    timeout_s: 60\n",
        ),
    )

    config = load_runtime_config(path)

    assert config.providers["test"].timeout_s == 60.0
    assert config.models["example"].timeout_s == 45.0


@pytest.mark.parametrize(
    ("section", "field"),
    [("provider", "timeout_s: 0\n"), ("model", "timeout_s: -1\n")],
)
def test_non_positive_timeout_override_is_rejected(
    tmp_path: Path, section: str, field: str
) -> None:
    provider_settings = f"    {field}" if section == "provider" else ""
    model_timeout = f"    {field}" if section == "model" else ""
    path = _write_config(
        tmp_path,
        _config_with_model(
            "    provider: test\n"
            "    model: example-model\n"
            "    zone: local\n"
            "    enabled: true\n"
            "    capabilities: [text]\n"
            f"{model_timeout}",
            provider_settings=provider_settings,
        ),
    )

    with pytest.raises(ConfigurationError, match="timeout_s must be greater than zero"):
        load_runtime_config(path)


def test_malformed_yaml_raises_normalized_configuration_error(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "runtime: [not: valid")

    with pytest.raises(ConfigurationError, match="invalid YAML"):
        load_runtime_config(path)


def test_missing_required_model_metadata_is_rejected(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        _config_with_model(
            "    zone: local\n    enabled: true\n    capabilities: [text]\n"
        ),
    )

    with pytest.raises(ConfigurationError, match="models.example.provider is required"):
        load_runtime_config(path)


def test_invalid_model_zone_is_rejected(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        _config_with_model(
            "    provider: test\n"
            "    model: example-model\n"
            "    zone: moon\n"
            "    enabled: true\n"
            "    capabilities: [text]\n"
        ),
    )

    with pytest.raises(ConfigurationError, match="models.example.zone must be one of"):
        load_runtime_config(path)


def test_unknown_policy_model_reference_is_rejected(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        _config_with_model(
            "    provider: test\n"
            "    model: example-model\n"
            "    zone: local\n"
            "    enabled: true\n"
            "    capabilities: [text]\n",
            model_priority="      - unknown-model\n",
        ),
    )

    with pytest.raises(
        ConfigurationError, match="model_priority references unknown model"
    ):
        load_runtime_config(path)


@pytest.mark.parametrize("field_name", ["allowed_providers", "forbidden_providers"])
def test_unknown_policy_provider_reference_is_rejected(
    tmp_path: Path, field_name: str
) -> None:
    path = _write_config(
        tmp_path,
        _config_with_model(
            "    provider: test\n"
            "    model: example-model\n"
            "    zone: local\n"
            "    enabled: true\n"
            "    capabilities: [text]\n",
            policy_fields=f"    {field_name}:\n      - unknown-provider\n",
        ),
    )

    with pytest.raises(
        ConfigurationError,
        match=f"{field_name} references unknown provider 'unknown-provider'",
    ):
        load_runtime_config(path)


def test_known_policy_provider_references_are_accepted(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        _config_with_model(
            "    provider: test\n"
            "    model: example-model\n"
            "    zone: local\n"
            "    enabled: true\n"
            "    capabilities: [text]\n",
            policy_fields=(
                "    allowed_providers:\n"
                "      - test\n"
                "    forbidden_providers:\n"
                "      - test\n"
            ),
        ),
    )

    policy = load_runtime_config(path).policies["default"]

    assert policy.allowed_providers == ("test",)
    assert policy.forbidden_providers == ("test",)


def test_duplicate_logical_model_ids_are_rejected(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        _config_with_model(
            "    provider: test\n"
            "    model: example-model\n"
            "    zone: local\n"
            "    enabled: true\n"
            "    capabilities: [text]\n"
            "  example:\n"
            "    provider: test\n"
            "    model: another-model\n"
            "    zone: local\n"
            "    enabled: true\n"
            "    capabilities: [text]\n"
        ),
    )

    with pytest.raises(ConfigurationError, match="invalid YAML"):
        load_runtime_config(path)


def test_provider_environment_variables_are_resolved_on_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNTIME_TEST_TOKEN", "resolved-credential")
    monkeypatch.setenv("RUNTIME_TEST_BASE_URL", "http://localhost:11434")
    path = _write_config(
        tmp_path,
        _config_with_model(
            "    provider: test\n"
            "    model: example-model\n"
            "    zone: local\n"
            "    enabled: true\n"
            "    capabilities: [text]\n",
            provider_settings=(
                "    credential_env: RUNTIME_TEST_TOKEN\n"
                "    base_url_env: RUNTIME_TEST_BASE_URL\n"
            ),
        ),
    )

    provider = load_runtime_config(path).providers["test"]

    assert provider.resolve_credential() == "resolved-credential"
    assert provider.resolve_base_url() == "http://localhost:11434"


def test_example_configuration_contains_no_secret_values() -> None:
    contents = EXAMPLE_CONFIG.read_text(encoding="utf-8")

    assert "credential_env:" in contents
    assert "sk-" not in contents
    assert "api_key:" not in contents.lower()


def _write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "runtime.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def _config_with_model(
    model_fields: str,
    *,
    model_priority: str = "      - example\n",
    provider_settings: str = "",
    policy_fields: str = "",
) -> str:
    provider_config = (
        f"  test:\n{provider_settings}" if provider_settings else "  test: {}\n"
    )
    return (
        "runtime:\n"
        "  timeout_s: 30\n"
        "  retries_per_candidate: 1\n"
        "providers:\n"
        f"{provider_config}"
        "models:\n"
        "  example:\n"
        f"{model_fields}"
        "policies:\n"
        "  default:\n"
        f"    model_priority:\n{model_priority}"
        "    fallback: true\n"
        f"{policy_fields}"
    )
