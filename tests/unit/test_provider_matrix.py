"""Mocked consistency tests for the OpenAI and Anthropic adapters."""

from __future__ import annotations

import json
import socket
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from resilient_ai_runtime.core import (
    ConfigurationError,
    ProviderAuthenticationError,
    ProviderConfig,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from resilient_ai_runtime.providers import (
    AnthropicAdapter,
    OpenAIAdapter,
    create_provider_adapter,
)


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


@pytest.fixture(params=["openai", "anthropic"])
def provider_case(request: pytest.FixtureRequest) -> dict[str, Any]:
    if request.param == "openai":
        return {
            "adapter": OpenAIAdapter,
            "credential_env": "OPENAI_API_KEY",
            "urlopen_path": "resilient_ai_runtime.providers.openai.urlopen",
            "response": {
                "choices": [{"message": {"content": "Generated text"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
            "minimal_response": {
                "choices": [{"message": {"content": "Generated text"}}]
            },
            "expected_payload": {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Summarize this."},
                ],
            },
        }
    return {
        "adapter": AnthropicAdapter,
        "credential_env": "ANTHROPIC_API_KEY",
        "urlopen_path": "resilient_ai_runtime.providers.anthropic.urlopen",
        "response": {
            "content": [{"type": "text", "text": "Generated text"}],
            "usage": {"input_tokens": 12, "output_tokens": 4},
        },
        "minimal_response": {"content": [{"type": "text", "text": "Generated text"}]},
        "expected_payload": {
            "model": "test-model",
            "max_tokens": 1024,
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "Summarize this."}],
        },
    }


def test_cloud_adapter_normalizes_text_usage_and_system_prompt(
    provider_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(provider_case["credential_env"], "test-token")
    captured_request: Any = None

    def fake_urlopen(request: Any, **kwargs: object) -> _Response:
        nonlocal captured_request
        captured_request = request
        return _Response(provider_case["response"])

    monkeypatch.setattr(provider_case["urlopen_path"], fake_urlopen)

    result = provider_case["adapter"]().generate(
        model="test-model",
        prompt="Summarize this.",
        system_prompt="Be concise.",
        timeout_s=5,
    )

    assert result.text == "Generated text"
    assert result.input_tokens == 12
    assert result.output_tokens == 4
    assert json.loads(captured_request.data) == provider_case["expected_payload"]


def test_cloud_adapter_allows_missing_usage_metadata(
    provider_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(provider_case["credential_env"], "test-token")
    monkeypatch.setattr(
        provider_case["urlopen_path"],
        lambda *args, **kwargs: _Response(provider_case["minimal_response"]),
    )

    result = provider_case["adapter"]().generate(
        model="test-model", prompt="Hello", system_prompt=None, timeout_s=5
    )

    assert result.input_tokens is None
    assert result.output_tokens is None


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (408, ProviderTimeoutError),
        (429, ProviderRateLimitError),
        (401, ProviderAuthenticationError),
        (400, ProviderRequestError),
        (500, ProviderUnavailableError),
    ],
)
def test_cloud_adapter_normalizes_http_failures(
    provider_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    error_type: type[Exception],
) -> None:
    monkeypatch.setenv(provider_case["credential_env"], "test-token")

    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        raise HTTPError("https://provider.test", status_code, "error", {}, None)

    monkeypatch.setattr(provider_case["urlopen_path"], fake_urlopen)

    with pytest.raises(error_type):
        provider_case["adapter"]().generate(
            model="test-model", prompt="Hello", system_prompt=None, timeout_s=5
        )


def test_cloud_adapter_normalizes_network_timeout(
    provider_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(provider_case["credential_env"], "test-token")

    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        raise URLError(socket.timeout())

    monkeypatch.setattr(provider_case["urlopen_path"], fake_urlopen)

    with pytest.raises(ProviderTimeoutError):
        provider_case["adapter"]().generate(
            model="test-model", prompt="Hello", system_prompt=None, timeout_s=5
        )


def test_cloud_adapter_normalizes_generic_network_failure(
    provider_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(provider_case["credential_env"], "test-token")

    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        raise URLError("connection refused")

    monkeypatch.setattr(provider_case["urlopen_path"], fake_urlopen)

    with pytest.raises(ProviderUnavailableError):
        provider_case["adapter"]().generate(
            model="test-model", prompt="Hello", system_prompt=None, timeout_s=5
        )


def test_cloud_adapter_normalizes_missing_credentials(
    provider_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(provider_case["credential_env"], raising=False)

    with pytest.raises(ProviderAuthenticationError):
        provider_case["adapter"]().generate(
            model="test-model", prompt="Hello", system_prompt=None, timeout_s=5
        )


def test_factory_constructs_all_registered_cloud_adapters() -> None:
    openai = create_provider_adapter(
        ProviderConfig(name="openai", credential_env="OPENAI_TEST_KEY")
    )
    anthropic = create_provider_adapter(
        ProviderConfig(name="anthropic", credential_env="ANTHROPIC_TEST_KEY")
    )

    assert isinstance(openai, OpenAIAdapter)
    assert isinstance(anthropic, AnthropicAdapter)


def test_factory_still_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigurationError, match="no adapter is registered"):
        create_provider_adapter(ProviderConfig(name="unknown"))
