"""Mocked unit tests for the Mistral provider adapter."""

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
    MistralAdapter,
    OllamaAdapter,
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


def test_mistral_response_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-token")

    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        return _Response(
            {
                "choices": [{"message": {"content": "Generated text"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }
        )

    monkeypatch.setattr("resilient_ai_runtime.providers.mistral.urlopen", fake_urlopen)

    result = MistralAdapter().generate(
        model="mistral-test-model",
        prompt="Summarize this.",
        system_prompt=None,
        timeout_s=5,
    )

    assert result.text == "Generated text"
    assert result.input_tokens == 12
    assert result.output_tokens == 4


def test_mistral_structured_text_content_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-token")

    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Generated "},
                                {"type": "text", "text": "text"},
                            ]
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("resilient_ai_runtime.providers.mistral.urlopen", fake_urlopen)

    result = MistralAdapter().generate(
        model="mistral-test-model",
        prompt="Summarize this.",
        system_prompt=None,
        timeout_s=5,
    )

    assert result.text == "Generated text"


def test_mistral_normalizes_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-token")
    captured_request: Any = None

    def fake_urlopen(request: Any, **kwargs: object) -> _Response:
        nonlocal captured_request
        captured_request = request
        return _Response({"choices": [{"message": {"content": "Generated text"}}]})

    monkeypatch.setattr("resilient_ai_runtime.providers.mistral.urlopen", fake_urlopen)

    MistralAdapter().generate(
        model="mistral-test-model",
        prompt="Summarize this.",
        system_prompt="Be concise.",
        timeout_s=5,
    )

    assert json.loads(captured_request.data) == {
        "model": "mistral-test-model",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Summarize this."},
        ],
    }


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
def test_mistral_http_failures_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    error_type: type[Exception],
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-token")

    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        raise HTTPError("https://api.mistral.ai", status_code, "error", {}, None)

    monkeypatch.setattr("resilient_ai_runtime.providers.mistral.urlopen", fake_urlopen)

    with pytest.raises(error_type):
        MistralAdapter().generate(
            model="mistral-test-model",
            prompt="Hello",
            system_prompt=None,
            timeout_s=5,
        )


def test_mistral_network_timeout_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-token")

    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        raise URLError(socket.timeout())

    monkeypatch.setattr("resilient_ai_runtime.providers.mistral.urlopen", fake_urlopen)

    with pytest.raises(ProviderTimeoutError):
        MistralAdapter().generate(
            model="mistral-test-model",
            prompt="Hello",
            system_prompt=None,
            timeout_s=5,
        )


def test_mistral_network_failure_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-token")

    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        raise URLError("connection refused")

    monkeypatch.setattr("resilient_ai_runtime.providers.mistral.urlopen", fake_urlopen)

    with pytest.raises(ProviderUnavailableError):
        MistralAdapter().generate(
            model="mistral-test-model",
            prompt="Hello",
            system_prompt=None,
            timeout_s=5,
        )


def test_mistral_missing_credentials_are_normalized() -> None:
    with pytest.raises(ProviderAuthenticationError, match="credentials"):
        MistralAdapter(credential_env="MISSING_MISTRAL_CREDENTIAL").generate(
            model="mistral-test-model",
            prompt="Hello",
            system_prompt=None,
            timeout_s=5,
        )


def test_mistral_missing_usage_metadata_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-token")

    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        return _Response({"choices": [{"message": {"content": "Generated text"}}]})

    monkeypatch.setattr("resilient_ai_runtime.providers.mistral.urlopen", fake_urlopen)

    result = MistralAdapter().generate(
        model="mistral-test-model",
        prompt="Hello",
        system_prompt=None,
        timeout_s=5,
    )

    assert result.input_tokens is None
    assert result.output_tokens is None


def test_factory_constructs_registered_adapters() -> None:
    ollama = create_provider_adapter(
        ProviderConfig(name="ollama", base_url="http://localhost:11434")
    )
    mistral = create_provider_adapter(
        ProviderConfig(name="mistral", credential_env="CUSTOM_MISTRAL_KEY")
    )

    assert isinstance(ollama, OllamaAdapter)
    assert isinstance(mistral, MistralAdapter)


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigurationError, match="no adapter is registered"):
        create_provider_adapter(ProviderConfig(name="unknown"))
