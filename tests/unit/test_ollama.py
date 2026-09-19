"""Mocked unit tests for the Ollama provider adapter."""

from __future__ import annotations

import json
import socket
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from resilient_ai_runtime.core import (
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from resilient_ai_runtime.providers import OllamaAdapter


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_ollama_response_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        return _Response(
            {"response": "Generated text", "prompt_eval_count": 12, "eval_count": 4}
        )

    monkeypatch.setattr("resilient_ai_runtime.providers.ollama.urlopen", fake_urlopen)

    result = OllamaAdapter("http://ollama.test").generate(
        model="qwen3:8b",
        prompt="Summarize this.",
        system_prompt=None,
        timeout_s=5,
    )

    assert result.text == "Generated text"
    assert result.input_tokens == 12
    assert result.output_tokens == 4


def test_ollama_includes_optional_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: Any = None

    def fake_urlopen(request: Any, **kwargs: object) -> _Response:
        nonlocal captured_request
        captured_request = request
        return _Response({"response": "Generated text"})

    monkeypatch.setattr("resilient_ai_runtime.providers.ollama.urlopen", fake_urlopen)

    OllamaAdapter().generate(
        model="qwen3:8b",
        prompt="Summarize this.",
        system_prompt="Be concise.",
        timeout_s=5,
    )

    assert json.loads(captured_request.data) == {
        "model": "qwen3:8b",
        "prompt": "Summarize this.",
        "stream": False,
        "system": "Be concise.",
    }


def test_ollama_timeout_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        raise URLError(socket.timeout())

    monkeypatch.setattr("resilient_ai_runtime.providers.ollama.urlopen", fake_urlopen)

    with pytest.raises(ProviderTimeoutError):
        OllamaAdapter().generate(
            model="qwen3:8b", prompt="Hello", system_prompt=None, timeout_s=5
        )


def test_ollama_network_failure_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        raise URLError("connection refused")

    monkeypatch.setattr("resilient_ai_runtime.providers.ollama.urlopen", fake_urlopen)

    with pytest.raises(ProviderUnavailableError):
        OllamaAdapter().generate(
            model="qwen3:8b", prompt="Hello", system_prompt=None, timeout_s=5
        )


def test_ollama_invalid_request_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        raise HTTPError("http://ollama.test/api/generate", 400, "Bad Request", {}, None)

    monkeypatch.setattr("resilient_ai_runtime.providers.ollama.urlopen", fake_urlopen)

    with pytest.raises(ProviderRequestError):
        OllamaAdapter().generate(
            model="qwen3:8b", prompt="Hello", system_prompt=None, timeout_s=5
        )


def test_ollama_missing_usage_metadata_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*args: object, **kwargs: object) -> _Response:
        return _Response({"response": "Generated text"})

    monkeypatch.setattr("resilient_ai_runtime.providers.ollama.urlopen", fake_urlopen)

    result = OllamaAdapter().generate(
        model="qwen3:8b", prompt="Hello", system_prompt=None, timeout_s=5
    )

    assert result.input_tokens is None
    assert result.output_tokens is None
