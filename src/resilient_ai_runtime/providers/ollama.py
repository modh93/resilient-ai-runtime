"""Ollama adapter for local, non-streaming text generation."""

from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from resilient_ai_runtime.core.errors import (
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from resilient_ai_runtime.core.response import ProviderResult


class OllamaAdapter:
    """Execute normalized text-generation requests against an Ollama server."""

    name = "ollama"

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: str | None,
        timeout_s: float,
    ) -> ProviderResult:
        """Generate one complete response without streaming or retrying."""

        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(_payload(model, prompt, system_prompt)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_s) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            _raise_http_error(error)
        except URLError as error:
            _raise_url_error(error)
        except (TimeoutError, socket.timeout) as error:
            raise ProviderTimeoutError("Ollama request timed out") from error
        except OSError as error:
            raise ProviderUnavailableError("Ollama is unavailable") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProviderRequestError("Ollama returned an invalid response") from error

        return _normalize_response(response_data)


def _payload(model: str, prompt: str, system_prompt: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if system_prompt is not None:
        payload["system"] = system_prompt
    return payload


def _normalize_response(payload: Any) -> ProviderResult:
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
        raise ProviderRequestError("Ollama response is missing generated text")
    return ProviderResult(
        text=payload["response"],
        input_tokens=_optional_int(payload.get("prompt_eval_count")),
        output_tokens=_optional_int(payload.get("eval_count")),
    )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _raise_http_error(error: HTTPError) -> None:
    if error.code == 408 or error.code == 504:
        raise ProviderTimeoutError("Ollama request timed out") from error
    if 400 <= error.code < 500:
        raise ProviderRequestError("Ollama rejected the request") from error
    raise ProviderUnavailableError("Ollama is unavailable") from error


def _raise_url_error(error: URLError) -> None:
    if isinstance(error.reason, (TimeoutError, socket.timeout)):
        raise ProviderTimeoutError("Ollama request timed out") from error
    raise ProviderUnavailableError("Ollama is unavailable") from error
