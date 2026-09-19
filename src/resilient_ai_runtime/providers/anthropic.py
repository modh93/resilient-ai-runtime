"""Anthropic adapter for non-streaming cloud text generation."""

from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from resilient_ai_runtime.core.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from resilient_ai_runtime.core.response import ProviderResult


class AnthropicAdapter:
    """Execute normalized text-generation requests against Anthropic's API."""

    name = "anthropic"

    def __init__(
        self,
        base_url: str | None = None,
        credential_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self._credential_env = credential_env

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: str | None,
        timeout_s: float,
    ) -> ProviderResult:
        """Generate one complete response without streaming or retrying."""

        api_key = os.getenv(self._credential_env)
        if not api_key:
            raise ProviderAuthenticationError(
                "Anthropic credentials are not configured"
            )

        request = Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(_payload(model, prompt, system_prompt)).encode("utf-8"),
            headers={
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "x-api-key": api_key,
            },
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
            raise ProviderTimeoutError("Anthropic request timed out") from error
        except OSError as error:
            raise ProviderUnavailableError("Anthropic is unavailable") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProviderRequestError(
                "Anthropic returned an invalid response"
            ) from error

        return _normalize_response(response_data)


def _payload(model: str, prompt: str, system_prompt: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt is not None:
        payload["system"] = system_prompt
    return payload


def _normalize_response(payload: Any) -> ProviderResult:
    try:
        content = payload["content"]
    except (KeyError, TypeError) as error:
        raise ProviderRequestError(
            "Anthropic response is missing generated text"
        ) from error
    if not isinstance(content, list):
        raise ProviderRequestError("Anthropic response text is invalid")
    text = "".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    if not text:
        raise ProviderRequestError("Anthropic response text is invalid")

    usage = payload.get("usage") if isinstance(payload, dict) else None
    return ProviderResult(
        text=text,
        input_tokens=_optional_int(usage, "input_tokens"),
        output_tokens=_optional_int(usage, "output_tokens"),
    )


def _optional_int(usage: Any, field_name: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(field_name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _raise_http_error(error: HTTPError) -> None:
    if error.code in {408, 504}:
        raise ProviderTimeoutError("Anthropic request timed out") from error
    if error.code == 429:
        raise ProviderRateLimitError("Anthropic rate limit exceeded") from error
    if error.code in {401, 403}:
        raise ProviderAuthenticationError("Anthropic authentication failed") from error
    if 400 <= error.code < 500:
        raise ProviderRequestError("Anthropic rejected the request") from error
    raise ProviderUnavailableError("Anthropic is unavailable") from error


def _raise_url_error(error: URLError) -> None:
    if isinstance(error.reason, (TimeoutError, socket.timeout)):
        raise ProviderTimeoutError("Anthropic request timed out") from error
    raise ProviderUnavailableError("Anthropic is unavailable") from error
