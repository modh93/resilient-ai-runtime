"""Mistral adapter for non-streaming cloud text generation."""

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


class MistralAdapter:
    """Execute normalized text-generation requests against Mistral's API."""

    name = "mistral"

    def __init__(
        self,
        base_url: str | None = None,
        credential_env: str = "MISTRAL_API_KEY",
    ) -> None:
        self.base_url = (base_url or "https://api.mistral.ai").rstrip("/")
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
            raise ProviderAuthenticationError("Mistral credentials are not configured")

        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(_payload(model, prompt, system_prompt)).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
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
            raise ProviderTimeoutError("Mistral request timed out") from error
        except OSError as error:
            raise ProviderUnavailableError("Mistral is unavailable") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProviderRequestError(
                "Mistral returned an invalid response"
            ) from error

        return _normalize_response(response_data)


def _payload(model: str, prompt: str, system_prompt: str | None) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return {"model": model, "messages": messages}


def _normalize_response(payload: Any) -> ProviderResult:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError) as error:
        raise ProviderRequestError(
            "Mistral response is missing generated text"
        ) from error

    text = _normalize_text_content(content)
    usage = payload.get("usage") if isinstance(payload, dict) else None
    return ProviderResult(
        text=text,
        input_tokens=_optional_int(usage, "prompt_tokens"),
        output_tokens=_optional_int(usage, "completion_tokens"),
    )


def _normalize_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks = [
            chunk["text"]
            for chunk in content
            if isinstance(chunk, dict)
            and chunk.get("type") == "text"
            and isinstance(chunk.get("text"), str)
        ]
        if text_chunks:
            return "".join(text_chunks)
    raise ProviderRequestError("Mistral response text is invalid")


def _optional_int(usage: Any, field_name: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(field_name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _raise_http_error(error: HTTPError) -> None:
    if error.code in {408, 504}:
        raise ProviderTimeoutError("Mistral request timed out") from error
    if error.code == 429:
        raise ProviderRateLimitError("Mistral rate limit exceeded") from error
    if error.code in {401, 403}:
        raise ProviderAuthenticationError("Mistral authentication failed") from error
    if 400 <= error.code < 500:
        raise ProviderRequestError("Mistral rejected the request") from error
    raise ProviderUnavailableError("Mistral is unavailable") from error


def _raise_url_error(error: URLError) -> None:
    if isinstance(error.reason, (TimeoutError, socket.timeout)):
        raise ProviderTimeoutError("Mistral request timed out") from error
    raise ProviderUnavailableError("Mistral is unavailable") from error
