"""Opt-in public Runtime API smoke test for one configured cloud provider."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from resilient_ai_runtime import AIRuntime
from resilient_ai_runtime.core import GenerationResponse

pytestmark = [pytest.mark.integration, pytest.mark.cloud_integration]

_CREDENTIAL_ENVIRONMENTS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def test_cloud_runtime_path(tmp_path: Path) -> None:
    if os.getenv("RUN_CLOUD_INTEGRATION") != "1":
        pytest.skip("set RUN_CLOUD_INTEGRATION=1 to run a cloud provider test")

    provider = os.getenv("CLOUD_INTEGRATION_PROVIDER", "mistral")
    credential_env = _CREDENTIAL_ENVIRONMENTS.get(provider)
    if credential_env is None:
        pytest.skip(f"unsupported cloud integration provider {provider!r}")
    if not os.getenv(credential_env):
        pytest.skip(f"set {credential_env} to run the {provider} integration test")

    model = os.getenv("CLOUD_INTEGRATION_MODEL")
    if not model:
        pytest.skip("set CLOUD_INTEGRATION_MODEL to select a configured cloud model")

    response = AIRuntime.from_config(
        str(_write_config(tmp_path, provider, credential_env, model))
    ).generate(
        prompt="Reply with a short greeting.",
        policy="integration-cloud",
    )

    assert isinstance(response, GenerationResponse)
    assert response.text
    assert response.provider == provider
    assert response.model == model
    assert response.request_id
    assert response.attempts[-1].status == "success"


def _write_config(
    tmp_path: Path, provider: str, credential_env: str, model: str
) -> Path:
    config_path = tmp_path / "runtime-cloud.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  timeout_s: 30",
                "  retries_per_candidate: 0",
                "providers:",
                f"  {provider}:",
                f"    credential_env: {credential_env}",
                "models:",
                "  cloud-integration:",
                f"    provider: {provider}",
                f"    model: {json.dumps(model)}",
                "    zone: frontier",
                "    enabled: true",
                "    capabilities:",
                "      - text",
                "    pricing: null",
                "policies:",
                "  integration-cloud:",
                "    allowed_zones:",
                "      - frontier",
                "    model_priority:",
                "      - cloud-integration",
                "    fallback: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
