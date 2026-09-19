"""Opt-in end-to-end test for the public Runtime API with Ollama."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from resilient_ai_runtime import AIRuntime
from resilient_ai_runtime.core import GenerationResponse

pytestmark = pytest.mark.integration

_TEMPLATE_PATH = Path(__file__).parent / "fixtures" / "runtime-ollama.yaml"


def test_ollama_runtime_path(tmp_path: Path) -> None:
    if os.getenv("RUN_OLLAMA_INTEGRATION") != "1":
        pytest.skip("set RUN_OLLAMA_INTEGRATION=1 to run against local Ollama")

    model = os.getenv("OLLAMA_INTEGRATION_MODEL", "qwen3:8b")
    _skip_unless_ollama_model_is_available(model)
    config_path = _write_config(tmp_path, model)

    response = AIRuntime.from_config(str(config_path)).generate(
        prompt="Reply with a short greeting.",
        policy="integration-local",
    )

    assert isinstance(response, GenerationResponse)
    assert response.text
    assert response.provider == "ollama"
    assert response.model == model
    assert response.request_id
    assert len(response.attempts) == 1
    assert response.attempts[0].status == "success"
    assert not hasattr(response, "raw_response")


def _skip_unless_ollama_model_is_available(model: str) -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        with urlopen(f"{base_url}/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError):
        pytest.skip(f"Ollama is not reachable at {base_url}")

    available_models = {
        entry.get("name")
        for entry in payload.get("models", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    if model not in available_models:
        pytest.skip(f"Ollama model {model!r} is not installed")


def _write_config(tmp_path: Path, model: str) -> Path:
    config_path = tmp_path / "runtime-ollama.yaml"
    config_path.write_text(
        _TEMPLATE_PATH.read_text(encoding="utf-8").replace("__OLLAMA_MODEL__", model),
        encoding="utf-8",
    )
    return config_path
