"""Send one request to a local Ollama server.

Requires Ollama running locally with the model already pulled:
    ollama pull qwen3:4b
    python examples/local_ollama_request.py
"""

from pathlib import Path

from resilient_ai_runtime import AIRuntime

config = Path(__file__).with_name("local.yaml")
response = AIRuntime.from_config(str(config)).generate(
    prompt="Say hello in one short sentence.", policy="local-only"
)

print(response.text)
print(response.provider, response.model, f"{response.latency_ms:.0f} ms")
