"""Send one request through a cloud-only policy.

Requires MISTRAL_API_KEY in the environment (nothing is read from .env files).
    python examples/cloud_request.py
"""

from pathlib import Path

from resilient_ai_runtime import AIRuntime

config = Path(__file__).with_name("cloud.yaml")
response = AIRuntime.from_config(str(config)).generate(
    prompt="Say hello in one short sentence.", policy="cloud-only"
)

print(response.text)
print(response.provider, response.model, f"{response.latency_ms:.0f} ms")
