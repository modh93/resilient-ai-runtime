"""Try a local model first and fall back to the cloud if it fails.

Stop Ollama (or leave it unreachable) and set MISTRAL_API_KEY to see fallback.
    python examples/fallback_request.py
"""

from pathlib import Path

from resilient_ai_runtime import AIRuntime

config = Path(__file__).with_name("fallback.yaml")
response = AIRuntime.from_config(str(config)).generate(
    prompt="Say hello in one short sentence.", policy="local-then-cloud"
)

print(response.text)
print("served by:", response.provider, response.model)
for attempt in response.attempts:
    print(" attempt:", attempt)
