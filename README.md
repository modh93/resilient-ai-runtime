# Resilient AI Runtime

> **Status: V0 validated / pre-production.** Not production-ready. See [Limitations](#limitations).

Resilient AI Runtime is a lightweight execution layer that lets application code call one common interface while routing requests across local and cloud LLM providers through explicit, YAML-defined policies.

## Why it exists

Applications often become directly coupled to a single provider: its API shape, model identifiers, retry behavior, cost, latency, fallback story and data locality. This runtime makes those choices explicit and reviewable in a policy file instead of scattering them through application code.

## Current capabilities

- Adapters for **OpenAI**, **Anthropic**, **Mistral** and **Ollama** (standard library HTTP only; the sole dependency is PyYAML)
- Deterministic routing driven by each policy's `model_priority`
- Bounded retry per candidate and ordered fallback across candidates
- YAML policies: allowed zones, allowed/forbidden providers, latency and cost limits
- Structured execution metadata (no prompts, responses or secrets)
- Latency measurement and best-effort cost estimation
- Local/cloud routing through provider and zone abstractions
- Text generation only

## Quickstart

```bash
pip install -e ".[dev]"      # from a clone; Python 3.10+
export MISTRAL_API_KEY=...   # only for the providers your policy uses
```

```python
from resilient_ai_runtime import AIRuntime

ai = AIRuntime.from_config("examples/cloud.yaml")
response = ai.generate(prompt="Summarize: ...", policy="cloud-only")

print(response.text)
print(response.provider, response.model, response.latency_ms, response.estimated_cost_usd)
```

Secrets are read from environment variables named in the config (`credential_env`). `.env` files are not loaded automatically; see [`.env.example`](.env.example).

## Policy example: local → cloud

```yaml
models:
  local-default:   {provider: ollama,  model: qwen3:4b,             zone: local,    enabled: true, capabilities: [text]}
  mistral-default: {provider: mistral, model: mistral-small-latest, zone: eu-cloud, enabled: true, capabilities: [text]}

policies:
  local-then-cloud:
    allowed_zones: [local, eu-cloud]
    model_priority: [local-default, mistral-default]   # the only ordering mechanism
    fallback: true
    max_latency_ms: 20000
```

Full files (including required `runtime` and `providers` sections) are in [`examples/`](examples). [`examples/runtime.example.yaml`](examples/runtime.example.yaml) shows several policies, including `forbidden_providers` and cost limits. Model IDs and prices are illustrative and may be outdated; the OpenAI and Anthropic entries in `runtime.example.yaml` are `REPLACE_WITH_*` placeholders you must fill in with a model ID available to your account.

## Routing zones

Models declare a `zone`, and policies restrict which zones may be used. The concept maps onto three deployment postures:

| Zone | Meaning |
| --- | --- |
| **S0 Frontier** | Maximum capability; frontier APIs (e.g. OpenAI, Anthropic) |
| **S1 Hybrid** | Balance of capability, cost and control; local first with selective cloud escalation |
| **S2 Sovereign** | Priority on control; inference and data stay in an environment you control |

These are **conceptual routing/policy zones, not certifications or compliance guarantees.** Zone names in config (`local`, `eu-cloud`, `frontier`) are labels you assign; the runtime enforces the policy you write, not any legal or regulatory status of a provider.

## Fallback example

```python
ai = AIRuntime.from_config("examples/fallback.yaml")
response = ai.generate(prompt="Hello", policy="local-then-cloud")
```

With Ollama stopped and `retries_per_candidate: 1`, a recorded run looked like:

1. Ollama: `provider_unavailable`
2. Ollama: `provider_unavailable` (retry)
3. Mistral: success

`response.attempts` holds this history. If every candidate fails, `AllCandidatesFailedError` is raised with the attempts attached. See [`examples/fallback_request.py`](examples/fallback_request.py).

## Observability

Each generation emits one structured record through the standard `logging` logger `resilient_ai_runtime.execution` (level INFO): request ID, policy, provider/model, attempts, latency, token usage, estimated cost, status and timestamp. **Prompts, system prompts, generated text and secrets are not logged.** Enable it with `logging.basicConfig(level=logging.INFO)`. Logging failures never fail a generation.

## Benchmark note

One controlled run on an Intel i7-8650U, 16 GB RAM, CPU-only machine (10 runs each):

- `qwen3:4b` via Ollama: p50 ~31 s
- `mistral-small-latest` (cloud): p50 ~416 ms

This is **environment-specific** and **not a universal model ranking**; it says nothing about relative model quality. Local performance depends heavily on hardware and model, cloud latency on network and provider state. Details and method: [`docs/validation.md`](docs/validation.md). Reproduce with [`benchmarks/`](benchmarks).

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

The default suite is offline and needs no credentials. Real-provider tests are opt-in:

```bash
RUN_OLLAMA_INTEGRATION=1 OLLAMA_INTEGRATION_MODEL=qwen3:4b pytest tests/integration/test_ollama_integration.py
RUN_CLOUD_INTEGRATION=1 CLOUD_INTEGRATION_PROVIDER=mistral CLOUD_INTEGRATION_MODEL=mistral-small-latest \
  MISTRAL_API_KEY=... pytest tests/integration/test_cloud_integration.py   # may consume API credits
```

## Limitations

- V0 validated / pre-production; no full production deployment has been done
- No production SLA and no large-scale load testing
- No complete security audit
- Cost telemetry is best-effort (needs configured per-token pricing and provider-reported usage; local inference is not assumed free)
- Provider behavior, model IDs and pricing can change
- Local performance depends heavily on hardware and model
- Text generation only; the Anthropic adapter uses a fixed 1024-token output limit

## License

[MIT](LICENSE)
