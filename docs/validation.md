# Validation and benchmark record (V0)

This document records a controlled Runtime v0 validation pass. It is a
technical record of observed behavior on one environment, not a provider
comparison, capacity guarantee, or production-readiness claim.

## Scope validated

- Local inference through Ollama with `qwen3:4b`.
- Cloud inference through Mistral with `mistral-small-latest`.
- Retry and policy-bounded fallback from an unavailable Ollama provider to
  Mistral.
- Structured execution records, including provider/model, attempts, latency,
  tokens, estimated cost, status, request ID and timestamp.
- Provider/model timeout resolution and the dependency-free benchmark harness.

The benchmark configuration disables retry and fallback so that a benchmark run
measures one selected candidate. The standard Runtime configuration retains its
resilience settings.

## Environment

- CPU: Intel Core i7-8650U (4 cores / 8 threads)
- Memory: 16 GB RAM
- Local inference: CPU-only Ollama
- Local model: `qwen3:4b`
- Cloud model: `mistral-small-latest`

Latency results are specific to this machine, model, prompt, context, service
state and network/infrastructure conditions. Local and cloud results are not a
measure of relative model quality.

## Observed benchmark results

All latency values are milliseconds. Each benchmark used ten runs and completed
successfully ten times.

| Path | p50 | Mean | p95 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Runtime local, Ollama / `qwen3:4b` | 31,034.990 | 29,523.210 | 48,901.968 | 7,122.629 | 53,839.962 |
| Runtime cloud, Mistral / `mistral-small-latest` | 415.871 | 447.067 | 714.782 | 282.477 | 851.089 |
| Direct Ollama, `qwen3:4b`, `think: false` | 30,974.814 | 30,304.742 | 43,674.291 | 7,299.528 | 46,078.764 |

The Runtime local benchmark used `retries_per_candidate: 0` and `fallback:
false`. The direct Ollama benchmark used `think: false` to control Qwen3
reasoning output; it is deliberately a separate provider-native measurement.

For the direct Ollama runs, the observed means were:

| Native Ollama metric | Mean |
| --- | ---: |
| Evaluation throughput | 6.464 tokens/s |
| Evaluation duration | 28,812.272 ms |
| Model load duration | 1,245.069 ms |
| Total duration | 30,296.101 ms |

The benchmark harness computes p95 with Python's inclusive percentile method.
Successful-run latency statistics exclude failed runs; failures remain recorded
separately with their duration and error data.

## Controlled fallback scenario

Ollama was made unavailable with `retries_per_candidate: 1`. The Runtime
recorded the following normalized sequence:

1. Ollama — `provider_unavailable`
2. Ollama — `provider_unavailable`
3. Mistral — success

The final provider was `mistral` with model `mistral-small-latest`. Observed
total latency was 463 ms, input tokens were 21, output tokens were 3, and the
estimated cost was `0.000006 USD`.

This validates configured retry and fallback behavior for that scenario; it
does not predict outage behavior, latency or cost under other conditions.

## Structured observability

The real execution record contained the final provider/model, attempt history,
latency, token counts, estimated cost, status, request ID and timestamp. It did
not include the prompt, generated response or credentials.

## Reproduction boundaries

Use `benchmarks/runtime.benchmark.example.yaml` with
`benchmarks/benchmark_runtime.py` to reproduce a controlled Runtime benchmark.
The direct Ollama mode accepts `--ollama-think false` and reports native Ollama
durations without adding provider-specific fields to `GenerationResponse`.

These figures should be interpreted as environment-specific observed latency.
They were collected to validate Runtime behavior and measurement tooling, not to
declare one provider or execution mode superior to another.
