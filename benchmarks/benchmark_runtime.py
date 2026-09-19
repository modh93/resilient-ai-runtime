"""Run reproducible Runtime or direct Ollama benchmarks without dependencies.

The generic path measures normalized ``AIRuntime`` responses. The optional
direct-Ollama path is deliberately separate: it can set ``think: false`` and
reports Ollama-only durations without extending the Runtime response contract.
The p95 uses Python's inclusive percentile method, not an index approximation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.request import Request, urlopen

from resilient_ai_runtime import AIRuntime


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a benchmark and return nonzero when at least one run fails."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.ollama_direct:
        if not args.ollama_model:
            parser.error("--ollama-model is required with --ollama-direct")
        report = benchmark_ollama_direct(
            model=args.ollama_model,
            prompt=args.prompt,
            runs=args.runs,
            base_url=args.ollama_base_url,
            timeout_s=args.ollama_timeout_s,
            think=args.ollama_think == "true",
        )
    else:
        if not args.config or not args.policy:
            parser.error("--config and --policy are required for a Runtime benchmark")
        report = benchmark_runtime(
            AIRuntime.from_config(args.config),
            policy=args.policy,
            prompt=args.prompt,
            runs=args.runs,
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if report["success_count"] == args.runs else 1


def benchmark_runtime(
    runtime: AIRuntime,
    *,
    policy: str,
    prompt: str,
    runs: int,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Measure normalized Runtime calls while retaining individual failures."""

    records: list[dict[str, Any]] = []
    for index in range(1, runs + 1):
        started_at = clock()
        try:
            response = runtime.generate(prompt=prompt, policy=policy)
        except Exception as error:
            records.append(
                _failure_record(index, _elapsed_ms(started_at, clock()), error)
            )
            continue
        records.append(
            {
                "run": index,
                "status": "success",
                "duration_ms": _elapsed_ms(started_at, clock()),
                "provider": response.provider,
                "model": response.model,
                "runtime_latency_ms": response.latency_ms,
                "attempts": [_attempt_dict(attempt) for attempt in response.attempts],
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "estimated_cost_usd": response.estimated_cost_usd,
            }
        )
    return _report("runtime", records)


def benchmark_ollama_direct(
    *,
    model: str,
    prompt: str,
    runs: int,
    base_url: str,
    timeout_s: float,
    think: bool,
    clock: Callable[[], float] = perf_counter,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Benchmark Ollama directly, including opt-in thinking and native timings."""

    records: list[dict[str, Any]] = []
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    for index in range(1, runs + 1):
        started_at = clock()
        try:
            request = Request(
                endpoint,
                data=json.dumps(
                    {
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "think": think,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with opener(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            records.append(
                _ollama_success_record(
                    index, _elapsed_ms(started_at, clock()), model, payload
                )
            )
        except Exception as error:
            records.append(
                _failure_record(index, _elapsed_ms(started_at, clock()), error)
            )
    return _report("ollama-direct", records, think=think)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--policy")
    parser.add_argument("--runs", type=_positive_int, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ollama-direct", action="store_true")
    parser.add_argument(
        "--ollama-base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-timeout-s", type=_positive_float, default=90.0)
    parser.add_argument("--ollama-think", choices=("true", "false"), default="false")
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _elapsed_ms(started_at: float, finished_at: float) -> float:
    return round((finished_at - started_at) * 1_000, 3)


def _attempt_dict(attempt: Any) -> dict[str, Any]:
    return {
        "provider": attempt.provider,
        "model": attempt.model,
        "status": attempt.status,
        "latency_ms": attempt.latency_ms,
        "error_code": attempt.error_code,
    }


def _failure_record(index: int, duration_ms: float, error: Exception) -> dict[str, Any]:
    attempts = getattr(error, "attempts", ())
    return {
        "run": index,
        "status": "failed",
        "duration_ms": duration_ms,
        "error_type": type(error).__name__,
        "error": str(error),
        "attempts": [_attempt_dict(attempt) for attempt in attempts],
    }


def _ollama_success_record(
    index: int, duration_ms: float, model: str, payload: Any
) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
        raise ValueError("Ollama response is missing generated text")
    metrics = {
        "eval_count": _optional_int(payload.get("eval_count")),
        "eval_duration_ms": _nanoseconds_to_ms(payload.get("eval_duration")),
        "load_duration_ms": _nanoseconds_to_ms(payload.get("load_duration")),
        "total_duration_ms": _nanoseconds_to_ms(payload.get("total_duration")),
    }
    if metrics["eval_count"] is not None and metrics["eval_duration_ms"]:
        metrics["eval_tokens_per_second"] = round(
            metrics["eval_count"] / (metrics["eval_duration_ms"] / 1_000), 3
        )
    return {
        "run": index,
        "status": "success",
        "duration_ms": duration_ms,
        "provider": "ollama",
        "model": model,
        "attempts": [
            {
                "provider": "ollama",
                "model": model,
                "status": "success",
                "latency_ms": duration_ms,
                "error_code": None,
            }
        ],
        "input_tokens": _optional_int(payload.get("prompt_eval_count")),
        "output_tokens": metrics["eval_count"],
        "estimated_cost_usd": None,
        "ollama_metrics": metrics,
    }


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _nanoseconds_to_ms(value: Any) -> float | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return round(value / 1_000_000, 3)


def _report(
    kind: str, records: list[dict[str, Any]], **metadata: Any
) -> dict[str, Any]:
    successes = [record for record in records if record["status"] == "success"]
    latencies = [record["duration_ms"] for record in successes]
    report: dict[str, Any] = {
        "benchmark": kind,
        **metadata,
        "run_count": len(records),
        "success_count": len(successes),
        "failure_count": len(records) - len(successes),
        "success_rate": len(successes) / len(records) if records else 0.0,
        "latency_ms": _latency_summary(latencies),
        "runs": records,
    }
    if kind == "ollama-direct":
        report["ollama_metrics"] = _ollama_summary(successes)
    return report


def _latency_summary(latencies: list[float]) -> dict[str, float | None]:
    if not latencies:
        return {"mean": None, "median_p50": None, "p95": None, "min": None, "max": None}
    return {
        "mean": round(statistics.fmean(latencies), 3),
        "median_p50": round(statistics.median(latencies), 3),
        "p95": round(_p95_inclusive(latencies), 3),
        "min": min(latencies),
        "max": max(latencies),
    }


def _p95_inclusive(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def _ollama_summary(records: list[dict[str, Any]]) -> dict[str, float | None]:
    metrics = [record.get("ollama_metrics", {}) for record in records]
    return {
        "mean_eval_tokens_per_second": _mean_metric(metrics, "eval_tokens_per_second"),
        "mean_eval_duration_ms": _mean_metric(metrics, "eval_duration_ms"),
        "mean_load_duration_ms": _mean_metric(metrics, "load_duration_ms"),
        "mean_total_duration_ms": _mean_metric(metrics, "total_duration_ms"),
    }


def _mean_metric(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [metric[key] for metric in metrics if metric.get(key) is not None]
    return round(statistics.fmean(values), 3) if values else None


if __name__ == "__main__":
    sys.exit(main())
