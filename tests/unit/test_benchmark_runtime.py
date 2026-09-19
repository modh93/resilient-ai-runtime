"""Offline tests for the dependency-free Runtime benchmark harness."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

from resilient_ai_runtime.core import (
    AllCandidatesFailedError,
    AttemptRecord,
    GenerationResponse,
)

_SCRIPT_PATH = Path(__file__).parents[2] / "benchmarks" / "benchmark_runtime.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_runtime", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = benchmark
_SPEC.loader.exec_module(benchmark)


@dataclass
class _Response:
    payload: dict[str, object]

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeRuntime:
    def __init__(self) -> None:
        self.outcomes: list[GenerationResponse | Exception] = [
            GenerationResponse(
                text="ok",
                provider="mistral",
                model="mistral-small-latest",
                latency_ms=12,
                request_id="one",
                input_tokens=4,
                output_tokens=2,
                estimated_cost_usd=0.01,
                attempts=[
                    AttemptRecord(
                        provider="mistral",
                        model="mistral-small-latest",
                        status="success",
                        latency_ms=12,
                    )
                ],
            ),
            AllCandidatesFailedError(
                [
                    AttemptRecord(
                        provider="mistral",
                        model="mistral-small-latest",
                        status="failed",
                        latency_ms=5,
                        error_code="provider_timeout",
                    )
                ]
            ),
        ]

    def generate(self, *, prompt: str, policy: str) -> GenerationResponse:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_runtime_benchmark_keeps_failures_out_of_success_latency_stats() -> None:
    clock = iter([0.0, 0.1, 0.1, 0.4])

    report = benchmark.benchmark_runtime(
        _FakeRuntime(),  # type: ignore[arg-type]
        policy="benchmark-mistral",
        prompt="hello",
        runs=2,
        clock=lambda: next(clock),
    )

    assert report["success_count"] == 1
    assert report["failure_count"] == 1
    assert report["success_rate"] == 0.5
    assert report["latency_ms"] == {
        "mean": 100.0,
        "median_p50": 100.0,
        "p95": 100.0,
        "min": 100.0,
        "max": 100.0,
    }
    assert report["runs"][1]["duration_ms"] == 300.0
    assert report["runs"][1]["attempts"][0]["error_code"] == "provider_timeout"


def test_inclusive_p95_is_interpolated_instead_of_an_index_lookup() -> None:
    assert benchmark._p95_inclusive(list(range(1, 11))) == 9.55


def test_direct_ollama_benchmark_sends_think_false_and_reports_native_metrics() -> None:
    requests: list[dict[str, object]] = []
    clock = iter([0.0, 2.5])

    def opener(request: object, *, timeout: float) -> _Response:
        requests.append(
            {
                "payload": json.loads(request.data.decode("utf-8")),  # type: ignore[attr-defined]
                "timeout": timeout,
            }
        )
        return _Response(
            {
                "response": "ok",
                "prompt_eval_count": 3,
                "eval_count": 10,
                "eval_duration": 2_000_000_000,
                "load_duration": 100_000_000,
                "total_duration": 2_100_000_000,
            }
        )

    report = benchmark.benchmark_ollama_direct(
        model="qwen3:4b",
        prompt="hello",
        runs=1,
        base_url="http://ollama.test",
        timeout_s=90,
        think=False,
        clock=lambda: next(clock),
        opener=opener,
    )

    assert requests == [
        {
            "payload": {
                "model": "qwen3:4b",
                "prompt": "hello",
                "stream": False,
                "think": False,
            },
            "timeout": 90,
        }
    ]
    assert report["ollama_metrics"] == {
        "mean_eval_tokens_per_second": 5.0,
        "mean_eval_duration_ms": 2000.0,
        "mean_load_duration_ms": 100.0,
        "mean_total_duration_ms": 2100.0,
    }


def test_main_can_persist_a_raw_report_without_prompt_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "run_count": 1,
        "success_count": 1,
        "runs": [{"status": "success", "provider": "mistral"}],
    }
    monkeypatch.setattr(benchmark.AIRuntime, "from_config", lambda _: object())
    monkeypatch.setattr(benchmark, "benchmark_runtime", lambda *args, **kwargs: report)
    output = tmp_path / "benchmark.json"

    assert (
        benchmark.main(
            [
                "--config",
                "benchmarks/runtime.benchmark.example.yaml",
                "--policy",
                "benchmark-mistral",
                "--runs",
                "1",
                "--prompt",
                "secret prompt",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert "secret prompt" not in output.read_text(encoding="utf-8")
