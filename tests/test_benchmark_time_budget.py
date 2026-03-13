"""Tests for free-tier benchmark time-budget behavior."""

from __future__ import annotations

from pathlib import Path

from autolabel.benchmark.baselines import BaselineRunner
from autolabel.benchmark.report import generate_report
from autolabel.logging.progress import ProgressDisplay
from tests.conftest import MockLLMProvider


class FakeClock:
    """Deterministic clock for budget-exhaustion tests."""

    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)
        self._last = values[-1] if values else 0.0

    def __call__(self) -> float:
        try:
            self._last = next(self._values)
        except StopIteration:
            pass
        return self._last


class TimeoutOnCallProvider(MockLLMProvider):
    """Mock provider that raises TimeoutError on a configured call index."""

    def __init__(self, timeout_call_index: int) -> None:
        super().__init__(responses=["Air Canada"])
        self.timeout_call_index = timeout_call_index

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        request_timeout_seconds: float | None = None,
    ):
        if self._call_idx == self.timeout_call_index:
            self._call_idx += 1
            raise TimeoutError("request timed out")
        return super().generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout_seconds=request_timeout_seconds,
        )


def test_baseline_runner_without_budget_preserves_llm_baselines(sample_dataset):
    runner = BaselineRunner(sample_dataset, provider=MockLLMProvider(responses=["Air Canada"]))

    results = runner.run_all()

    assert [row["method"] for row in results] == [
        "Random",
        "Majority Class",
        "TF-IDF + LogReg",
        "Zero-shot LLM",
        "Few-shot LLM",
    ]
    assert results[3]["status"] == "completed"
    assert results[3]["evaluated_examples"] == len(sample_dataset.test_texts)
    assert results[4]["status"] == "completed"


def test_budget_exhausted_before_zero_shot_marks_llm_rows_skipped(sample_dataset):
    runner = BaselineRunner(
        sample_dataset,
        provider=MockLLMProvider(responses=["Air Canada"]),
        llm_deadline_s=0.0,
        clock=lambda: 1.0,
    )

    results = runner.run_all()

    assert results[3]["status"] == "skipped_budget"
    assert results[3]["evaluated_examples"] == 0
    assert results[4]["status"] == "skipped_budget"
    assert results[4]["evaluated_examples"] == 0


def test_budget_exhausted_mid_zero_shot_marks_partial_timeout(sample_dataset):
    clock = FakeClock([0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.6, 0.6])
    runner = BaselineRunner(
        sample_dataset,
        provider=MockLLMProvider(responses=["Air Canada"]),
        llm_deadline_s=0.5,
        clock=clock,
    )

    results = runner.run_all()

    assert results[3]["status"] == "timed_out"
    assert results[3]["evaluated_examples"] == 1
    assert "f1" not in results[3]
    assert results[4]["status"] == "skipped_budget"


def test_few_shot_uses_remaining_shared_budget(sample_dataset):
    clock = FakeClock([0.0, 0.0, 0.0, 0.1, 0.2, 0.95, 1.1, 1.1])
    runner = BaselineRunner(
        sample_dataset,
        provider=MockLLMProvider(responses=["Air Canada"]),
        llm_deadline_s=1.0,
        clock=clock,
    )

    results = runner.run_all()

    assert results[3]["status"] == "completed"
    assert results[4]["status"] == "skipped_budget"
    assert results[4]["evaluated_examples"] == 0


def test_request_timeout_marks_zero_shot_timed_out(sample_dataset):
    runner = BaselineRunner(
        sample_dataset,
        provider=TimeoutOnCallProvider(timeout_call_index=1),
    )

    results = runner.run_all()

    assert results[3]["status"] == "timed_out"
    assert results[3]["evaluated_examples"] == 1


def test_generate_report_renders_incomplete_rows_honestly(tmp_path: Path):
    output_path = tmp_path / "benchmark.md"
    report = generate_report(
        {
            "airline_tweets": [
                {"method": "Random", "f1": 0.1, "accuracy": 0.1, "coverage": 1.0},
                {
                    "method": "Zero-shot LLM",
                    "status": "timed_out",
                    "evaluated_examples": 42,
                    "total_examples": 500,
                },
                {
                    "method": "Few-shot LLM",
                    "status": "skipped_budget",
                    "evaluated_examples": 0,
                    "total_examples": 500,
                },
            ]
        },
        output_path,
    )

    assert "TIMEOUT (42/500)" in report
    assert "SKIPPED (budget exhausted)" in report


def test_progress_display_handles_incomplete_rows():
    display = ProgressDisplay()

    display.print_benchmark_table(
        [
            {"method": "Random", "f1": 0.1, "accuracy": 0.1, "coverage": 1.0},
            {
                "method": "Zero-shot LLM",
                "status": "timed_out",
                "evaluated_examples": 2,
                "total_examples": 3,
            },
        ]
    )
