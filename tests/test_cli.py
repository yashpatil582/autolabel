"""Tests for the CLI."""

from __future__ import annotations

from click.testing import CliRunner

from autolabel.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "AutoLabel" in result.output


def test_cli_run_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output


def test_cli_benchmark_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--llm-time-budget-minutes" in result.output
    assert "--llm-request-timeout-seconds" in result.output


def test_cli_benchmark_passes_budget_flags(monkeypatch):
    import autolabel.benchmark.runner as benchmark_runner_module
    import autolabel.llm as llm_module

    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, provider, config=None) -> None:
            captured["provider"] = provider

        def run(self, **kwargs):
            captured.update(kwargs)
            return {}

    monkeypatch.setattr(benchmark_runner_module, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(llm_module, "get_provider", lambda **kwargs: object())

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "benchmark",
            "-d",
            "airline_tweets",
            "-p",
            "groq",
            "--llm-time-budget-minutes",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert captured["llm_time_budget_minutes"] == 10.0
    assert captured["llm_request_timeout_seconds"] == 20.0


def test_cli_evaluate_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0


def test_cli_cost_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "--help"])
    assert result.exit_code == 0
