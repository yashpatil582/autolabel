"""Tests for the CLI."""

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


def test_cli_evaluate_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0


def test_cli_cost_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["cost", "--help"])
    assert result.exit_code == 0
