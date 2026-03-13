"""Tests for experiment visualization utilities and CLI wiring."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from autolabel.benchmark.visualize import (
    ExperimentData,
    generate_all_charts,
    plot_baseline_comparison,
    plot_coverage_accuracy,
    plot_f1_trajectory,
    plot_lf_efficiency,
    plot_strategy_analysis,
)
from autolabel.cli import cli


@pytest.fixture
def mock_experiment_dir(tmp_path: Path) -> Path:
    """Create a synthetic experiment directory with 5 logged iterations."""
    experiment_dir = tmp_path / "experiments" / "mock_run"
    experiment_dir.mkdir(parents=True)

    meta = {
        "dataset": "airline_tweets",
        "task": "Extract the airline mentioned in this tweet",
        "provider": "MockProvider",
        "max_iterations": 5,
        "label_model": "majority",
        "config": {"min_improvement": 0.005, "lfs_per_iteration": 5},
        "start_time": 0.0,
    }
    summary = {
        "best_dev_f1": 0.47,
        "test_f1": 0.44,
        "test_accuracy": 0.62,
        "test_coverage": 0.71,
        "total_iterations": 5,
        "active_lfs": 8,
        "total_generated": 13,
        "total_cost": 0,
        "f1_trajectory": [0.18, 0.18, 0.31, 0.31, 0.47],
        "total_time_s": 12.5,
    }
    iterations = [
        {
            "iteration": 1,
            "timestamp": 1.0,
            "elapsed_s": 1.0,
            "strategy": "keyword",
            "target_label": "Air Canada",
            "new_lfs_generated": 3,
            "new_lfs_valid": 2,
            "f1_before": 0.0,
            "f1_after": 0.18,
            "f1_delta": 0.18,
            "kept": True,
            "active_lf_count": 2,
            "coverage": 0.25,
            "accuracy": 0.72,
            "label_model_type": "majority",
            "error": None,
        },
        {
            "iteration": 2,
            "timestamp": 2.0,
            "elapsed_s": 2.0,
            "strategy": "regex",
            "target_label": "United Airlines",
            "new_lfs_generated": 2,
            "new_lfs_valid": 1,
            "f1_before": 0.18,
            "f1_after": 0.18,
            "f1_delta": 0.0,
            "kept": False,
            "active_lf_count": 2,
            "coverage": 0.91,
            "accuracy": 0.11,
            "label_model_type": "majority",
            "error": None,
        },
        {
            "iteration": 3,
            "timestamp": 3.0,
            "elapsed_s": 3.0,
            "strategy": "regex",
            "target_label": "Delta Air Lines",
            "new_lfs_generated": 4,
            "new_lfs_valid": 2,
            "f1_before": 0.18,
            "f1_after": 0.31,
            "f1_delta": 0.13,
            "kept": True,
            "active_lf_count": 4,
            "coverage": 0.42,
            "accuracy": 0.76,
            "label_model_type": "majority",
            "error": None,
        },
        {
            "iteration": 4,
            "timestamp": 4.0,
            "elapsed_s": 4.0,
            "strategy": "semantic",
            "target_label": "Delta Air Lines",
            "new_lfs_generated": 1,
            "new_lfs_valid": 1,
            "f1_before": 0.31,
            "f1_after": 0.31,
            "f1_delta": 0.0,
            "kept": False,
            "active_lf_count": 4,
            "coverage": 0.88,
            "accuracy": 0.14,
            "label_model_type": "majority",
            "error": None,
        },
        {
            "iteration": 5,
            "timestamp": 5.0,
            "elapsed_s": 5.0,
            "strategy": "fuzzy",
            "target_label": "Air France",
            "new_lfs_generated": 3,
            "new_lfs_valid": 3,
            "f1_before": 0.31,
            "f1_after": 0.47,
            "f1_delta": 0.16,
            "kept": True,
            "active_lf_count": 8,
            "coverage": 0.66,
            "accuracy": 0.81,
            "label_model_type": "majority",
            "error": None,
        },
    ]

    (experiment_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (experiment_dir / "final_summary.json").write_text(json.dumps(summary, indent=2))
    (experiment_dir / "experiment.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in iterations) + "\n"
    )
    return experiment_dir


@pytest.fixture
def mock_benchmark_results(mock_experiment_dir: Path) -> Path:
    """Create measured benchmark results alongside the synthetic experiment."""
    benchmark_dir = mock_experiment_dir.parent / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = benchmark_dir / "results.json"
    results = {
        "airline_tweets": [
            {"method": "Random", "f1": 0.08},
            {"method": "Majority Class", "f1": 0.11},
            {"method": "TF-IDF + LogReg", "f1": 0.31},
            {"method": "Zero-shot LLM", "f1": 0.24},
            {"method": "Few-shot LLM", "f1": 0.28},
        ]
    }
    benchmark_path.write_text(json.dumps(results, indent=2))
    return benchmark_path


def _assert_non_empty_png(path: Path) -> None:
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


def test_experiment_data_loads_correctly(mock_experiment_dir: Path):
    data = ExperimentData.load(mock_experiment_dir)

    assert data.dataset_name == "airline_tweets"
    assert data.task == "Extract the airline mentioned in this tweet"
    assert data.best_dev_f1 == pytest.approx(0.47)
    assert data.test_f1 == pytest.approx(0.44)
    assert data.total_generated == 13
    assert data.active_lfs == 8
    assert data.iteration_numbers == [1, 2, 3, 4, 5]
    assert data.ratcheted_f1() == pytest.approx([0.18, 0.18, 0.31, 0.31, 0.47])
    coverage_series, accuracy_series = data.active_state_series()
    assert coverage_series == pytest.approx([0.25, 0.25, 0.42, 0.42, 0.66])
    assert accuracy_series == pytest.approx([0.72, 0.72, 0.76, 0.76, 0.81])


def test_experiment_data_raises_for_incomplete_dir(tmp_path: Path):
    with pytest.raises(ValueError, match="missing"):
        ExperimentData.load(tmp_path)


@pytest.mark.parametrize(
    ("filename", "plotter"),
    [
        ("f1.png", lambda data, output: plot_f1_trajectory(data, output, baseline_f1=0.31)),
        (
            "baseline.png",
            lambda data, output: plot_baseline_comparison(
                data,
                output,
                [
                    {"method": "Random", "f1": 0.08},
                    {"method": "Majority Class", "f1": 0.11},
                    {"method": "TF-IDF + LogReg", "f1": 0.31},
                    {"method": "Zero-shot LLM", "f1": 0.24},
                    {"method": "Few-shot LLM", "f1": 0.28},
                    {"method": "AutoLabel (Ours)", "f1": data.test_f1},
                ],
            ),
        ),
        ("strategy.png", plot_strategy_analysis),
        ("coverage.png", plot_coverage_accuracy),
        ("efficiency.png", plot_lf_efficiency),
    ],
)
def test_plot_functions_create_pngs(mock_experiment_dir: Path, tmp_path: Path, filename, plotter):
    data = ExperimentData.load(mock_experiment_dir)
    output = tmp_path / filename
    created = plotter(data, output)

    assert created == output
    _assert_non_empty_png(output)


def test_generate_all_charts_skips_baseline_without_results(
    mock_experiment_dir: Path, tmp_path: Path, monkeypatch
):
    # Prevent auto-discovery from finding a real benchmark/results.json on disk
    monkeypatch.setattr(
        "autolabel.benchmark.visualize._benchmark_candidates",
        lambda experiment_dir: [experiment_dir.parent / "benchmark" / "results.json"],
    )
    output_dir = tmp_path / "charts"
    created, skipped = generate_all_charts(mock_experiment_dir, output_dir=output_dir)

    assert len(created) == 4
    assert len(skipped) == 1
    assert "baseline_comparison.png" in skipped[0]
    for path in created:
        _assert_non_empty_png(path)
    assert not (output_dir / "baseline_comparison.png").exists()


def test_generate_all_charts_auto_discovers_benchmark_results(
    mock_experiment_dir: Path,
    mock_benchmark_results: Path,
):
    created, skipped = generate_all_charts(mock_experiment_dir)

    assert len(created) == 5
    assert skipped == []
    assert any(path.name == "baseline_comparison.png" for path in created)


def test_generate_all_charts_ignores_incomplete_benchmark_rows(
    mock_experiment_dir: Path,
    tmp_path: Path,
):
    benchmark_path = tmp_path / "results.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "airline_tweets": [
                    {"method": "Random", "f1": 0.08, "status": "completed"},
                    {"method": "Majority Class", "f1": 0.11, "status": "completed"},
                    {"method": "TF-IDF + LogReg", "f1": 0.31, "status": "completed"},
                    {
                        "method": "Zero-shot LLM",
                        "status": "timed_out",
                        "evaluated_examples": 42,
                        "total_examples": 500,
                    },
                ]
            },
            indent=2,
        )
    )

    created, skipped = generate_all_charts(
        mock_experiment_dir,
        benchmark_results_path=benchmark_path,
    )

    assert skipped == []
    assert any(path.name == "baseline_comparison.png" for path in created)


def test_generate_all_charts_skips_baseline_when_only_incomplete_rows(
    mock_experiment_dir: Path,
    tmp_path: Path,
):
    benchmark_path = tmp_path / "results.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "airline_tweets": [
                    {
                        "method": "Zero-shot LLM",
                        "status": "timed_out",
                        "evaluated_examples": 42,
                        "total_examples": 500,
                    }
                ]
            },
            indent=2,
        )
    )

    created, skipped = generate_all_charts(
        mock_experiment_dir,
        benchmark_results_path=benchmark_path,
    )

    assert len(created) == 4
    assert len(skipped) == 1
    assert "no measured benchmark data" in skipped[0]


def test_cli_visualize_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", "--help"])

    assert result.exit_code == 0
    assert "--benchmark-results" in result.output


def test_cli_visualize_generates_all_charts_with_benchmark_data(
    mock_experiment_dir: Path,
    mock_benchmark_results: Path,
    tmp_path: Path,
):
    runner = CliRunner()
    output_dir = tmp_path / "cli-charts"
    result = runner.invoke(
        cli,
        [
            "visualize",
            str(mock_experiment_dir),
            "--output-dir",
            str(output_dir),
            "--benchmark-results",
            str(mock_benchmark_results),
        ],
    )

    assert result.exit_code == 0
    assert result.output.count("Generated:") == 5
    _assert_non_empty_png(output_dir / "baseline_comparison.png")


def test_cli_visualize_fails_for_incomplete_experiment(tmp_path: Path):
    experiment_dir = tmp_path / "broken"
    experiment_dir.mkdir()
    runner = CliRunner()

    result = runner.invoke(cli, ["visualize", str(experiment_dir)])

    assert result.exit_code != 0
    assert "Experiment data is incomplete" in result.output


def test_cli_visualize_fails_for_nonexistent_experiment_dir():
    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", "/tmp/does-not-exist-autolabel"])

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_cli_visualize_reports_missing_viz_dependency(monkeypatch, mock_experiment_dir: Path):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "autolabel.benchmark.visualize":
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", str(mock_experiment_dir)])

    assert result.exit_code != 0
    assert 'pip install -e ".[viz]"' in result.output
