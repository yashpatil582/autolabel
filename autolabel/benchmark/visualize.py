"""Publication-quality experiment visualizations for AutoLabel."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / "autolabel-mpl"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
if "XDG_CACHE_HOME" not in os.environ:
    cache_dir = Path(tempfile.gettempdir()) / "autolabel-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from autolabel.config import AutoLabelConfig

COLORS = {
    "keep": "#2ecc71",
    "discard": "#e74c3c",
    "ratchet": "#3498db",
    "coverage": "#3498db",
    "accuracy": "#e67e22",
    "baseline": "#7f8c8d",
    "ours": "#2ecc71",
    "bars": "#9bb8d3",
    "text": "#2c3e50",
    "grid": "#dfe6e9",
}
CHART_FILENAMES = {
    "f1": "f1_trajectory.png",
    "baseline": "baseline_comparison.png",
    "strategy": "strategy_analysis.png",
    "coverage": "coverage_accuracy.png",
    "efficiency": "lf_efficiency.png",
}
DPI = 150
REQUIRED_FILES = ("meta.json", "final_summary.json", "experiment.jsonl")


@dataclass(slots=True)
class ExperimentData:
    """Parsed experiment artifacts from a single run directory."""

    experiment_dir: Path
    meta: dict[str, Any]
    summary: dict[str, Any]
    iterations: list[dict[str, Any]]

    @classmethod
    def load(cls, experiment_dir: str | Path) -> ExperimentData:
        """Load and validate experiment metadata, summary, and iteration logs."""
        exp_dir = Path(experiment_dir)
        missing = [name for name in REQUIRED_FILES if not (exp_dir / name).exists()]
        if missing:
            raise ValueError(
                f"Experiment data is incomplete in {exp_dir}: missing {', '.join(missing)}."
            )

        try:
            meta = json.loads((exp_dir / "meta.json").read_text())
            summary = json.loads((exp_dir / "final_summary.json").read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse experiment metadata in {exp_dir}: {exc}") from exc

        try:
            raw_lines = (exp_dir / "experiment.jsonl").read_text().splitlines()
            iterations = [json.loads(line) for line in raw_lines if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse experiment log in {exp_dir}: {exc}") from exc

        if not iterations:
            raise ValueError(f"Experiment log is empty in {exp_dir}.")

        return cls(
            experiment_dir=exp_dir,
            meta=meta,
            summary=summary,
            iterations=sorted(iterations, key=lambda entry: int(entry.get("iteration", 0))),
        )

    @property
    def dataset_name(self) -> str:
        return str(self.meta.get("dataset", self.experiment_dir.name))

    @property
    def task(self) -> str:
        return str(self.meta.get("task", ""))

    @property
    def best_dev_f1(self) -> float:
        return float(self.summary.get("best_dev_f1", 0.0))

    @property
    def test_f1(self) -> float:
        return float(self.summary.get("test_f1", 0.0))

    @property
    def total_generated(self) -> int:
        return int(self.summary.get("total_generated", 0))

    @property
    def active_lfs(self) -> int:
        return int(self.summary.get("active_lfs", 0))

    @property
    def iteration_numbers(self) -> list[int]:
        return [int(entry.get("iteration", idx + 1)) for idx, entry in enumerate(self.iterations)]

    def ratcheted_f1(self) -> list[float]:
        """Best-dev-F1 trajectory after the keep/discard ratchet."""
        best = 0.0
        trajectory: list[float] = []
        for entry in self.iterations:
            best = max(best, float(entry.get("f1_after", best)))
            trajectory.append(best)
        return trajectory

    def active_state_series(self) -> tuple[list[float], list[float]]:
        """Coverage and accuracy of the currently active LF set over time.

        The experiment log stores candidate metrics for discard iterations too, so
        we forward-fill the last kept values to reflect the active model state.
        """
        coverage = 0.0
        accuracy = 0.0
        coverage_series: list[float] = []
        accuracy_series: list[float] = []

        for entry in self.iterations:
            if entry.get("kept"):
                coverage = float(entry.get("coverage", coverage))
                accuracy = float(entry.get("accuracy", accuracy))
            coverage_series.append(coverage)
            accuracy_series.append(accuracy)

        return coverage_series, accuracy_series


def plot_f1_trajectory(
    experiment: ExperimentData,
    output_path: str | Path,
    baseline_f1: float | None = None,
) -> Path:
    """Plot keep/discard decisions and the ratcheted best-dev-F1 trajectory."""
    output = Path(output_path)
    iterations = experiment.iteration_numbers
    f1_values = [float(entry.get("f1_after", 0.0)) for entry in experiment.iterations]
    ratchet = experiment.ratcheted_f1()
    keep_points = [entry.get("kept", False) for entry in experiment.iterations]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for iteration, f1_value, kept in zip(iterations, f1_values, keep_points):
        ax.scatter(
            iteration,
            f1_value,
            c=COLORS["keep"] if kept else COLORS["discard"],
            marker="o" if kept else "x",
            s=80,
            linewidths=2 if not kept else 0,
            zorder=3,
        )

    ax.plot(iterations, ratchet, color=COLORS["ratchet"], linewidth=2.5, label="Best dev F1")
    ax.fill_between(iterations, 0, ratchet, color=COLORS["ratchet"], alpha=0.12)

    if baseline_f1 is not None:
        ax.axhline(
            baseline_f1,
            color=COLORS["baseline"],
            linestyle="--",
            linewidth=1.8,
            label=f"TF-IDF baseline ({baseline_f1:.3f})",
        )

    ax.set_title(
        f"{experiment.dataset_name}: Dev F1 Trajectory\n"
        f"Best Dev F1 = {experiment.best_dev_f1:.3f} | Test F1 = {experiment.test_f1:.3f}",
        color=COLORS["text"],
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.7)

    legend_handles = [
        Line2D([], [], color=COLORS["ratchet"], linewidth=2.5, label="Best dev F1"),
        Line2D([], [], color=COLORS["keep"], marker="o", linestyle="", markersize=8, label="KEEP"),
        Line2D(
            [], [], color=COLORS["discard"], marker="x", linestyle="", markersize=8, label="DISCARD"
        ),
    ]
    if baseline_f1 is not None:
        legend_handles.append(
            Line2D(
                [],
                [],
                color=COLORS["baseline"],
                linestyle="--",
                linewidth=1.8,
                label=f"TF-IDF baseline ({baseline_f1:.3f})",
            )
        )
    ax.legend(handles=legend_handles, loc="lower right")

    _save_figure(fig, output)
    return output


def plot_baseline_comparison(
    experiment: ExperimentData,
    output_path: str | Path,
    measured_results: list[dict[str, Any]],
) -> Path:
    """Plot a measured baseline comparison for the experiment dataset."""
    if not measured_results:
        raise ValueError(
            "Measured benchmark results are required for the baseline comparison chart."
        )

    output = Path(output_path)
    sorted_results = sorted(measured_results, key=lambda item: float(item["f1"]))
    labels = [str(item["method"]) for item in sorted_results]
    f1_scores = [float(item["f1"]) for item in sorted_results]
    colors = [
        COLORS["ours"] if label == "AutoLabel (Ours)" else COLORS["baseline"] for label in labels
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bars = ax.barh(labels, f1_scores, color=colors, edgecolor="white", linewidth=1.5)
    ax.invert_yaxis()

    for bar, score in zip(bars, f1_scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center")

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 Score")
    ax.set_title(f"{experiment.dataset_name}: Measured Baseline Comparison", color=COLORS["text"])
    ax.grid(True, axis="x", color=COLORS["grid"], alpha=0.7)

    _save_figure(fig, output)
    return output


def plot_strategy_analysis(experiment: ExperimentData, output_path: str | Path) -> Path:
    """Plot per-strategy usage, kept counts, and keep rate."""
    output = Path(output_path)
    strategy_stats: dict[str, dict[str, int]] = {}
    for entry in experiment.iterations:
        strategy = str(entry.get("strategy", "unknown"))
        stats = strategy_stats.setdefault(strategy, {"tried": 0, "kept": 0})
        stats["tried"] += 1
        if entry.get("kept"):
            stats["kept"] += 1

    strategies = sorted(strategy_stats)
    tried_counts = [strategy_stats[strategy]["tried"] for strategy in strategies]
    kept_counts = [strategy_stats[strategy]["kept"] for strategy in strategies]
    keep_rates = [kept / tried if tried else 0.0 for kept, tried in zip(kept_counts, tried_counts)]
    x = np.arange(len(strategies))

    fig, (ax_tried, ax_rate) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    ax_tried.bar(x - 0.18, tried_counts, width=0.36, color=COLORS["ratchet"], label="Tried")
    ax_tried.bar(x + 0.18, kept_counts, width=0.36, color=COLORS["keep"], label="Kept")
    ax_tried.set_xticks(x, strategies, rotation=35, ha="right")
    ax_tried.set_ylabel("Count")
    ax_tried.set_title("Tried vs Kept")
    ax_tried.legend()
    ax_tried.grid(True, axis="y", color=COLORS["grid"], alpha=0.7)

    ax_rate.bar(strategies, keep_rates, color=COLORS["discard"], alpha=0.85)
    ax_rate.set_ylim(0, 1.0)
    ax_rate.set_ylabel("Keep Rate")
    ax_rate.set_title("Keep Rate by Strategy")
    ax_rate.tick_params(axis="x", labelrotation=35)
    ax_rate.grid(True, axis="y", color=COLORS["grid"], alpha=0.7)

    fig.suptitle(f"{experiment.dataset_name}: Strategy Analysis", color=COLORS["text"])
    _save_figure(fig, output)
    return output


def plot_coverage_accuracy(experiment: ExperimentData, output_path: str | Path) -> Path:
    """Plot active-state coverage and conditional accuracy across iterations."""
    output = Path(output_path)
    iterations = experiment.iteration_numbers
    coverage, accuracy = experiment.active_state_series()

    fig, ax_coverage = plt.subplots(figsize=(12, 6.5))
    ax_accuracy = ax_coverage.twinx()

    ax_coverage.plot(
        iterations, coverage, color=COLORS["coverage"], linewidth=2.4, label="Coverage"
    )
    ax_accuracy.plot(
        iterations,
        accuracy,
        color=COLORS["accuracy"],
        linewidth=2.4,
        label="Accuracy on covered examples",
    )

    ax_coverage.set_xlabel("Iteration")
    ax_coverage.set_ylabel("Coverage", color=COLORS["coverage"])
    ax_accuracy.set_ylabel("Accuracy on covered examples", color=COLORS["accuracy"])
    ax_coverage.set_ylim(0, 1.0)
    ax_accuracy.set_ylim(0, 1.0)
    ax_coverage.set_title(
        f"{experiment.dataset_name}: Active-State Coverage and Accuracy",
        color=COLORS["text"],
    )
    ax_coverage.grid(True, axis="y", color=COLORS["grid"], alpha=0.7)

    lines = ax_coverage.get_lines() + ax_accuracy.get_lines()
    ax_coverage.legend(lines, [line.get_label() for line in lines], loc="lower right")

    _save_figure(fig, output)
    return output


def plot_lf_efficiency(experiment: ExperimentData, output_path: str | Path) -> Path:
    """Plot LF growth against the ratcheted dev-F1 trajectory."""
    output = Path(output_path)
    iterations = experiment.iteration_numbers
    active_lf_counts = [int(entry.get("active_lf_count", 0)) for entry in experiment.iterations]
    ratchet = experiment.ratcheted_f1()
    efficiency = (
        experiment.active_lfs / experiment.total_generated if experiment.total_generated else 0.0
    )

    fig, ax_lfs = plt.subplots(figsize=(12, 6.5))
    ax_f1 = ax_lfs.twinx()

    ax_lfs.bar(iterations, active_lf_counts, color=COLORS["bars"], alpha=0.85, label="Active LFs")
    ax_f1.plot(iterations, ratchet, color=COLORS["ratchet"], linewidth=2.5, label="Best dev F1")

    ax_lfs.set_xlabel("Iteration")
    ax_lfs.set_ylabel("Active LF Count")
    ax_f1.set_ylabel("Best Dev F1", color=COLORS["ratchet"])
    ax_f1.set_ylim(0, 1.0)
    ax_lfs.set_title(f"{experiment.dataset_name}: LF Efficiency", color=COLORS["text"])
    ax_lfs.grid(True, axis="y", color=COLORS["grid"], alpha=0.7)

    annotation = (
        f"Generated: {experiment.total_generated}\n"
        f"Active: {experiment.active_lfs}\n"
        f"Efficiency: {efficiency:.1%}"
    )
    ax_lfs.text(
        0.02,
        0.98,
        annotation,
        transform=ax_lfs.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": COLORS["grid"],
        },
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["bars"], alpha=0.85, label="Active LFs"),
        Line2D([], [], color=COLORS["ratchet"], linewidth=2.5, label="Best dev F1"),
    ]
    ax_lfs.legend(handles=legend_handles, loc="upper left")

    _save_figure(fig, output)
    return output


def generate_all_charts(
    experiment_dir: str | Path,
    output_dir: str | Path | None = None,
    benchmark_results_path: str | Path | None = None,
) -> tuple[list[Path], list[str]]:
    """Generate all available charts for a completed experiment run."""
    experiment = ExperimentData.load(experiment_dir)
    destination = Path(output_dir) if output_dir is not None else experiment.experiment_dir
    destination.mkdir(parents=True, exist_ok=True)

    created_paths: list[Path] = []
    skipped_messages: list[str] = []

    benchmark_path, benchmark_results = _resolve_benchmark_results(
        experiment.experiment_dir,
        benchmark_results_path,
    )
    measured_baselines = _dataset_benchmark_results(experiment, benchmark_results)
    baseline_f1 = _tfidf_baseline_f1(measured_baselines)

    created_paths.append(
        plot_f1_trajectory(experiment, destination / CHART_FILENAMES["f1"], baseline_f1)
    )
    created_paths.append(
        plot_strategy_analysis(experiment, destination / CHART_FILENAMES["strategy"])
    )
    created_paths.append(
        plot_coverage_accuracy(experiment, destination / CHART_FILENAMES["coverage"])
    )
    created_paths.append(
        plot_lf_efficiency(experiment, destination / CHART_FILENAMES["efficiency"])
    )

    if measured_baselines:
        created_paths.append(
            plot_baseline_comparison(
                experiment,
                destination / CHART_FILENAMES["baseline"],
                measured_baselines,
            )
        )
    else:
        source = benchmark_path if benchmark_path is not None else "no benchmark results were found"
        skipped_messages.append(
            f"Skipped {CHART_FILENAMES['baseline']}: no measured benchmark data for "
            f"{experiment.dataset_name} ({source})."
        )

    return created_paths, skipped_messages


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _resolve_benchmark_results(
    experiment_dir: Path,
    benchmark_results_path: str | Path | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    if benchmark_results_path is not None:
        explicit_path = Path(benchmark_results_path)
        return explicit_path, _load_benchmark_results(explicit_path)

    for candidate in _benchmark_candidates(experiment_dir):
        results = _load_benchmark_results(candidate)
        if results is not None:
            return candidate, results

    return None, None


def _benchmark_candidates(experiment_dir: Path) -> list[Path]:
    config = AutoLabelConfig()
    candidates = [
        experiment_dir.parent / "benchmark" / "results.json",
        config.experiments_dir / "benchmark" / "results.json",
    ]
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique_candidates.append(candidate)
            seen.add(resolved)
    return unique_candidates


def _load_benchmark_results(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _dataset_benchmark_results(
    experiment: ExperimentData,
    benchmark_results: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not benchmark_results:
        return []

    dataset_results = benchmark_results.get(experiment.dataset_name)
    if not isinstance(dataset_results, list):
        return []

    normalized: list[dict[str, Any]] = []
    for entry in dataset_results:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if status not in (None, "completed"):
            continue
        method = str(entry.get("method", "")).strip()
        if not method or _is_autolabel_method(method):
            continue
        try:
            f1 = float(entry["f1"])
        except (KeyError, TypeError, ValueError):
            continue
        normalized.append({"method": method, "f1": f1})

    if not normalized:
        return []

    normalized.append({"method": "AutoLabel (Ours)", "f1": experiment.test_f1})
    return normalized


def _is_autolabel_method(method: str) -> bool:
    return method.strip().lower().startswith("autolabel")


def _tfidf_baseline_f1(measured_results: list[dict[str, Any]]) -> float | None:
    for entry in measured_results:
        method = str(entry.get("method", "")).lower()
        if "tf-idf" in method and "logreg" in method:
            return float(entry["f1"])
    return None
