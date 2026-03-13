"""Click CLI for AutoLabel."""

from __future__ import annotations

import json
from pathlib import Path

import click

from autolabel.config import AutoLabelConfig


@click.group()
@click.version_option(package_name="autolabel")
def cli() -> None:
    """AutoLabel: Autonomous self-improving data labeling system."""
    pass


@cli.command()
@click.option("-d", "--dataset", required=True, help="Dataset name (e.g. airline_tweets)")
@click.option("-t", "--task", default=None, help="Task description (overrides dataset default)")
@click.option(
    "-l", "--labels", default=None, help="Comma-separated label space (overrides dataset default)"
)
@click.option(
    "-p", "--provider", default="anthropic", help="LLM provider: anthropic, openai, groq, ollama"
)
@click.option("-m", "--model", default=None, help="Model name (provider-specific)")
@click.option("-n", "--max-iterations", default=50, help="Maximum loop iterations")
@click.option(
    "--label-model", default="generative", help="Label model: majority, weighted, generative"
)
@click.option("--min-improvement", default=0.005, type=float, help="Minimum F1 improvement to keep")
@click.option("--run-name", default=None, help="Experiment run name")
@click.option("--language", default="en", help="Language code: en, hi, mr, ta, bn")
@click.option(
    "--small-model", is_flag=True, default=False, help="Optimize for small models (8B and below)"
)
def run(
    dataset: str,
    task: str | None,
    labels: str | None,
    provider: str,
    model: str | None,
    max_iterations: int,
    label_model: str,
    min_improvement: float,
    run_name: str | None,
    language: str,
    small_model: bool,
) -> None:
    """Run the autonomous labeling loop."""
    from autolabel.data.loaders import DATASET_LOADERS
    from autolabel.llm import get_provider
    from autolabel.core.loop import AutonomousLoop

    config = AutoLabelConfig()
    config.max_iterations = max_iterations
    config.min_improvement = min_improvement
    config.language = language
    config.small_model_mode = small_model

    # Load dataset
    if dataset not in DATASET_LOADERS:
        raise click.BadParameter(
            f"Unknown dataset '{dataset}'. Available: {sorted(DATASET_LOADERS)}",
            param_hint="--dataset",
        )
    ds = DATASET_LOADERS[dataset](config.datasets_dir)

    if task:
        ds.task_description = task
    if labels:
        ds.label_space = [lbl.strip() for lbl in labels.split(",")]

    # Create provider
    api_keys = {
        "anthropic": config.anthropic_api_key,
        "openai": config.openai_api_key,
        "groq": config.groq_api_key,
    }
    llm = get_provider(
        name=provider,
        model=model or "",
        api_key=api_keys.get(provider, ""),
    )

    # Run loop
    loop = AutonomousLoop(
        dataset=ds,
        provider=llm,
        config=config,
        label_model_type=label_model,
        run_name=run_name,
    )
    loop.run(max_iterations)


@cli.command()
@click.option("-d", "--dataset", default="airline_tweets", help="Dataset name or 'all'")
@click.option("-p", "--provider", default="anthropic", help="LLM provider")
@click.option("-m", "--model", default=None, help="Model name")
@click.option("-n", "--max-iterations", default=30, help="Max iterations for AutoLabel")
@click.option(
    "--llm-time-budget-minutes",
    type=float,
    default=None,
    help="Optional wall-clock budget for zero-shot/few-shot LLM baselines.",
)
@click.option(
    "--llm-request-timeout-seconds",
    type=float,
    default=None,
    help="Optional per-request timeout for LLM baselines (defaults to 20s when a budget is set).",
)
def benchmark(
    dataset: str,
    provider: str,
    model: str | None,
    max_iterations: int,
    llm_time_budget_minutes: float | None,
    llm_request_timeout_seconds: float | None,
) -> None:
    """Run benchmark comparing AutoLabel against baselines."""
    from autolabel.benchmark.runner import BenchmarkRunner
    from autolabel.llm import get_provider

    config = AutoLabelConfig()
    if llm_time_budget_minutes is not None and llm_request_timeout_seconds is None:
        llm_request_timeout_seconds = 20.0
    api_keys = {
        "anthropic": config.anthropic_api_key,
        "openai": config.openai_api_key,
        "groq": config.groq_api_key,
    }
    llm = get_provider(
        name=provider,
        model=model or "",
        api_key=api_keys.get(provider, ""),
    )

    runner = BenchmarkRunner(provider=llm, config=config)

    if dataset == "all":
        from autolabel.data.loaders import DATASET_LOADERS

        datasets = list(DATASET_LOADERS.keys())
    else:
        datasets = [dataset]

    runner.run(
        datasets=datasets,
        max_iterations=max_iterations,
        llm_time_budget_minutes=llm_time_budget_minutes,
        llm_request_timeout_seconds=llm_request_timeout_seconds,
    )


@cli.command()
@click.argument("experiment_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory to write PNG charts to (defaults to the experiment directory).",
)
@click.option(
    "--benchmark-results",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to benchmark/results.json for measured baseline comparison.",
)
def visualize(
    experiment_dir: Path,
    output_dir: Path | None,
    benchmark_results: Path | None,
) -> None:
    """Generate publication-quality PNG charts for an experiment run."""
    try:
        from autolabel.benchmark.visualize import generate_all_charts
    except ImportError as exc:
        raise click.ClickException(
            "Visualization dependencies are not installed. Install them with: "
            'pip install -e ".[viz]"'
        ) from exc

    try:
        created_paths, skipped = generate_all_charts(
            experiment_dir=experiment_dir,
            output_dir=output_dir,
            benchmark_results_path=benchmark_results,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    for chart_path in created_paths:
        click.echo(f"Generated: {chart_path}")
    for message in skipped:
        click.echo(message)


@cli.command()
@click.argument("experiment_dir", type=click.Path(exists=True))
def evaluate(experiment_dir: str) -> None:
    """Evaluate a completed experiment run."""
    exp_dir = Path(experiment_dir)
    summary_file = exp_dir / "final_summary.json"
    if not summary_file.exists():
        click.echo(f"No final_summary.json found in {exp_dir}")
        return

    summary = json.loads(summary_file.read_text())
    click.echo(f"\nExperiment: {exp_dir.name}")
    click.echo(f"  Best Dev F1:  {summary.get('best_dev_f1', 0):.4f}")
    click.echo(f"  Test F1:      {summary.get('test_f1', 0):.4f}")
    click.echo(f"  Test Accuracy:{summary.get('test_accuracy', 0):.4f}")
    click.echo(f"  Active LFs:   {summary.get('active_lfs', 0)}")
    click.echo(f"  Iterations:   {summary.get('total_iterations', 0)}")
    click.echo(f"  Total Cost:   ${summary.get('total_cost', 0):.4f}")


@cli.command()
@click.argument("experiment_dir", type=click.Path(exists=True))
def cost(experiment_dir: str) -> None:
    """Show cost breakdown for an experiment run."""
    exp_dir = Path(experiment_dir)
    summary_file = exp_dir / "final_summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        click.echo(f"\nTotal Cost: ${summary.get('total_cost', 0):.4f}")
        click.echo(f"Total Time: {summary.get('total_time_s', 0):.1f}s")
    else:
        click.echo("No summary file found.")
