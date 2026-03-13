"""Benchmark runner — compares AutoLabel against baselines."""

from __future__ import annotations

import json
import logging
from typing import Any

from autolabel.benchmark.baselines import BaselineRunner
from autolabel.config import AutoLabelConfig
from autolabel.core.loop import AutonomousLoop
from autolabel.data.loaders import DATASET_LOADERS
from autolabel.logging.progress import ProgressDisplay

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs full benchmark: baselines + AutoLabel on specified datasets."""

    def __init__(self, provider: Any, config: AutoLabelConfig | None = None) -> None:
        self.provider = provider
        self.config = config or AutoLabelConfig()
        self.display = ProgressDisplay()

    def run(self, datasets: list[str], max_iterations: int = 30) -> dict[str, Any]:
        """Run benchmark on all specified datasets."""
        all_results: dict[str, list[dict]] = {}

        for ds_name in datasets:
            if ds_name not in DATASET_LOADERS:
                logger.warning("Unknown dataset '%s', skipping", ds_name)
                continue

            self.display.print_info(f"\n{'='*60}")
            self.display.print_info(f"Benchmarking: {ds_name}")
            self.display.print_info(f"{'='*60}")

            dataset = DATASET_LOADERS[ds_name](self.config.datasets_dir)

            # Run baselines
            baseline_runner = BaselineRunner(dataset, provider=self.provider)
            results = baseline_runner.run_all()

            # Run AutoLabel
            loop = AutonomousLoop(
                dataset=dataset,
                provider=self.provider,
                config=self.config,
                run_name=f"benchmark_{ds_name}",
            )
            loop.run(max_iterations)
            test_result = loop.evaluate_test()
            results.append({
                "method": "AutoLabel",
                "f1": test_result["f1"],
                "accuracy": test_result["accuracy"],
                "coverage": test_result["coverage"],
            })

            all_results[ds_name] = results
            self.display.print_benchmark_table(results)

        # Save results
        output_dir = self.config.experiments_dir / "benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "results.json"
        output_file.write_text(json.dumps(all_results, indent=2))
        self.display.print_info(f"\nResults saved to {output_file}")

        return all_results
