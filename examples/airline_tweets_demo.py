#!/usr/bin/env python3
"""Demo: Run AutoLabel on the airline tweets dataset.

Usage:
    python examples/airline_tweets_demo.py
"""

from pathlib import Path

from autolabel.config import AutoLabelConfig
from autolabel.core.loop import AutonomousLoop
from autolabel.data.loaders import load_airline_tweets
from autolabel.llm import get_provider


def main() -> None:
    config = AutoLabelConfig()

    # Load dataset
    dataset = load_airline_tweets(config.datasets_dir)
    print(f"Loaded: {dataset}")
    print(f"Label space ({dataset.num_classes}): {dataset.label_space}")

    # Create LLM provider
    provider = get_provider(
        name=config.default_provider,
        model=config.default_model,
    )

    # Run autonomous loop
    loop = AutonomousLoop(
        dataset=dataset,
        provider=provider,
        config=config,
        label_model_type="generative",
        run_name="airline_demo",
    )
    results = loop.run(max_iterations=50)

    # Evaluate on test set
    test_result = loop.evaluate_test()
    print(f"\nFinal Test Results: {test_result}")


if __name__ == "__main__":
    main()
