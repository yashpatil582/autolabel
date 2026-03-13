#!/usr/bin/env python3
"""Demo: Using AutoLabel with a custom dataset.

Usage:
    python examples/custom_task.py
"""

from autolabel.config import AutoLabelConfig
from autolabel.core.loop import AutonomousLoop
from autolabel.data.dataset import AutoLabelDataset
from autolabel.llm import get_provider


def main() -> None:
    # Define your own dataset
    texts = [
        "The movie was absolutely fantastic, I loved every minute!",
        "What a waste of time, terrible acting throughout.",
        "A masterpiece of modern cinema, brilliant direction.",
        "I fell asleep halfway through, so boring.",
        "One of the best films I've seen this year!",
        "Horrible plot, wooden dialogue, avoid at all costs.",
        "Beautifully shot with amazing performances.",
        "Predictable and dull, not worth the ticket price.",
    ]
    labels = ["positive", "negative", "positive", "negative",
              "positive", "negative", "positive", "negative"]

    dataset = AutoLabelDataset(
        name="custom_sentiment",
        task_description="Classify the sentiment of this movie review",
        label_space=["positive", "negative"],
        texts=texts,
        labels=labels,
        train_indices=[0, 1, 2, 3],
        dev_indices=[4, 5],
        test_indices=[6, 7],
    )

    config = AutoLabelConfig()
    provider = get_provider(name=config.default_provider)

    loop = AutonomousLoop(
        dataset=dataset,
        provider=provider,
        config=config,
        label_model_type="majority",
        run_name="custom_sentiment_demo",
    )
    results = loop.run(max_iterations=10)
    print(f"\nTest results: {loop.evaluate_test()}")


if __name__ == "__main__":
    main()
