"""Per-dataset benchmark configurations."""

from __future__ import annotations

BENCHMARK_CONFIGS: dict[str, dict] = {
    "airline_tweets": {
        "max_iterations": 50,
        "expected_baseline_f1": 0.31,
        "target_f1": 0.85,
    },
    "imdb": {
        "max_iterations": 30,
        "expected_baseline_f1": 0.80,
        "target_f1": 0.90,
    },
    "ag_news": {
        "max_iterations": 30,
        "expected_baseline_f1": 0.85,
        "target_f1": 0.90,
    },
    "yelp": {
        "max_iterations": 30,
        "expected_baseline_f1": 0.45,
        "target_f1": 0.55,
    },
    "sms_spam": {
        "max_iterations": 20,
        "expected_baseline_f1": 0.90,
        "target_f1": 0.95,
    },
    "trec": {
        "max_iterations": 30,
        "expected_baseline_f1": 0.70,
        "target_f1": 0.80,
    },
}
