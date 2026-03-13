from autolabel.benchmark.runner import BenchmarkRunner
from autolabel.benchmark.baselines import BaselineRunner

__all__ = ["BenchmarkRunner", "BaselineRunner", "generate_all_charts"]


def __getattr__(name: str):
    if name == "generate_all_charts":
        from autolabel.benchmark.visualize import generate_all_charts

        return generate_all_charts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
