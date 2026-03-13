"""Tests for the autonomous loop."""

from autolabel.core.loop import AutonomousLoop
from autolabel.core.ratchet import Ratchet
from autolabel.core.strategy import StrategySelector, STRATEGIES
from tests.conftest import MockLLMProvider


MOCK_LF_RESPONSE = '''```python
def lf_keyword_delta_01(text: str):
    """Detects Delta keyword."""
    if "delta" in text.lower():
        return "Delta Air Lines"
    return None
```

```python
def lf_keyword_delta_02(text: str):
    """Detects Delta abbreviation DL."""
    if " dl " in text.lower():
        return "Delta Air Lines"
    return None
```
'''

MOCK_STRATEGY_RESPONSE = (
    '{"strategy": "keyword", "target_label": "Delta Air Lines", "reasoning": "Low coverage"}'
)


class TestRatchet:
    def test_keep_on_improvement(self):
        r = Ratchet(min_improvement=0.005)
        assert r.should_keep(0.5, 0.51)

    def test_discard_on_no_improvement(self):
        r = Ratchet(min_improvement=0.005)
        assert not r.should_keep(0.5, 0.502)

    def test_discard_on_regression(self):
        r = Ratchet(min_improvement=0.005)
        assert not r.should_keep(0.5, 0.49)


class TestStrategySelector:
    def test_all_strategies_valid(self):
        assert len(STRATEGIES) == 8
        for s in [
            "keyword",
            "regex",
            "fuzzy",
            "semantic",
            "abbreviation",
            "negation",
            "context",
            "compositional",
        ]:
            assert s in STRATEGIES

    def test_fallback_on_bad_response(self, sample_dataset):
        provider = MockLLMProvider(responses=["not valid json"])
        selector = StrategySelector(
            provider, sample_dataset.label_space, sample_dataset.task_description
        )
        strategy, label = selector.select(
            current_f1=0.5,
            num_active_lfs=0,
            iteration=1,
            label_coverage={lbl: 0.0 for lbl in sample_dataset.label_space},
            recent_history=[],
        )
        assert strategy in STRATEGIES
        assert label in sample_dataset.label_space


class TestAutonomousLoop:
    def test_loop_runs(self, sample_dataset):
        provider = MockLLMProvider(responses=[MOCK_STRATEGY_RESPONSE, MOCK_LF_RESPONSE] * 5)
        loop = AutonomousLoop(
            dataset=sample_dataset,
            provider=provider,
            label_model_type="majority",
            run_name="test_run",
        )
        results = loop.run(max_iterations=2)
        assert len(results) == 2
        assert all(hasattr(r, "f1_after") for r in results)
