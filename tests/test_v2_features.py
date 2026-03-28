"""Tests for AutoLabel v2.0 features."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from autolabel.config import AutoLabelConfig
from autolabel.core.agent import AgenticRefiner
from autolabel.core.bootstrap import ZeroLabelBootstrap
from autolabel.core.experiment import IterationResult
from autolabel.core.failure_analysis import FailureAnalyzer, FailureReport
from autolabel.core.meta import MetaLearner
from autolabel.core.ratchet import GranularRatchet, MultiObjectiveRatchet, Ratchet
from autolabel.core.strategy import STRATEGIES
from autolabel.data.dataset import AutoLabelDataset
from autolabel.lf.base import LabelingFunction
from autolabel.lf.library import LFLibrary
from autolabel.lf.registry import LFRegistry
from autolabel.lf.scorer import LFScore, LFScorer
from tests.conftest import MockLLMProvider


# ---------------------------------------------------------------------------
# Feature 1: Per-LF Granular Scoring + Smart Pruning
# ---------------------------------------------------------------------------


class TestLFScorer:
    def test_score_lf_basic(self, sample_dataset, sample_lf):
        from autolabel.label_model import get_label_model

        scorer = LFScorer(sample_dataset.label_space)
        score = scorer.score_lf(
            lf=sample_lf,
            dev_texts=sample_dataset.dev_texts,
            dev_labels=sample_dataset.dev_labels,
            all_lfs=[sample_lf],
            label_model_factory=lambda: get_label_model("majority"),
            num_classes=sample_dataset.num_classes,
        )
        assert isinstance(score, LFScore)
        assert score.name == sample_lf.name
        assert 0.0 <= score.precision <= 1.0
        assert 0.0 <= score.recall <= 1.0
        assert 0.0 <= score.coverage <= 1.0
        assert isinstance(score.composite_score, float)

    def test_score_batch(self, sample_dataset, sample_lf):
        from autolabel.label_model import get_label_model

        scorer = LFScorer(sample_dataset.label_space)
        scores = scorer.score_batch(
            lfs=[sample_lf],
            dev_texts=sample_dataset.dev_texts,
            dev_labels=sample_dataset.dev_labels,
            all_lfs=[sample_lf],
            label_model_factory=lambda: get_label_model("majority"),
            num_classes=sample_dataset.num_classes,
        )
        assert len(scores) == 1
        assert scores[0].name == sample_lf.name

    def test_score_to_dict(self):
        score = LFScore(
            name="test",
            precision=0.8,
            recall=0.6,
            coverage=0.3,
            correlation=0.1,
            marginal_f1_delta=0.05,
            composite_score=0.7,
        )
        d = score.to_dict()
        assert d["name"] == "test"
        assert d["precision"] == 0.8


class TestGranularRatchet:
    def test_inherits_ratchet(self):
        gr = GranularRatchet(min_improvement=0.01, min_precision=0.5)
        assert isinstance(gr, Ratchet)
        assert gr.should_keep(0.5, 0.52)
        assert not gr.should_keep(0.5, 0.505)


class TestMultiObjectiveRatchet:
    def test_keep_on_multi_objective_improvement(self):
        r = MultiObjectiveRatchet(min_improvement=0.005)
        assert r.should_keep_multi(
            f1_before=0.5,
            f1_after=0.52,
            coverage_before=0.3,
            coverage_after=0.4,
            n_strategies_before=2,
            n_strategies_after=3,
        )

    def test_discard_on_no_improvement(self):
        r = MultiObjectiveRatchet(min_improvement=0.005)
        assert not r.should_keep_multi(
            f1_before=0.5,
            f1_after=0.5,
            coverage_before=0.3,
            coverage_after=0.3,
            n_strategies_before=2,
            n_strategies_after=2,
        )


class TestRegistryPrune:
    def test_prune_retires_lfs(self, sample_lf):
        registry = LFRegistry()
        registry.add(sample_lf)
        assert len(registry.active_lfs) == 1
        pruned = registry.prune([sample_lf.name])
        assert pruned == 1
        assert len(registry.active_lfs) == 0
        assert len(registry.retired_lfs) == 1

    def test_prune_skips_missing(self):
        registry = LFRegistry()
        pruned = registry.prune(["nonexistent_lf"])
        assert pruned == 0

    def test_scores_field(self):
        registry = LFRegistry()
        assert isinstance(registry.scores, dict)


# ---------------------------------------------------------------------------
# Feature 2: Agentic Self-Debugging Loop
# ---------------------------------------------------------------------------


class TestFailureAnalyzer:
    def test_classify_errors(self, sample_dataset, sample_lf):
        analyzer = FailureAnalyzer(sample_dataset.label_space)
        report = analyzer.classify_errors(
            lf=sample_lf,
            dev_texts=sample_dataset.dev_texts,
            dev_labels=sample_dataset.dev_labels,
        )
        assert isinstance(report, FailureReport)
        assert report.lf_name == sample_lf.name
        assert 0.0 <= report.precision <= 1.0
        assert 0.0 <= report.recall <= 1.0
        assert report.error_taxonomy in {
            "overly_broad",
            "too_narrow",
            "wrong_pattern",
            "acceptable",
            "unknown",
        }

    def test_report_summary(self, sample_dataset, sample_lf):
        analyzer = FailureAnalyzer(sample_dataset.label_space)
        report = analyzer.classify_errors(
            lf=sample_lf,
            dev_texts=sample_dataset.dev_texts,
            dev_labels=sample_dataset.dev_labels,
        )
        summary = report.summary()
        assert sample_lf.name in summary
        assert "Precision" in summary


class TestAgenticRefiner:
    def test_generate_and_refine(self, sample_dataset):
        refined_lf_response = '''```python
def lf_keyword_delta_refined_01(text: str):
    """Detects Delta keyword precisely."""
    lower = text.lower()
    if "delta" in lower and "air" not in lower.split("delta")[0][-10:]:
        return "Delta Air Lines"
    return None
```'''
        provider = MockLLMProvider(
            responses=[
                # Initial generation
                '''```python
def lf_keyword_delta_01(text: str):
    """Detects Delta keyword."""
    if "delta" in text.lower():
        return "Delta Air Lines"
    return None
```''',
                # Refinement
                refined_lf_response,
            ]
        )

        from autolabel.lf.generator import LFGenerator

        generator = LFGenerator(
            provider=provider,
            label_space=sample_dataset.label_space,
            task_description=sample_dataset.task_description,
        )

        agent = AgenticRefiner(
            generator=generator,
            provider=provider,
            label_space=sample_dataset.label_space,
            dev_texts=sample_dataset.dev_texts,
            dev_labels=sample_dataset.dev_labels,
            max_turns=1,
            min_precision=0.5,
        )

        lfs = agent.generate_and_refine(
            strategy="keyword",
            target_label="Delta Air Lines",
            examples=["I love Delta", "Delta is great"],
            existing_lf_descriptions=[],
            num_lfs=1,
            iteration=1,
        )
        # Should return at least the original or refined LF
        assert isinstance(lfs, list)


# ---------------------------------------------------------------------------
# Feature 3: Zero-Label Bootstrap Mode
# ---------------------------------------------------------------------------


class TestAutoLabelDatasetV2:
    def test_has_labels(self, sample_dataset):
        assert sample_dataset.has_labels

    def test_from_unlabeled(self):
        texts = ["text1", "text2", "text3", "text4", "text5"]
        ds = AutoLabelDataset.from_unlabeled(
            texts=texts,
            label_space=["A", "B"],
            task_description="Classify",
        )
        assert ds.name == "unlabeled"
        assert len(ds.texts) == 5
        assert all(lb == "" for lb in ds.labels)
        assert len(ds.train_indices) + len(ds.dev_indices) + len(ds.test_indices) == 5

    def test_pseudo_label_fields(self):
        ds = AutoLabelDataset(
            name="test",
            task_description="test",
            label_space=["A", "B"],
            texts=["t1", "t2"],
            labels=["A", "B"],
        )
        assert ds.pseudo_labels == []
        assert ds.pseudo_confidence == []

    def test_has_labels_false_with_pseudo(self):
        ds = AutoLabelDataset(
            name="test",
            task_description="test",
            label_space=["A", "B"],
            texts=["t1", "t2"],
            labels=["A", "B"],
            pseudo_labels=["A", "B"],
            pseudo_confidence=[0.9, 0.9],
        )
        assert not ds.has_labels  # has pseudo_labels set


class TestZeroLabelBootstrap:
    def test_bootstrap_generates_pseudo_labels(self):
        provider = MockLLMProvider(
            responses=["Delta Air Lines", "United Airlines", "Air Canada"] * 200
        )
        ds = AutoLabelDataset.from_unlabeled(
            texts=[f"text {i}" for i in range(50)],
            label_space=["Delta Air Lines", "United Airlines", "Air Canada"],
            task_description="Extract airline",
        )

        bootstrapper = ZeroLabelBootstrap(
            provider=provider,
            label_space=ds.label_space,
            task_description=ds.task_description,
            sample_size=10,
            consistency_k=3,
            confidence_threshold=0.8,
        )
        bootstrapper.generate_pseudo_labels(ds)

        assert len(ds.pseudo_labels) == 50
        assert len(ds.pseudo_confidence) == 50
        assert all(lb for lb in ds.labels)  # All labels filled

    def test_temperature_schedule(self):
        bs = ZeroLabelBootstrap(
            provider=MockLLMProvider(),
            label_space=["A"],
            task_description="test",
            consistency_k=3,
        )
        temps = bs._get_temperatures()
        assert len(temps) == 3
        assert temps[0] == 0.0


class TestLoadUnlabeled:
    def test_load_unlabeled(self):
        from autolabel.data.loaders import load_unlabeled

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line one\nline two\nline three\n")
            f.flush()
            path = Path(f.name)

        try:
            ds = load_unlabeled(
                filepath=path,
                label_space=["A", "B"],
                task_description="Classify",
            )
            assert len(ds.texts) == 3
            assert all(lb == "" for lb in ds.labels)
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Feature 4: Meta-Learning
# ---------------------------------------------------------------------------


class TestMetaLearner:
    def test_update_and_weights(self):
        ml = MetaLearner(STRATEGIES)
        ml.update("keyword", iteration=1, coverage=0.3, kept=True, f1_delta=0.05)
        ml.update("keyword", iteration=2, coverage=0.4, kept=True, f1_delta=0.03)
        ml.update("regex", iteration=3, coverage=0.5, kept=False, f1_delta=0.0)

        weights = ml.get_strategy_weights()
        assert len(weights) == len(STRATEGIES)
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
        # keyword should have higher weight than regex
        assert weights["keyword"] > weights["regex"]

    def test_temperature_adaptation(self):
        ml = MetaLearner(STRATEGIES)
        # Early phase: high temperature
        assert ml.get_temperature() >= 0.7

        # Stagnation: temperature should increase
        for i in range(5):
            ml.update("keyword", iteration=i, coverage=0.3, kept=False, f1_delta=0.0)
        temp_stagnant = ml.get_temperature()
        assert temp_stagnant > 0.7

    def test_success_rate(self):
        ml = MetaLearner(STRATEGIES)
        ml.update("keyword", iteration=1, coverage=0.3, kept=True, f1_delta=0.05)
        ml.update("keyword", iteration=2, coverage=0.3, kept=False, f1_delta=0.0)

        rate = ml.get_success_rate("keyword", iteration=1, coverage=0.3)
        assert 0.0 < rate < 1.0

    def test_suggest_strategy(self):
        ml = MetaLearner(STRATEGIES)
        # Not enough data
        assert ml.suggest_strategy(0.5, 1) is None

        for i in range(5):
            ml.update("keyword", iteration=i, coverage=0.3, kept=True, f1_delta=0.05)

        suggestion = ml.suggest_strategy(0.5, 6)
        assert suggestion in STRATEGIES


# ---------------------------------------------------------------------------
# Feature 5: Label Model Ensemble (tested via loop integration)
# ---------------------------------------------------------------------------


class TestEnsembleConfig:
    def test_config_has_ensemble_flag(self):
        config = AutoLabelConfig()
        assert hasattr(config, "ensemble_label_models")
        assert config.ensemble_label_models is True


# ---------------------------------------------------------------------------
# Feature 6: Cross-Dataset LF Transfer
# ---------------------------------------------------------------------------


class TestLFLibrary:
    def test_save_and_find(self, sample_lf):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = LFLibrary(tmpdir)
            saved = lib.save([sample_lf], domain="airlines", scores={"lf_keyword_delta_01": 0.8})
            assert saved == 1
            assert len(lib) == 1

            # Find transferable
            candidates = lib.find_transferable(
                target_domain="other_dataset",
                label_space=["A", "B"],
            )
            assert len(candidates) == 1
            assert candidates[0]["domain"] == "airlines"

    def test_no_self_transfer(self, sample_lf):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = LFLibrary(tmpdir)
            lib.save([sample_lf], domain="airlines")

            candidates = lib.find_transferable(
                target_domain="airlines",
                label_space=["A"],
            )
            assert len(candidates) == 0

    def test_persistence(self, sample_lf):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib1 = LFLibrary(tmpdir)
            lib1.save([sample_lf], domain="airlines")

            # Reload
            lib2 = LFLibrary(tmpdir)
            assert len(lib2) == 1

    def test_no_duplicates(self, sample_lf):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = LFLibrary(tmpdir)
            lib.save([sample_lf], domain="airlines")
            lib.save([sample_lf], domain="airlines")  # duplicate
            assert len(lib) == 1

    def test_min_score_filter(self, sample_lf):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = LFLibrary(tmpdir)
            lib.save([sample_lf], domain="airlines", scores={"lf_keyword_delta_01": 0.1})
            candidates = lib.find_transferable(
                target_domain="other",
                label_space=["A"],
                min_score=0.5,
            )
            assert len(candidates) == 0


class TestLabelingFunctionV2:
    def test_new_fields_defaults(self):
        lf = LabelingFunction(
            name="test",
            source="def lf_test(text): return None",
            strategy="keyword",
            description="test",
            target_label="A",
            iteration=0,
        )
        assert lf.domain == ""
        assert lf.abstract_pattern == ""
        assert lf.transferability_score == 0.0

    def test_new_fields_set(self):
        lf = LabelingFunction(
            name="test",
            source="def lf_test(text): return None",
            strategy="keyword",
            description="test",
            target_label="A",
            iteration=0,
            domain="airlines",
            abstract_pattern="keyword_match",
            transferability_score=0.8,
        )
        assert lf.domain == "airlines"
        assert lf.transferability_score == 0.8


# ---------------------------------------------------------------------------
# IterationResult v2 fields
# ---------------------------------------------------------------------------


class TestIterationResultV2:
    def test_new_fields(self):
        result = IterationResult(
            iteration=1,
            strategy="keyword",
            target_label="A",
            new_lfs_generated=5,
            new_lfs_valid=3,
            f1_before=0.5,
            f1_after=0.55,
            f1_delta=0.05,
            kept=True,
            active_lf_count=10,
            coverage=0.6,
            accuracy=0.7,
            refinement_turns=2,
            lfs_kept=3,
            lfs_pruned=1,
            reasoning="Low coverage on label A",
        )
        d = result.to_dict()
        assert d["refinement_turns"] == 2
        assert d["lfs_kept"] == 3
        assert d["lfs_pruned"] == 1
        assert d["reasoning"] == "Low coverage on label A"


# ---------------------------------------------------------------------------
# Config v2 fields
# ---------------------------------------------------------------------------


class TestConfigV2:
    def test_all_new_config_fields(self):
        config = AutoLabelConfig()
        assert config.prune_interval == 10
        assert config.min_lf_precision == 0.6
        assert config.max_lf_correlation == 0.95
        assert config.agent_max_turns == 3
        assert config.agent_min_precision == 0.7
        assert config.bootstrap_sample_size == 200
        assert config.bootstrap_consistency_k == 3
        assert config.bootstrap_confidence_threshold == 0.8
        assert config.meta_learning is True
        assert config.ensemble_label_models is True
        assert config.lf_library_path == ""


# ---------------------------------------------------------------------------
# Strategy reasoning storage
# ---------------------------------------------------------------------------


class TestStrategyReasoning:
    def test_reasoning_stored(self, sample_dataset):
        from autolabel.core.strategy import StrategySelector

        response = (
            '{"strategy": "keyword", "target_label": "Delta Air Lines", "reasoning": "test reason"}'
        )
        provider = MockLLMProvider(responses=[response])
        selector = StrategySelector(
            provider, sample_dataset.label_space, sample_dataset.task_description
        )
        selector.select(
            current_f1=0.5,
            num_active_lfs=0,
            iteration=1,
            label_coverage={lbl: 0.0 for lbl in sample_dataset.label_space},
            recent_history=[],
        )
        assert selector.last_reasoning == "test reason"


# ---------------------------------------------------------------------------
# Metrics v2
# ---------------------------------------------------------------------------


class TestPseudoF1:
    def test_compute_pseudo_f1(self):
        from autolabel.evaluation.metrics import compute_pseudo_f1

        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "B", "B", "A"]
        confidence = [0.9, 0.9, 0.5, 0.5]
        label_space = ["A", "B"]

        score = compute_pseudo_f1(y_true, y_pred, label_space, confidence)
        assert 0.0 < score < 1.0

    def test_compute_pseudo_f1_all_correct(self):
        from autolabel.evaluation.metrics import compute_pseudo_f1

        y_true = ["A", "B"]
        y_pred = ["A", "B"]
        confidence = [1.0, 1.0]
        label_space = ["A", "B"]

        score = compute_pseudo_f1(y_true, y_pred, label_space, confidence)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration: Loop with v2 features
# ---------------------------------------------------------------------------


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


class TestLoopV2:
    def test_loop_with_v2_features(self, sample_dataset):
        from autolabel.core.loop import AutonomousLoop

        provider = MockLLMProvider(responses=[MOCK_STRATEGY_RESPONSE, MOCK_LF_RESPONSE] * 10)
        loop = AutonomousLoop(
            dataset=sample_dataset,
            provider=provider,
            label_model_type="majority",
            run_name="test_v2",
        )
        results = loop.run(max_iterations=2)
        assert len(results) == 2
        assert all(hasattr(r, "refinement_turns") for r in results)
        assert all(hasattr(r, "reasoning") for r in results)

    def test_loop_constructor_new_params(self, sample_dataset):
        from autolabel.core.loop import AutonomousLoop

        provider = MockLLMProvider()
        loop = AutonomousLoop(
            dataset=sample_dataset,
            provider=provider,
            label_model_type="majority",
            run_name="test_params",
            zero_label=False,
            library_path="",
        )
        assert loop.zero_label is False
        assert loop.library is None
        assert loop.meta_learner is not None
        assert loop.scorer is not None
