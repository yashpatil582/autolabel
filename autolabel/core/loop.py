"""AutonomousLoop — the main engine of AutoLabel."""

from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Any


from autolabel.config import AutoLabelConfig
from autolabel.core.experiment import IterationResult
from autolabel.core.ratchet import GranularRatchet
from autolabel.core.strategy import StrategySelector
from autolabel.data.dataset import AutoLabelDataset
from autolabel.evaluation.evaluator import Evaluator
from autolabel.evaluation.metrics import compute_coverage
from autolabel.label_model import get_label_model
from autolabel.lf.applicator import LFApplicator
from autolabel.lf.base import ABSTAIN, LabelingFunction
from autolabel.lf.generator import LFGenerator
from autolabel.lf.registry import LFRegistry
from autolabel.lf.scorer import LFScorer
from autolabel.logging.experiment_log import ExperimentLogger
from autolabel.logging.progress import ProgressDisplay
from autolabel.llm.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class AutonomousLoop:
    """Core autonomous improvement loop.

    1. Analyse failures
    2. LLM selects strategy + target label
    3. LLM generates labeling functions
    4. Apply all LFs -> label matrix
    5. Label model aggregates -> predictions
    6. Evaluate F1
    7. KEEP if improved, else DISCARD
    8. Log and repeat
    """

    def __init__(
        self,
        dataset: AutoLabelDataset,
        provider: Any,
        config: AutoLabelConfig | None = None,
        label_model_type: str = "majority",
        run_name: str | None = None,
        zero_label: bool = False,
        library_path: str = "",
    ) -> None:
        self.config = config or AutoLabelConfig()
        self.dataset = dataset
        self.provider = provider
        self.zero_label = zero_label

        self.registry = LFRegistry()
        self.generator = LFGenerator(
            provider,
            dataset.label_space,
            dataset.task_description,
            max_lf_lines=self.config.max_lf_lines,
            language=self.config.language,
            small_model_mode=self.config.small_model_mode,
        )

        # Feature 4: Meta-learner
        self.meta_learner = None
        if self.config.meta_learning:
            from autolabel.core.meta import MetaLearner
            from autolabel.core.strategy import STRATEGIES

            self.meta_learner = MetaLearner(STRATEGIES)

        self.strategy_selector = StrategySelector(
            provider,
            dataset.label_space,
            dataset.task_description,
            language=self.config.language,
            meta_learner=self.meta_learner,
        )
        self.evaluator = Evaluator(dataset)
        self.ratchet = GranularRatchet(
            self.config.min_improvement,
            min_precision=self.config.min_lf_precision,
        )
        self.label_model_type = label_model_type
        self.cost_tracker = CostTracker()

        # Feature 1: Per-LF scorer
        self.scorer = LFScorer(dataset.label_space)

        # Feature 2: Agentic refiner
        self.agent = None

        # Feature 5: Ensemble label models
        self.ensemble_label_models = self.config.ensemble_label_models

        # Feature 6: LF Library
        self.library = None
        effective_lib_path = library_path or self.config.lf_library_path
        if effective_lib_path:
            from autolabel.lf.library import LFLibrary

            self.library = LFLibrary(effective_lib_path)

        # Output directory
        if run_name is None:
            run_name = f"{dataset.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_name = run_name
        self.log_dir = self.config.get_experiments_dir(run_name)
        self.logger = ExperimentLogger(self.log_dir)
        self.display = ProgressDisplay()

        # State
        self.best_f1 = 0.0
        self.best_coverage = 0.0
        self.history: list[IterationResult] = []
        self.total_generated = 0

    def run(self, max_iterations: int | None = None) -> list[IterationResult]:
        """Run the autonomous loop for up to max_iterations."""
        max_iter = max_iterations or self.config.max_iterations

        self.display.print_header(
            dataset=self.dataset.name,
            task=self.dataset.task_description,
            provider=type(self.provider).__name__,
            max_iter=max_iter,
        )

        self.logger.log_meta(
            {
                "dataset": self.dataset.name,
                "task": self.dataset.task_description,
                "label_space": self.dataset.label_space,
                "provider": type(self.provider).__name__,
                "max_iterations": max_iter,
                "label_model": self.label_model_type,
                "language": self.config.language,
                "small_model_mode": self.config.small_model_mode,
                "zero_label": self.zero_label,
                "config": {
                    "min_improvement": self.config.min_improvement,
                    "lfs_per_iteration": self.config.lfs_per_iteration,
                    "agent_max_turns": self.config.agent_max_turns,
                    "ensemble_label_models": self.config.ensemble_label_models,
                    "meta_learning": self.config.meta_learning,
                },
            }
        )

        # Feature 3: Zero-label bootstrap
        if self.zero_label and not self.dataset.has_labels:
            self._run_bootstrap()

        # Feature 6: Seed from LF library
        if self.library is not None:
            self._seed_from_library()

        # Initialize agentic refiner (Feature 2) — needs dataset labels
        if self.dataset.dev_texts and self.dataset.dev_labels:
            from autolabel.core.agent import AgenticRefiner

            self.agent = AgenticRefiner(
                generator=self.generator,
                provider=self.provider,
                label_space=self.dataset.label_space,
                dev_texts=self.dataset.dev_texts,
                dev_labels=self.dataset.dev_labels,
                max_turns=self.config.agent_max_turns,
                min_precision=self.config.agent_min_precision,
            )

        # Warmup phase: generate simple LFs for each label
        if self.config.warmup or self.config.small_model_mode:
            self._run_warmup()

        for iteration in range(1, max_iter + 1):
            try:
                result = self._run_iteration(iteration)
                self.history.append(result)
                self.logger.log_iteration(iteration, result.to_dict())

                # Feature 1: Periodic pruning
                if (
                    iteration % self.config.prune_interval == 0
                    and len(self.registry.active_lfs) > 3
                ):
                    self._prune_lfs()

            except Exception as e:
                logger.error("Iteration %d failed: %s", iteration, e, exc_info=True)
                self.display.print_error(f"Iteration {iteration} failed: {e}")
                result = IterationResult(
                    iteration=iteration,
                    strategy="error",
                    target_label="",
                    new_lfs_generated=0,
                    new_lfs_valid=0,
                    f1_before=self.best_f1,
                    f1_after=self.best_f1,
                    f1_delta=0.0,
                    kept=False,
                    active_lf_count=len(self.registry.active_lfs),
                    coverage=0.0,
                    accuracy=0.0,
                    error=str(e),
                )
                self.history.append(result)

        # Feature 6: Save high-scoring LFs to library
        if self.library is not None:
            self._save_to_library()

        # Final summary
        self._log_final_summary()
        return self.history

    def _run_bootstrap(self) -> None:
        """Run zero-label bootstrap to generate pseudo-labels."""
        from autolabel.core.bootstrap import ZeroLabelBootstrap

        self.display.print_info("Running zero-label bootstrap...")
        bootstrapper = ZeroLabelBootstrap(
            provider=self.provider,
            label_space=self.dataset.label_space,
            task_description=self.dataset.task_description,
            sample_size=self.config.bootstrap_sample_size,
            consistency_k=self.config.bootstrap_consistency_k,
            confidence_threshold=self.config.bootstrap_confidence_threshold,
        )
        bootstrapper.generate_pseudo_labels(self.dataset)

        n_confident = sum(
            1
            for c in self.dataset.pseudo_confidence
            if c >= self.config.bootstrap_confidence_threshold
        )
        self.display.print_info(
            f"Bootstrap complete: {n_confident} high-confidence pseudo-labels generated"
        )

    def _seed_from_library(self) -> None:
        """Seed the registry with adapted LFs from the library."""
        if self.library is None or len(self.library) == 0:
            return

        candidates = self.library.find_transferable(
            target_domain=self.dataset.name,
            label_space=self.dataset.label_space,
        )

        if not candidates:
            return

        self.display.print_info(f"Found {len(candidates)} transferable LFs in library")

        seeded = 0
        for entry in candidates[:10]:  # Cap at 10 seed LFs
            adapted = self.library.adapt_lf(
                entry=entry,
                provider=self.provider,
                new_domain=self.dataset.name,
                new_label_space=self.dataset.label_space,
                task_description=self.dataset.task_description,
            )
            if adapted is not None:
                self.registry.add(adapted)
                seeded += 1

        if seeded > 0:
            # Evaluate seeded LFs
            f1, coverage, _ = self._evaluate_lfs(self.registry.active_lfs, split="dev")
            self.best_f1 = f1
            self.best_coverage = coverage
            self.display.print_info(f"Seeded {seeded} LFs from library, starting F1={f1:.4f}")

    def _run_warmup(self) -> None:
        """Generate simple keyword and regex LFs for each label before the main loop."""
        self.display.print_info("Running warmup phase...")
        warmup_strategies = ["keyword", "regex"]

        for label in self.dataset.label_space:
            examples = self._get_examples_for_label(label, n=10)
            existing_descriptions = [lf.description for lf in self.registry.active_lfs]

            for strategy in warmup_strategies:
                new_lfs = self.generator.generate(
                    strategy=strategy,
                    target_label=label,
                    examples=examples,
                    existing_lf_descriptions=existing_descriptions,
                    num_lfs=2,
                    iteration=0,
                )
                self.total_generated += len(new_lfs)

                if not new_lfs:
                    continue

                # Precision filter
                max_fire_rate = min(0.35, 3.0 / max(self.dataset.num_classes, 1))
                dev_texts = self.dataset.dev_texts
                filtered_lfs = []
                for lf in new_lfs:
                    fires = sum(1 for t in dev_texts if lf.apply(t) != ABSTAIN)
                    fire_rate = fires / len(dev_texts) if dev_texts else 0
                    if fire_rate <= max_fire_rate:
                        filtered_lfs.append(lf)

                if not filtered_lfs:
                    continue

                # Ratchet check: only keep if F1 improves
                candidate_lfs = self.registry.active_lfs + filtered_lfs
                f1_after, _, _ = self._evaluate_lfs(candidate_lfs, split="dev")

                if self.ratchet.should_keep(self.best_f1, f1_after):
                    self.registry.add_batch(filtered_lfs)
                    self.best_f1 = f1_after
                    self.display.print_info(
                        f"  Warmup: {strategy}/{label} -> F1={f1_after:.4f} (KEEP)"
                    )

        self.display.print_info(
            f"Warmup complete: {len(self.registry.active_lfs)} LFs, F1={self.best_f1:.4f}"
        )

    def _run_iteration(self, iteration: int) -> IterationResult:
        """Execute a single iteration of the loop."""
        f1_before = self.best_f1

        # 1. Analyse failures to guide strategy selection
        label_coverage = self._compute_label_coverage()

        # 2. Select strategy and target label
        recent = [r.to_dict() for r in self.history[-5:]]
        strategy, target_label = self.strategy_selector.select(
            current_f1=self.best_f1,
            num_active_lfs=len(self.registry.active_lfs),
            iteration=iteration,
            label_coverage=label_coverage,
            recent_history=recent,
        )
        reasoning = self.strategy_selector.last_reasoning
        self.display.print_iteration_start(iteration, strategy, target_label)

        # 3. Gather examples for the target label
        examples = self._get_examples_for_label(target_label, n=10)
        failure_examples = self._get_failure_examples(target_label, n=5)
        existing_descriptions = [lf.description for lf in self.registry.active_lfs]

        # 4. Generate new LFs (Feature 2: use agent if available)
        refinement_turns = 0
        if self.agent is not None:
            new_lfs = self.agent.generate_and_refine(
                strategy=strategy,
                target_label=target_label,
                examples=examples,
                existing_lf_descriptions=existing_descriptions,
                failure_examples=failure_examples,
                num_lfs=self.config.lfs_per_iteration,
                iteration=iteration,
            )
            refinement_turns = self.agent.total_refinement_turns
        else:
            new_lfs = self.generator.generate(
                strategy=strategy,
                target_label=target_label,
                examples=examples,
                existing_lf_descriptions=existing_descriptions,
                failure_examples=failure_examples,
                num_lfs=self.config.lfs_per_iteration,
                iteration=iteration,
            )
        self.total_generated += len(new_lfs)

        # 4b. Precision filter: discard LFs that fire on too many examples
        max_fire_rate = min(0.35, 3.0 / max(self.dataset.num_classes, 1))
        dev_texts = self.dataset.dev_texts
        filtered_lfs = []
        for lf in new_lfs:
            fires = sum(1 for t in dev_texts if lf.apply(t) != ABSTAIN)
            fire_rate = fires / len(dev_texts) if dev_texts else 0
            if fire_rate <= max_fire_rate:
                filtered_lfs.append(lf)
                self.display.print_lf_generated(lf.name, True)
            else:
                self.display.print_info(
                    f"    LF {lf.name}: DROPPED (fires on {fire_rate:.0%} of dev, max {max_fire_rate:.0%})"
                )
        new_lfs = filtered_lfs

        if not new_lfs:
            self.display.print_info("No valid LFs generated this iteration")
            return IterationResult(
                iteration=iteration,
                strategy=strategy,
                target_label=target_label,
                new_lfs_generated=0,
                new_lfs_valid=0,
                f1_before=f1_before,
                f1_after=f1_before,
                f1_delta=0.0,
                kept=False,
                active_lf_count=len(self.registry.active_lfs),
                coverage=0.0,
                accuracy=self.best_f1,
                label_model_type=self.label_model_type,
                reasoning=reasoning,
            )

        # Feature 1: Per-LF granular scoring and greedy addition
        def label_model_factory():
            return get_label_model(self.label_model_type)

        scores = self.scorer.score_batch(
            new_lfs,
            self.dataset.dev_texts,
            self.dataset.dev_labels,
            self.registry.active_lfs + new_lfs,
            label_model_factory,
            self.dataset.num_classes,
        )

        kept_lfs = self.ratchet.filter_batch(
            candidate_lfs=new_lfs,
            scores=scores,
            current_f1=f1_before,
            evaluate_fn=lambda lfs: self._evaluate_lfs(lfs, split="dev"),
            active_lfs=self.registry.active_lfs,
        )

        # If granular ratchet kept nothing, fall back to batch evaluation
        if not kept_lfs:
            candidate_lfs = self.registry.active_lfs + new_lfs
            f1_after, coverage, accuracy = self._evaluate_lfs(candidate_lfs, split="dev")

            if self.ratchet.should_keep(f1_before, f1_after):
                kept_lfs = new_lfs

        # Evaluate final state
        if kept_lfs:
            final_lfs = self.registry.active_lfs + kept_lfs
            f1_after, coverage, accuracy = self._evaluate_lfs(final_lfs, split="dev")
            kept = True
            self.registry.add_batch(kept_lfs)
            self.best_f1 = f1_after
            self.best_coverage = coverage

            # Store scores
            for lf, score in zip(new_lfs, scores):
                if lf in kept_lfs:
                    self.registry.scores[lf.name] = score
        else:
            f1_after = f1_before
            coverage = self.best_coverage
            accuracy = self.best_f1
            kept = False

        # Feature 4: Update meta-learner
        if self.meta_learner is not None:
            self.meta_learner.update(
                strategy=strategy,
                iteration=iteration,
                coverage=coverage,
                kept=kept,
                f1_delta=f1_after - f1_before if kept else 0.0,
            )

        self.display.print_iteration_result(
            iteration=iteration,
            f1=f1_after if kept else f1_before,
            prev_f1=f1_before,
            kept=kept,
            new_lfs=len(kept_lfs) if kept else 0,
            total_lfs=len(self.registry.active_lfs),
            coverage=coverage,
        )

        return IterationResult(
            iteration=iteration,
            strategy=strategy,
            target_label=target_label,
            new_lfs_generated=len(new_lfs),
            new_lfs_valid=len(new_lfs),
            f1_before=f1_before,
            f1_after=f1_after if kept else f1_before,
            f1_delta=(f1_after - f1_before) if kept else 0.0,
            kept=kept,
            active_lf_count=len(self.registry.active_lfs),
            coverage=coverage,
            accuracy=accuracy if kept else self.best_f1,
            label_model_type=self.label_model_type,
            refinement_turns=refinement_turns,
            lfs_kept=len(kept_lfs) if kept else 0,
            reasoning=reasoning,
        )

    def _evaluate_lfs(
        self, lfs: list[LabelingFunction], split: str = "dev"
    ) -> tuple[float, float, float]:
        """Apply LFs, aggregate with label model, evaluate."""
        if not lfs:
            return 0.0, 0.0, 0.0

        texts = self.dataset.dev_texts if split == "dev" else self.dataset.test_texts

        # Apply all LFs
        label_matrix = LFApplicator.apply_lfs(lfs, texts, self.dataset.label_space)

        # Feature 5: Ensemble label models — try all 3, pick best
        if self.ensemble_label_models and split == "dev":
            return self._evaluate_with_ensemble(label_matrix, split)

        # Standard single label model
        label_model = get_label_model(self.label_model_type)
        label_model.fit(label_matrix, self.dataset.num_classes)
        pred_indices = label_model.predict(label_matrix)

        predictions = self._indices_to_labels(pred_indices)
        result = self.evaluator.evaluate(predictions, split=split)
        coverage = compute_coverage(predictions)

        return result.f1, coverage, result.accuracy

    def _evaluate_with_ensemble(
        self, label_matrix: object, split: str
    ) -> tuple[float, float, float]:
        """Try all label models and return the best result."""
        best_f1 = -1.0
        best_result = (0.0, 0.0, 0.0)
        best_model_type = self.label_model_type

        for model_type in ["majority", "weighted", "generative"]:
            try:
                label_model = get_label_model(model_type)
                label_model.fit(label_matrix, self.dataset.num_classes)
                pred_indices = label_model.predict(label_matrix)

                predictions = self._indices_to_labels(pred_indices)
                result = self.evaluator.evaluate(predictions, split=split)
                coverage = compute_coverage(predictions)

                if result.f1 > best_f1:
                    best_f1 = result.f1
                    best_result = (result.f1, coverage, result.accuracy)
                    best_model_type = model_type
            except Exception:
                continue

        if best_model_type != self.label_model_type:
            logger.info(
                "Ensemble: switched from %s to %s (F1 %.4f)",
                self.label_model_type,
                best_model_type,
                best_f1,
            )
            self.label_model_type = best_model_type

        return best_result

    def _indices_to_labels(self, pred_indices: object) -> list[str | None]:
        """Convert label model prediction indices to label strings."""
        predictions: list[str | None] = []
        for idx in pred_indices:
            if idx == -1:
                predictions.append(None)
            else:
                predictions.append(self.dataset.label_space[idx])
        return predictions

    def _prune_lfs(self) -> None:
        """Prune redundant and harmful LFs from the registry.

        Removes LFs with:
        - Correlation > max_lf_correlation with any other active LF
        - Marginal F1 delta <= 0 (removing them doesn't hurt)
        """
        if len(self.registry.active_lfs) <= 3:
            return

        def label_model_factory():
            return get_label_model(self.label_model_type)

        scores = self.scorer.score_batch(
            self.registry.active_lfs,
            self.dataset.dev_texts,
            self.dataset.dev_labels,
            self.registry.active_lfs,
            label_model_factory,
            self.dataset.num_classes,
        )

        to_prune = []
        for lf, score in zip(self.registry.active_lfs, scores):
            if score.correlation > self.config.max_lf_correlation:
                to_prune.append(lf.name)
                logger.info("Pruning %s: correlation %.2f", lf.name, score.correlation)
            elif score.marginal_f1_delta <= 0 and score.precision < self.config.min_lf_precision:
                to_prune.append(lf.name)
                logger.info(
                    "Pruning %s: marginal_f1=%.4f, precision=%.2f",
                    lf.name,
                    score.marginal_f1_delta,
                    score.precision,
                )

        if to_prune:
            pruned = self.registry.prune(to_prune)
            # Re-evaluate after pruning
            if self.registry.active_lfs:
                f1, coverage, _ = self._evaluate_lfs(self.registry.active_lfs, split="dev")
                self.best_f1 = f1
                self.best_coverage = coverage
            self.display.print_info(
                f"Pruned {pruned} LFs, now {len(self.registry.active_lfs)} active"
            )

    def _save_to_library(self) -> None:
        """Save high-scoring active LFs to the library."""
        if self.library is None:
            return

        scores_dict = {}
        for name, score in self.registry.scores.items():
            scores_dict[name] = getattr(score, "composite_score", 0.0)

        # Save LFs with composite score > 0.3
        good_lfs = [lf for lf in self.registry.active_lfs if scores_dict.get(lf.name, 0.0) > 0.3]

        if good_lfs:
            saved = self.library.save(good_lfs, domain=self.dataset.name, scores=scores_dict)
            self.display.print_info(f"Saved {saved} LFs to library")

    def _compute_label_coverage(self) -> dict[str, float]:
        """Compute per-label coverage on dev set."""
        if not self.registry.active_lfs:
            return {label: 0.0 for label in self.dataset.label_space}

        texts = self.dataset.dev_texts
        true_labels = self.dataset.dev_labels

        label_matrix = LFApplicator.apply_lfs(
            self.registry.active_lfs, texts, self.dataset.label_space
        )

        label_model = get_label_model(self.label_model_type)
        label_model.fit(label_matrix, self.dataset.num_classes)
        pred_indices = label_model.predict(label_matrix)

        coverage: dict[str, float] = {}
        for label in self.dataset.label_space:
            label_idx = self.dataset.label_space.index(label)
            mask = [tl == label for tl in true_labels]
            total = sum(mask)
            if total == 0:
                coverage[label] = 0.0
                continue
            correct = sum(1 for i, m in enumerate(mask) if m and pred_indices[i] == label_idx)
            coverage[label] = correct / total

        return coverage

    def _get_examples_for_label(self, label: str, n: int = 10) -> list[str]:
        """Get example texts from training set for a given label."""
        examples = [
            text
            for text, lbl in zip(self.dataset.train_texts, self.dataset.train_labels)
            if lbl == label
        ]
        if len(examples) > n:
            return random.sample(examples, n)
        return examples

    def _get_failure_examples(self, target_label: str, n: int = 5) -> list[str]:
        """Get dev examples where the target label is the true label but we get it wrong."""
        if not self.registry.active_lfs:
            return []

        texts = self.dataset.dev_texts
        true_labels = self.dataset.dev_labels

        label_matrix = LFApplicator.apply_lfs(
            self.registry.active_lfs, texts, self.dataset.label_space
        )
        label_model = get_label_model(self.label_model_type)
        label_model.fit(label_matrix, self.dataset.num_classes)
        pred_indices = label_model.predict(label_matrix)

        target_idx = self.dataset.label_space.index(target_label)
        failures = []
        for i, (true_lbl, pred_idx) in enumerate(zip(true_labels, pred_indices)):
            if true_lbl == target_label and pred_idx != target_idx:
                failures.append(texts[i])

        if len(failures) > n:
            return random.sample(failures, n)
        return failures

    def _log_final_summary(self) -> None:
        """Log final summary and display it."""
        # Evaluate on test set
        test_f1, test_coverage, test_accuracy = 0.0, 0.0, 0.0
        if self.registry.active_lfs:
            test_f1, test_coverage, test_accuracy = self._evaluate_lfs(
                self.registry.active_lfs, split="test"
            )

        summary = {
            "best_dev_f1": self.best_f1,
            "test_f1": test_f1,
            "test_accuracy": test_accuracy,
            "test_coverage": test_coverage,
            "total_iterations": len(self.history),
            "active_lfs": len(self.registry.active_lfs),
            "total_generated": self.total_generated,
            "total_cost": self.cost_tracker.total_cost(),
            "f1_trajectory": [r.f1_after for r in self.history],
            "zero_label": self.zero_label,
        }
        self.logger.log_final(summary)

        self.display.print_final_summary(
            best_f1=self.best_f1,
            total_iterations=len(self.history),
            active_lfs=len(self.registry.active_lfs),
            total_generated=self.total_generated,
            total_cost=self.cost_tracker.total_cost() or None,
        )

        self.display.print_info(f"Test F1: {test_f1:.4f} | Test Accuracy: {test_accuracy:.4f}")
        self.display.print_info(f"Results saved to {self.log_dir}")

    def evaluate_test(self) -> dict[str, float]:
        """Evaluate current LFs on the test set."""
        if not self.registry.active_lfs:
            return {"f1": 0.0, "accuracy": 0.0, "coverage": 0.0}
        f1, coverage, accuracy = self._evaluate_lfs(self.registry.active_lfs, split="test")
        return {"f1": f1, "accuracy": accuracy, "coverage": coverage}
