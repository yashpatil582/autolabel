"""Agentic self-debugging loop for LF refinement."""

from __future__ import annotations

import logging
from typing import Any

from autolabel.core.failure_analysis import FailureAnalyzer, FailureReport
from autolabel.lf.base import LabelingFunction
from autolabel.lf.generator import LFGenerator

logger = logging.getLogger(__name__)


class AgenticRefiner:
    """Wraps LFGenerator with multi-turn self-debugging.

    For each generated LF:
    1. Trial-execute on dev examples
    2. Classify failures into structured reports
    3. Send failure report back to LLM for refinement
    4. Iterate up to max_turns or until precision target is met
    """

    def __init__(
        self,
        generator: LFGenerator,
        provider: Any,
        label_space: list[str],
        dev_texts: list[str],
        dev_labels: list[str],
        max_turns: int = 3,
        min_precision: float = 0.7,
        trial_size: int = 100,
    ) -> None:
        self.generator = generator
        self.provider = provider
        self.label_space = label_space
        self.dev_texts = dev_texts
        self.dev_labels = dev_labels
        self.max_turns = max_turns
        self.min_precision = min_precision
        self.trial_size = min(trial_size, len(dev_texts))
        self.failure_analyzer = FailureAnalyzer(label_space)
        self.total_refinement_turns = 0

    def generate_and_refine(
        self,
        strategy: str,
        target_label: str,
        examples: list[str],
        existing_lf_descriptions: list[str],
        failure_examples: list[str] | None = None,
        num_lfs: int = 5,
        iteration: int = 0,
    ) -> list[LabelingFunction]:
        """Generate LFs then refine imprecise ones via self-debugging.

        Returns all LFs that meet the precision target after refinement,
        plus any that were already precise enough.
        """
        # Step 1: Generate initial LFs
        initial_lfs = self.generator.generate(
            strategy=strategy,
            target_label=target_label,
            examples=examples,
            existing_lf_descriptions=existing_lf_descriptions,
            failure_examples=failure_examples,
            num_lfs=num_lfs,
            iteration=iteration,
        )

        if not initial_lfs:
            return []

        # Step 2: Trial-execute and refine each LF
        trial_texts = self.dev_texts[: self.trial_size]
        trial_labels = self.dev_labels[: self.trial_size]

        refined_lfs: list[LabelingFunction] = []

        for lf in initial_lfs:
            report = self.failure_analyzer.classify_errors(lf, trial_texts, trial_labels)

            if report.precision >= self.min_precision:
                refined_lfs.append(lf)
                continue

            if report.error_taxonomy == "too_narrow" and report.fire_rate < 0.01:
                # Too narrow to refine meaningfully — keep as is if any fires
                refined_lfs.append(lf)
                continue

            # Try to refine
            refined = self._refine_lf(lf, report, strategy, target_label, examples, iteration)
            if refined is not None:
                refined_lfs.append(refined)
            else:
                # Refinement failed — keep original if precision > 0.5
                if report.precision >= 0.5:
                    refined_lfs.append(lf)

        return refined_lfs

    def _refine_lf(
        self,
        lf: LabelingFunction,
        report: FailureReport,
        strategy: str,
        target_label: str,
        examples: list[str],
        iteration: int,
    ) -> LabelingFunction | None:
        """Attempt to refine an LF through multi-turn debugging.

        Returns the refined LF if precision target is met, else None.
        """
        trial_texts = self.dev_texts[: self.trial_size]
        trial_labels = self.dev_labels[: self.trial_size]
        current_lf = lf
        current_report = report

        for turn in range(self.max_turns):
            self.total_refinement_turns += 1

            refined_lf = self._request_refinement(
                current_lf, current_report, strategy, target_label, examples, iteration
            )

            if refined_lf is None:
                logger.info(
                    "Refinement turn %d/%d for %s: LLM failed to produce valid code",
                    turn + 1,
                    self.max_turns,
                    lf.name,
                )
                return None

            # Re-evaluate
            new_report = self.failure_analyzer.classify_errors(
                refined_lf, trial_texts, trial_labels
            )

            logger.info(
                "Refinement turn %d/%d for %s: precision %.2f -> %.2f",
                turn + 1,
                self.max_turns,
                lf.name,
                current_report.precision,
                new_report.precision,
            )

            if new_report.precision >= self.min_precision:
                return refined_lf

            # If precision got worse, stop
            if new_report.precision < current_report.precision - 0.1:
                logger.info("Precision degraded, stopping refinement for %s", lf.name)
                return current_lf if current_report.precision >= 0.5 else None

            current_lf = refined_lf
            current_report = new_report

        # Exhausted turns — return best if acceptable
        if current_report.precision >= 0.5:
            return current_lf
        return None

    def _request_refinement(
        self,
        lf: LabelingFunction,
        report: FailureReport,
        strategy: str,
        target_label: str,
        examples: list[str],
        iteration: int,
    ) -> LabelingFunction | None:
        """Ask the LLM to refine an LF based on a failure report."""
        refined_lfs = self.generator.generate_with_context(
            strategy=strategy,
            target_label=target_label,
            prior_source=lf.source,
            failure_report=report.summary(),
            examples=examples,
            num_lfs=1,
            iteration=iteration,
        )
        return refined_lfs[0] if refined_lfs else None
