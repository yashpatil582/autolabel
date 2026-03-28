"""Keep/discard logic for the autonomous loop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autolabel.lf.base import LabelingFunction
    from autolabel.lf.scorer import LFScore

logger = logging.getLogger(__name__)


class Ratchet:
    """Decides whether to keep or discard new labeling functions.

    Keeps new LFs only if F1 improved by at least `min_improvement`.
    """

    def __init__(self, min_improvement: float = 0.005) -> None:
        self.min_improvement = min_improvement

    def should_keep(self, f1_before: float, f1_after: float) -> bool:
        """Return True if the improvement is sufficient to keep new LFs."""
        return (f1_after - f1_before) >= self.min_improvement


class GranularRatchet(Ratchet):
    """Per-LF filtering: keeps the subset of LFs that individually improve F1.

    Instead of batch keep/discard, scores each candidate LF individually
    and greedily adds LFs that improve F1.
    """

    def __init__(
        self,
        min_improvement: float = 0.005,
        min_precision: float = 0.6,
    ) -> None:
        super().__init__(min_improvement)
        self.min_precision = min_precision

    def filter_batch(
        self,
        candidate_lfs: list["LabelingFunction"],
        scores: list["LFScore"],
        current_f1: float,
        evaluate_fn: object,
        active_lfs: list["LabelingFunction"],
    ) -> list["LabelingFunction"]:
        """Return the subset of candidate LFs that individually improve F1.

        Args:
            candidate_lfs: New LFs to evaluate.
            scores: Corresponding LFScore for each candidate.
            current_f1: Current best F1 before adding candidates.
            evaluate_fn: Callable(lfs) -> (f1, coverage, accuracy).
            active_lfs: Currently active LFs in the registry.

        Returns:
            The subset of candidate_lfs to keep.
        """
        # Filter by minimum precision
        precision_ok = []
        for lf, score in zip(candidate_lfs, scores):
            if score.precision >= self.min_precision:
                precision_ok.append((lf, score))
            else:
                logger.info(
                    "GranularRatchet: dropping %s (precision=%.2f < %.2f)",
                    lf.name,
                    score.precision,
                    self.min_precision,
                )

        # Sort by composite score descending
        precision_ok.sort(key=lambda x: x[1].composite_score, reverse=True)

        # Greedy addition
        kept: list["LabelingFunction"] = []
        running_f1 = current_f1

        for lf, score in precision_ok:
            trial_lfs = active_lfs + kept + [lf]
            trial_f1, _, _ = evaluate_fn(trial_lfs)

            if trial_f1 - running_f1 >= self.min_improvement:
                kept.append(lf)
                running_f1 = trial_f1
                logger.info(
                    "GranularRatchet: keeping %s (F1 %.4f -> %.4f)",
                    lf.name,
                    current_f1,
                    running_f1,
                )

        return kept


class MultiObjectiveRatchet(Ratchet):
    """Multi-objective ratchet scoring F1, coverage, and diversity."""

    def __init__(
        self,
        min_improvement: float = 0.005,
        w_f1: float = 0.6,
        w_coverage: float = 0.2,
        w_diversity: float = 0.2,
    ) -> None:
        super().__init__(min_improvement)
        self.w_f1 = w_f1
        self.w_coverage = w_coverage
        self.w_diversity = w_diversity

    def should_keep_multi(
        self,
        f1_before: float,
        f1_after: float,
        coverage_before: float,
        coverage_after: float,
        n_strategies_before: int,
        n_strategies_after: int,
    ) -> bool:
        """Evaluate whether to keep using multiple objectives."""
        f1_delta = f1_after - f1_before
        coverage_delta = coverage_after - coverage_before
        diversity_delta = (n_strategies_after - n_strategies_before) * 0.1

        composite_delta = (
            self.w_f1 * f1_delta
            + self.w_coverage * coverage_delta
            + self.w_diversity * diversity_delta
        )

        return composite_delta >= self.min_improvement
