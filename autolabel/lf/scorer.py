"""Per-LF granular scoring for precision, recall, coverage, correlation, and marginal F1."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from autolabel.evaluation.metrics import compute_f1
from autolabel.lf.applicator import LFApplicator
from autolabel.lf.base import LabelingFunction

logger = logging.getLogger(__name__)


@dataclass
class LFScore:
    """Granular quality score for a single labeling function."""

    name: str
    precision: float  # correct fires / total fires
    recall: float  # correct fires / total target examples
    coverage: float  # fire rate on dev set
    correlation: float  # max abs correlation with any active LF
    marginal_f1_delta: float  # leave-one-out F1 contribution
    composite_score: float  # weighted combination

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "precision": self.precision,
            "recall": self.recall,
            "coverage": self.coverage,
            "correlation": self.correlation,
            "marginal_f1_delta": self.marginal_f1_delta,
            "composite_score": self.composite_score,
        }


class LFScorer:
    """Scores individual labeling functions against dev ground truth.

    Uses :class:`LFApplicator` to apply LFs and computes per-LF
    precision, recall, coverage, pairwise correlation, and marginal F1.
    """

    def __init__(
        self,
        label_space: list[str],
        w_precision: float = 0.4,
        w_recall: float = 0.2,
        w_coverage: float = 0.1,
        w_marginal_f1: float = 0.3,
    ) -> None:
        self.label_space = label_space
        self.w_precision = w_precision
        self.w_recall = w_recall
        self.w_coverage = w_coverage
        self.w_marginal_f1 = w_marginal_f1

    def score_lf(
        self,
        lf: LabelingFunction,
        dev_texts: list[str],
        dev_labels: list[str],
        all_lfs: list[LabelingFunction],
        label_model_factory: object,
        num_classes: int,
    ) -> LFScore:
        """Compute a granular score for a single LF.

        Args:
            lf: The labeling function to score.
            dev_texts: Development set texts.
            dev_labels: Ground-truth labels for dev set.
            all_lfs: All currently active LFs (for correlation and marginal F1).
            label_model_factory: Callable returning a label model instance.
            num_classes: Number of classes.
        """
        label_to_idx = {label: i for i, label in enumerate(self.label_space)}

        # Apply this single LF
        single_matrix = LFApplicator.apply_lfs([lf], dev_texts, self.label_space)
        col = single_matrix[:, 0]

        # Precision and recall
        target_idx = label_to_idx.get(lf.target_label, -1)
        fires_mask = col != -1
        n_fires = int(fires_mask.sum())
        encoded_labels = np.array([label_to_idx.get(lb, -1) for lb in dev_labels])

        if n_fires > 0:
            correct_fires = int((col[fires_mask] == encoded_labels[fires_mask]).sum())
            precision = correct_fires / n_fires
        else:
            precision = 0.0

        # Recall: how many of the target class does this LF correctly fire on
        target_mask = encoded_labels == target_idx
        n_target = int(target_mask.sum())
        if n_target > 0:
            correct_target = int(((col == target_idx) & target_mask).sum())
            recall = correct_target / n_target
        else:
            recall = 0.0

        coverage = n_fires / len(dev_texts) if dev_texts else 0.0

        # Correlation with other active LFs
        correlation = self._compute_max_correlation(lf, all_lfs, dev_texts)

        # Marginal F1 delta (leave-one-out)
        marginal_f1_delta = self._compute_marginal_f1(
            lf, all_lfs, dev_texts, dev_labels, label_model_factory, num_classes
        )

        # Composite score
        composite = (
            self.w_precision * precision
            + self.w_recall * recall
            + self.w_coverage * min(coverage * 5, 1.0)  # normalize coverage
            + self.w_marginal_f1 * max(marginal_f1_delta * 10, 0.0)  # scale delta
        )

        return LFScore(
            name=lf.name,
            precision=precision,
            recall=recall,
            coverage=coverage,
            correlation=correlation,
            marginal_f1_delta=marginal_f1_delta,
            composite_score=composite,
        )

    def score_batch(
        self,
        lfs: list[LabelingFunction],
        dev_texts: list[str],
        dev_labels: list[str],
        all_lfs: list[LabelingFunction],
        label_model_factory: object,
        num_classes: int,
    ) -> list[LFScore]:
        """Score a batch of LFs."""
        return [
            self.score_lf(lf, dev_texts, dev_labels, all_lfs, label_model_factory, num_classes)
            for lf in lfs
        ]

    def _compute_max_correlation(
        self,
        lf: LabelingFunction,
        all_lfs: list[LabelingFunction],
        dev_texts: list[str],
    ) -> float:
        """Compute max absolute Pearson correlation with any other active LF."""
        other_lfs = [other for other in all_lfs if other.name != lf.name]
        if not other_lfs:
            return 0.0

        combined = [lf] + other_lfs
        matrix = LFApplicator.apply_lfs(combined, dev_texts, self.label_space)

        target_col = matrix[:, 0].astype(float)
        max_corr = 0.0

        for j in range(1, matrix.shape[1]):
            other_col = matrix[:, j].astype(float)
            # Only correlate where both are non-abstain
            both_voted = (target_col != -1) & (other_col != -1)
            if both_voted.sum() < 3:
                continue
            t = target_col[both_voted]
            o = other_col[both_voted]
            if np.std(t) == 0 or np.std(o) == 0:
                continue
            corr = float(np.abs(np.corrcoef(t, o)[0, 1]))
            max_corr = max(max_corr, corr)

        return max_corr

    def _compute_marginal_f1(
        self,
        lf: LabelingFunction,
        all_lfs: list[LabelingFunction],
        dev_texts: list[str],
        dev_labels: list[str],
        label_model_factory: object,
        num_classes: int,
    ) -> float:
        """Compute F1(all LFs) - F1(all LFs minus this one)."""
        # F1 with all LFs
        f1_all = self._evaluate_lf_set(
            all_lfs, dev_texts, dev_labels, label_model_factory, num_classes
        )

        # F1 without this LF
        without = [other for other in all_lfs if other.name != lf.name]
        if not without:
            return f1_all

        f1_without = self._evaluate_lf_set(
            without, dev_texts, dev_labels, label_model_factory, num_classes
        )

        return f1_all - f1_without

    def _evaluate_lf_set(
        self,
        lfs: list[LabelingFunction],
        dev_texts: list[str],
        dev_labels: list[str],
        label_model_factory: object,
        num_classes: int,
    ) -> float:
        """Evaluate a set of LFs and return F1."""
        if not lfs:
            return 0.0

        label_matrix = LFApplicator.apply_lfs(lfs, dev_texts, self.label_space)
        label_model = label_model_factory()
        label_model.fit(label_matrix, num_classes)
        pred_indices = label_model.predict(label_matrix)

        predictions: list[str | None] = []
        for idx in pred_indices:
            if idx == -1:
                predictions.append(None)
            else:
                predictions.append(self.label_space[idx])

        return compute_f1(dev_labels, predictions, self.label_space)
