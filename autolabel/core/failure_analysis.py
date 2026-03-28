"""Structured error classification for LF debugging."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from autolabel.lf.base import ABSTAIN, LabelingFunction

logger = logging.getLogger(__name__)


@dataclass
class FailureReport:
    """Structured report of an LF's errors on a dev set."""

    lf_name: str
    false_positives: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (text, predicted, actual)
    false_negatives: list[tuple[str, str]] = field(default_factory=list)  # (text, actual_label)
    precision: float = 0.0
    recall: float = 0.0
    fire_rate: float = 0.0
    error_taxonomy: str = "unknown"  # "overly_broad", "too_narrow", "wrong_pattern"

    def summary(self) -> str:
        """Return a human-readable summary of failures."""
        lines = [
            f"LF: {self.lf_name}",
            f"Precision: {self.precision:.2f}, Recall: {self.recall:.2f}, Fire rate: {self.fire_rate:.2f}",
            f"Error type: {self.error_taxonomy}",
        ]

        if self.false_positives:
            lines.append(f"\nFalse Positives ({len(self.false_positives)}):")
            for text, predicted, actual in self.false_positives[:5]:
                lines.append(f"  Text: {text[:100]}...")
                lines.append(f"  Predicted: {predicted}, Actual: {actual}")

        if self.false_negatives:
            lines.append(f"\nFalse Negatives ({len(self.false_negatives)}):")
            for text, actual in self.false_negatives[:5]:
                lines.append(f"  Text: {text[:100]}...")
                lines.append(f"  Should be: {actual}")

        return "\n".join(lines)


class FailureAnalyzer:
    """Classifies LF errors into structured failure reports."""

    def __init__(self, label_space: list[str]) -> None:
        self.label_space = label_space

    def classify_errors(
        self,
        lf: LabelingFunction,
        dev_texts: list[str],
        dev_labels: list[str],
        max_examples: int = 5,
    ) -> FailureReport:
        """Analyze an LF's predictions against dev ground truth.

        Args:
            lf: The labeling function to analyze.
            dev_texts: Development set texts.
            dev_labels: Ground-truth labels.
            max_examples: Max false positive/negative examples to include.

        Returns:
            A structured FailureReport.
        """
        false_positives: list[tuple[str, str, str]] = []
        false_negatives: list[tuple[str, str]] = []
        correct_fires = 0
        total_fires = 0

        for text, true_label in zip(dev_texts, dev_labels):
            result = lf.apply(text)
            if result != ABSTAIN:
                total_fires += 1
                if result == true_label:
                    correct_fires += 1
                else:
                    false_positives.append((text, result, true_label))
            else:
                # Abstained — is this a false negative?
                if true_label == lf.target_label:
                    false_negatives.append((text, true_label))

        # Compute metrics
        precision = correct_fires / total_fires if total_fires > 0 else 0.0
        fire_rate = total_fires / len(dev_texts) if dev_texts else 0.0

        target_count = sum(1 for lb in dev_labels if lb == lf.target_label)
        target_correct = sum(
            1
            for text, lb in zip(dev_texts, dev_labels)
            if lb == lf.target_label and lf.apply(text) == lf.target_label
        )
        recall = target_correct / target_count if target_count > 0 else 0.0

        # Classify error type
        error_taxonomy = self._classify_error_type(precision, recall, fire_rate)

        return FailureReport(
            lf_name=lf.name,
            false_positives=false_positives[:max_examples],
            false_negatives=false_negatives[:max_examples],
            precision=precision,
            recall=recall,
            fire_rate=fire_rate,
            error_taxonomy=error_taxonomy,
        )

    @staticmethod
    def _classify_error_type(precision: float, recall: float, fire_rate: float) -> str:
        """Classify the error pattern."""
        if fire_rate > 0.3 and precision < 0.5:
            return "overly_broad"
        elif fire_rate < 0.05 and recall < 0.1:
            return "too_narrow"
        elif precision < 0.5:
            return "wrong_pattern"
        elif recall < 0.2:
            return "too_narrow"
        else:
            return "acceptable"
