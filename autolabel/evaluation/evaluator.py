"""High-level evaluator that wraps metric computation against a dataset split."""

from __future__ import annotations

from dataclasses import dataclass

from autolabel.data.dataset import AutoLabelDataset
from autolabel.evaluation.metrics import (
    compute_accuracy,
    compute_coverage,
    compute_f1,
    per_class_f1,
)


@dataclass
class EvaluationResult:
    """Container for the outcome of evaluating predictions on a split."""

    f1: float
    accuracy: float
    coverage: float
    per_class_f1: dict[str, float]
    num_correct: int
    num_total: int

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(f1={self.f1:.4f}, accuracy={self.accuracy:.4f}, "
            f"coverage={self.coverage:.4f}, num_correct={self.num_correct}, "
            f"num_total={self.num_total})"
        )


class Evaluator:
    """Evaluates predictions against a ground-truth :class:`AutoLabelDataset`.

    Usage::

        evaluator = Evaluator(dataset)
        result = evaluator.evaluate(predictions, split="dev")
        print(result.f1, result.accuracy)
    """

    def __init__(self, dataset: AutoLabelDataset) -> None:
        self.dataset = dataset

    def _get_split_labels(self, split: str) -> list[str]:
        """Return ground-truth labels for the requested split."""
        if split == "train":
            return self.dataset.train_labels
        elif split == "dev":
            return self.dataset.dev_labels
        elif split == "test":
            return self.dataset.test_labels
        else:
            raise ValueError(
                f"Unknown split {split!r}. Expected one of: 'train', 'dev', 'test'."
            )

    def evaluate(
        self,
        predictions: list[str | None],
        split: str = "dev",
    ) -> EvaluationResult:
        """Score *predictions* against ground truth for the given *split*.

        Parameters
        ----------
        predictions:
            One prediction per example in the split.  ``None`` or ``"ABSTAIN"``
            means the system abstained on that example.
        split:
            Which split to evaluate on (``"train"``, ``"dev"``, or ``"test"``).

        Returns
        -------
        EvaluationResult
        """
        y_true = self._get_split_labels(split)

        if len(predictions) != len(y_true):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) does not match "
                f"number of {split} examples ({len(y_true)})."
            )

        label_space = self.dataset.label_space

        f1 = compute_f1(y_true, predictions, label_space)
        accuracy = compute_accuracy(y_true, predictions)
        coverage = compute_coverage(predictions)
        class_f1 = per_class_f1(y_true, predictions, label_space)

        # Count correct (ignoring abstains)
        num_correct = sum(
            1
            for yt, yp in zip(y_true, predictions)
            if yp is not None and yp != "ABSTAIN" and yt == yp
        )
        num_total = len(y_true)

        return EvaluationResult(
            f1=f1,
            accuracy=accuracy,
            coverage=coverage,
            per_class_f1=class_f1,
            num_correct=num_correct,
            num_total=num_total,
        )
