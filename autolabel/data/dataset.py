"""Core dataset dataclass used throughout AutoLabel."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class AutoLabelDataset:
    """Holds texts, labels, label space, and train/dev/test split indices.

    Every dataset used in AutoLabel is represented as an instance of this
    class.  The ``label_space`` defines the canonical set of possible labels.
    ``labels`` contains the ground-truth label for each text (single-label
    classification).
    """

    name: str
    task_description: str
    label_space: list[str]
    texts: list[str]
    labels: list[str]
    train_indices: list[int] = field(default_factory=list)
    dev_indices: list[int] = field(default_factory=list)
    test_indices: list[int] = field(default_factory=list)

    # Zero-label bootstrap fields (Feature 3)
    pseudo_labels: list[str] = field(default_factory=list)
    pseudo_confidence: list[float] = field(default_factory=list)

    @property
    def has_labels(self) -> bool:
        """True if the dataset has ground-truth labels (not pseudo-labels)."""
        return bool(self.labels) and not bool(self.pseudo_labels)

    @classmethod
    def from_unlabeled(
        cls,
        texts: list[str],
        label_space: list[str],
        task_description: str,
        name: str = "unlabeled",
    ) -> "AutoLabelDataset":
        """Create a dataset from unlabeled texts.

        Labels are left empty; use :class:`ZeroLabelBootstrap` to populate them.
        """
        n = len(texts)
        indices = list(range(n))
        random.shuffle(indices)
        train_end = int(n * 0.6)
        dev_end = int(n * 0.8)

        return cls(
            name=name,
            task_description=task_description,
            label_space=label_space,
            texts=texts,
            labels=[""] * n,
            train_indices=indices[:train_end],
            dev_indices=indices[train_end:dev_end],
            test_indices=indices[dev_end:],
        )

    # -- convenience properties --------------------------------------------------

    @property
    def train_texts(self) -> list[str]:
        return [self.texts[i] for i in self.train_indices]

    @property
    def train_labels(self) -> list[str]:
        return [self.labels[i] for i in self.train_indices]

    @property
    def dev_texts(self) -> list[str]:
        return [self.texts[i] for i in self.dev_indices]

    @property
    def dev_labels(self) -> list[str]:
        return [self.labels[i] for i in self.dev_indices]

    @property
    def test_texts(self) -> list[str]:
        return [self.texts[i] for i in self.test_indices]

    @property
    def test_labels(self) -> list[str]:
        return [self.labels[i] for i in self.test_indices]

    @property
    def num_classes(self) -> int:
        return len(self.label_space)

    # -- dunder helpers ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.texts)

    def __repr__(self) -> str:
        return (
            f"AutoLabelDataset(name={self.name!r}, num_classes={self.num_classes}, "
            f"total={len(self)}, train={len(self.train_indices)}, "
            f"dev={len(self.dev_indices)}, test={len(self.test_indices)})"
        )
