"""Zero-label bootstrap mode — generate pseudo-labels from unlabeled data."""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any

from autolabel.data.dataset import AutoLabelDataset

logger = logging.getLogger(__name__)


class ZeroLabelBootstrap:
    """Generates pseudo-labels for unlabeled data using LLM zero-shot classification.

    Pipeline:
    1. LLM-as-Oracle: classify random texts zero-shot
    2. Self-Consistency Filter: classify K times at different temperatures,
       keep only where all K agree
    3. Use pseudo-labels as ground truth for the main loop
    """

    def __init__(
        self,
        provider: Any,
        label_space: list[str],
        task_description: str,
        sample_size: int = 200,
        consistency_k: int = 3,
        confidence_threshold: float = 0.8,
    ) -> None:
        self.provider = provider
        self.label_space = label_space
        self.task_description = task_description
        self.sample_size = sample_size
        self.consistency_k = consistency_k
        self.confidence_threshold = confidence_threshold

    def generate_pseudo_labels(
        self,
        dataset: AutoLabelDataset,
    ) -> None:
        """Generate pseudo-labels for the dataset in-place.

        Populates ``dataset.pseudo_labels`` and ``dataset.pseudo_confidence``,
        and sets ``dataset.labels`` to the pseudo-labels for compatibility
        with the main loop.
        """
        all_indices = list(range(len(dataset.texts)))
        sample_size = min(self.sample_size, len(all_indices))
        sample_indices = random.sample(all_indices, sample_size)

        logger.info(
            "Bootstrap: classifying %d texts with %d consistency passes",
            sample_size,
            self.consistency_k,
        )

        pseudo_labels: list[str] = [""] * len(dataset.texts)
        pseudo_confidence: list[float] = [0.0] * len(dataset.texts)

        temperatures = self._get_temperatures()

        for idx in sample_indices:
            text = dataset.texts[idx]
            votes: list[str | None] = []

            for temp in temperatures:
                label = self._classify_text(text, temp)
                votes.append(label)

            # Self-consistency: keep only if all K agree
            valid_votes = [v for v in votes if v is not None]
            if len(valid_votes) == self.consistency_k:
                counter = Counter(valid_votes)
                most_common, count = counter.most_common(1)[0]
                confidence = count / self.consistency_k

                if confidence >= self.confidence_threshold:
                    pseudo_labels[idx] = most_common
                    pseudo_confidence[idx] = confidence

        # Fill remaining with most-common pseudo-label
        assigned_labels = [lb for lb in pseudo_labels if lb]
        if assigned_labels:
            fallback = Counter(assigned_labels).most_common(1)[0][0]
        else:
            fallback = self.label_space[0]

        for i in range(len(pseudo_labels)):
            if not pseudo_labels[i]:
                pseudo_labels[i] = fallback
                pseudo_confidence[i] = 0.0

        # Apply to dataset
        dataset.labels = pseudo_labels
        dataset.pseudo_labels = pseudo_labels
        dataset.pseudo_confidence = pseudo_confidence

        # Create splits from sampled indices
        high_conf_indices = [
            idx for idx in sample_indices if pseudo_confidence[idx] >= self.confidence_threshold
        ]

        if len(high_conf_indices) < 10:
            # Not enough high-confidence labels — use all sampled
            high_conf_indices = sample_indices

        # Split high-confidence indices for train/dev/test
        random.shuffle(high_conf_indices)
        n = len(high_conf_indices)
        train_end = int(n * 0.6)
        dev_end = int(n * 0.8)

        dataset.train_indices = high_conf_indices[:train_end]
        dataset.dev_indices = high_conf_indices[train_end:dev_end]
        dataset.test_indices = high_conf_indices[dev_end:]

        # Add remaining (lower-confidence) indices to train
        remaining = [i for i in range(len(dataset.texts)) if i not in high_conf_indices]
        dataset.train_indices.extend(remaining)

        n_assigned = sum(1 for c in pseudo_confidence if c >= self.confidence_threshold)
        logger.info(
            "Bootstrap complete: %d/%d texts assigned high-confidence labels",
            n_assigned,
            sample_size,
        )

    def _classify_text(self, text: str, temperature: float) -> str | None:
        """Classify a single text using zero-shot LLM prompting."""
        labels_str = ", ".join(self.label_space)
        prompt = (
            f"Task: {self.task_description}\n"
            f"Labels: {labels_str}\n\n"
            f"Text: {text}\n\n"
            f"Reply with ONLY the label name, nothing else."
        )

        try:
            response = self.provider.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=64,
            )
            return self._match_label(response.text.strip())
        except Exception as e:
            logger.warning("Bootstrap classification failed: %s", e)
            return None

    def _match_label(self, pred: str) -> str | None:
        """Match LLM output to a valid label."""
        pred_lower = pred.lower().strip()
        for label in self.label_space:
            if label.lower() == pred_lower:
                return label
        for label in self.label_space:
            if label.lower() in pred_lower or pred_lower in label.lower():
                return label
        return None

    def _get_temperatures(self) -> list[float]:
        """Return temperature schedule for consistency passes."""
        if self.consistency_k == 1:
            return [0.0]
        elif self.consistency_k == 2:
            return [0.0, 0.3]
        elif self.consistency_k == 3:
            return [0.0, 0.3, 0.7]
        else:
            step = 0.7 / (self.consistency_k - 1)
            return [round(i * step, 2) for i in range(self.consistency_k)]
