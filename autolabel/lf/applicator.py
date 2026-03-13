"""Bulk application of labeling functions to produce a label matrix."""

from __future__ import annotations

import logging

import numpy as np

from autolabel.lf.base import ABSTAIN, LabelingFunction

logger = logging.getLogger(__name__)


class LFApplicator:
    """Applies a collection of labeling functions to a corpus of texts.

    The primary output is a *label matrix* suitable for consumption by
    a label model (e.g. Snorkel-style majority vote or learned aggregator).
    """

    @staticmethod
    def apply_lfs(
        lfs: list[LabelingFunction],
        texts: list[str],
        label_space: list[str],
    ) -> np.ndarray:
        """Apply every LF to every text and return an integer label matrix.

        Args:
            lfs: Labeling functions to apply.
            texts: Input texts to label.
            label_space: Ordered list of possible label strings.  An LF
                output is mapped to the index of the matching label in
                this list; outputs not in *label_space* or ``ABSTAIN``
                are encoded as ``-1``.

        Returns:
            A NumPy array of shape ``(len(texts), len(lfs))`` with dtype
            ``int``.  Cell ``(i, j)`` holds the label-space index
            returned by ``lfs[j]`` on ``texts[i]``, or ``-1`` when the
            LF abstained or errored.
        """
        label_to_idx: dict[str, int] = {label: idx for idx, label in enumerate(label_space)}

        n_texts = len(texts)
        n_lfs = len(lfs)
        matrix = np.full((n_texts, n_lfs), -1, dtype=int)

        for j, lf in enumerate(lfs):
            # Pre-compile once so we don't repeat per text
            try:
                if lf._compiled_fn is None:
                    lf.compile()
            except Exception:
                logger.warning("Failed to compile LF %s – marking all ABSTAIN", lf.name)
                continue

            for i, text in enumerate(texts):
                try:
                    result = lf.apply(text)
                    if result != ABSTAIN and result in label_to_idx:
                        matrix[i, j] = label_to_idx[result]
                except Exception:
                    logger.debug(
                        "LF %s raised on text index %d – treating as ABSTAIN",
                        lf.name,
                        i,
                    )

        return matrix
